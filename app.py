import streamlit as st
import requests
import openai
import opencc
import json
import logging
import io
import time
import hashlib
from pathlib import Path
import pandas as pd

# --- 0. Logging 設定 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

def get_log_stream():
    """取得或建立 session 專屬的 log stream"""
    if 'log_stream' not in st.session_state:
        st.session_state.log_stream = io.StringIO()
    return st.session_state.log_stream

def setup_logger():
    """設定 logger，確保每個 session 有獨立的 handler"""
    log_stream = get_log_stream()

    # 移除舊的 handlers 避免重複
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 建立新的 handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(console_handler)
    logger.addHandler(stream_handler)

SUPPORTED_AUDIO_TYPES = ["mp3", "wav", "m4a", "flac", "ogg"]

MIME_TYPE_MAP = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'm4a': 'audio/mp4',
    'flac': 'audio/flac',
    'ogg': 'audio/ogg',
}

UPLOAD_MAX_SIZE_MB = 500

def get_mime_type(filename):
    """根據檔案副檔名取得對應的 MIME type"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    return MIME_TYPE_MAP.get(ext, 'audio/mpeg')

# --- API 驗證與額度檢查 ---

def safe_int(value, default=0):
    """將數值安全轉為 int，失敗則回傳預設值"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def validate_elevenlabs_key(api_key):
    """驗證 ElevenLabs API Key 並取得帳戶資訊"""
    try:
        url = "https://api.elevenlabs.io/v1/user/subscription"
        headers = {"xi-api-key": api_key}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 401:
            return {"valid": False, "error": "API Key 無效"}
        elif response.status_code != 200:
            return {"valid": False, "error": f"API 錯誤: {response.status_code}"}

        data = response.json()
        character_count = safe_int(data.get("character_count"))
        character_limit = safe_int(data.get("character_limit"))

        return {
            "valid": True,
            "tier": data.get("tier", "unknown"),
            "character_count": character_count,
            "character_limit": character_limit,
            "remaining_characters": max(0, character_limit - character_count),
        }
    except requests.RequestException as e:
        return {"valid": False, "error": f"連線錯誤: {str(e)}"}

def validate_openai_key(api_key):
    """驗證 OpenAI API Key"""
    try:
        client = openai.OpenAI(api_key=api_key)
        # 使用簡單的 models list 來驗證 key
        models = client.models.list()
        return {"valid": True, "models_count": len(list(models))}
    except openai.AuthenticationError:
        return {"valid": False, "error": "API Key 無效"}
    except openai.APIError as e:
        return {"valid": False, "error": f"API 錯誤: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"錯誤: {str(e)}"}

# --- 費用計算 ---

# 模型定價 (每 1M tokens，USD)
# 若未知，設為 None，避免顯示錯誤成本估算
MODEL_PRICING = {
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": None,
}

# ElevenLabs Scribe 定價 ($0.48/hour = $0.008/分鐘)
ELEVENLABS_SCRIBE_PRICE_PER_MINUTE = 0.008

def init_cost_tracker():
    """初始化費用追蹤器"""
    if 'cost_tracker' not in st.session_state:
        st.session_state.cost_tracker = {
            "elevenlabs_minutes": 0.0,
            "elevenlabs_cost": 0.0,
            "openai_input_tokens": 0,
            "openai_output_tokens": 0,
            "openai_cost": 0.0,
            "openai_pricing_available": True,
            "openai_pricing_model": "",
            "total_cost": 0.0,
        }

def reset_cost_tracker():
    """重置費用追蹤器"""
    st.session_state.cost_tracker = {
        "elevenlabs_minutes": 0.0,
        "elevenlabs_cost": 0.0,
        "openai_input_tokens": 0,
        "openai_output_tokens": 0,
        "openai_cost": 0.0,
        "openai_pricing_available": True,
        "openai_pricing_model": "",
        "total_cost": 0.0,
    }

def track_elevenlabs_cost(audio_duration_seconds):
    """追蹤 ElevenLabs 費用"""
    minutes = audio_duration_seconds / 60.0
    cost = minutes * ELEVENLABS_SCRIBE_PRICE_PER_MINUTE
    st.session_state.cost_tracker["elevenlabs_minutes"] += minutes
    st.session_state.cost_tracker["elevenlabs_cost"] += cost
    st.session_state.cost_tracker["total_cost"] += cost
    return cost

def track_openai_cost(model, input_tokens, output_tokens):
    """追蹤 OpenAI 費用"""
    pricing = MODEL_PRICING.get(model)

    st.session_state.cost_tracker["openai_input_tokens"] += input_tokens
    st.session_state.cost_tracker["openai_output_tokens"] += output_tokens

    if not pricing:
        st.session_state.cost_tracker["openai_pricing_available"] = False
        st.session_state.cost_tracker["openai_pricing_model"] = model
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    st.session_state.cost_tracker["openai_cost"] += total_cost
    st.session_state.cost_tracker["total_cost"] += total_cost
    return total_cost

def get_cost_summary():
    """取得費用摘要"""
    tracker = st.session_state.cost_tracker
    return {
        "elevenlabs": {
            "minutes": tracker["elevenlabs_minutes"],
            "cost": tracker["elevenlabs_cost"],
        },
        "openai": {
            "input_tokens": tracker["openai_input_tokens"],
            "output_tokens": tracker["openai_output_tokens"],
            "cost": tracker["openai_cost"],
            "pricing_available": tracker["openai_pricing_available"],
            "pricing_model": tracker["openai_pricing_model"],
        },
        "total_cost": tracker["total_cost"],
    }

# --- 1. 輔助函式：SRT 時間格式化 ---
def format_timestamp(seconds):
    """將秒數轉換為 SRT 格式 (00:00:00,000)"""
    if seconds is None:
        return "00:00:00,000"
    try:
        total_millis = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        return "00:00:00,000"

    if total_millis < 0:
        total_millis = 0

    hours, remainder = divmod(total_millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- 2. ElevenLabs API 呼叫 ---
def transcribe_audio(
    file_obj,
    api_key,
    language_code=None,
    model_id="scribe_v2",
    diarize=False,
    keyterms=None,
    timeout=(30, 1800),
    max_retries=2,
    retry_backoff=5
):
    """
    使用 ElevenLabs Scribe 模型進行轉錄。
    支援 scribe_v1 / scribe_v2，以及 keyterms 和 diarize。
    """
    logger.info(f"Starting transcription with ElevenLabs {model_id}...")
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key
    }
    data = [
        ("model_id", model_id),
        ("tag_audio_events", "false"),
        ("timestamps_granularity", "character"),
    ]
    if language_code and language_code != "auto":
        data.append(("language_code", language_code))
    if diarize:
        data.append(("diarize", "true"))
    if keyterms:
        for term in keyterms[:100]:
            data.append(("keyterms", term[:50]))

    try:
        file_obj.seek(0)
    except Exception:
        pass

    mime_type = getattr(file_obj, "type", None) or get_mime_type(file_obj.name)
    files = {
        "file": (file_obj.name, file_obj, mime_type)
    }
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            try:
                file_obj.seek(0)
            except Exception:
                pass

            response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
            response.raise_for_status()

            logger.info("Transcription successful.")
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                sleep_seconds = retry_backoff * (2 ** attempt)
                logger.warning(f"Upload timeout/connection error. Retrying in {sleep_seconds}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_seconds)
                continue
            break
        except requests.RequestException as e:
            last_error = e
            break

    resp_text = ""
    if getattr(last_error, "response", None) is not None:
        resp_text = last_error.response.text or ""
    error_msg = f"ElevenLabs API Error: {resp_text or str(last_error)}"
    logger.error(error_msg)
    raise Exception(error_msg) from last_error

# --- 3. OpenAI API 呼叫 (斷句) ---

# 定義斷句輸出的 JSON Schema
SEGMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "description": "斷句後的字幕行陣列",
            "items": {
                "type": "string",
                "description": "單行字幕文字"
            }
        }
    },
    "required": ["lines"],
    "additionalProperties": False
}

# 分批處理設定
SEGMENTATION_BATCH_MAX_CHARS = 600
SEGMENTATION_BATCH_DELIMITERS = set("。！？!?.；;")
SEGMENTATION_BATCH_SOFT_DELIMITERS = set("，,、 \t")

# Few-shot 範例
FEW_SHOT_EXAMPLES = {
    "youtube": {
        "input": "今天我想跟大家聊一下關於人工智慧的發展其實最近這幾年AI的進步真的非常快從語音辨識到圖像生成每一個領域都有突破性的變化",
        "output": [
            "今天我想跟大家聊一下",
            "關於人工智慧的發展",
            "其實最近這幾年",
            "AI的進步真的非常快",
            "從語音辨識到圖像生成",
            "每一個領域都有突破性的變化"
        ]
    },
    "tiktok": {
        "input": "今天我想跟大家聊一下關於人工智慧的發展其實最近這幾年AI的進步真的非常快從語音辨識到圖像生成每一個領域都有突破性的變化",
        "output": [
            "今天我想跟大家",
            "聊一下",
            "關於人工智慧",
            "的發展",
            "其實最近這幾年",
            "AI的進步",
            "真的非常快",
            "從語音辨識",
            "到圖像生成",
            "每一個領域都有",
            "突破性的變化"
        ]
    }
}

def parse_lines_from_json(raw_text):
    """解析 JSON 並取得 lines 陣列，失敗回傳 None"""
    try:
        result = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(result, dict):
        return None

    lines = result.get("lines")
    if not isinstance(lines, list):
        return None

    return [str(line) for line in lines]

def split_text_into_batches(text, max_chars=SEGMENTATION_BATCH_MAX_CHARS):
    """將長文按自然斷點切分為多個 batch，保證每段不超過 max_chars"""
    if len(text) <= max_chars:
        return [text]

    batches = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            batches.append(remaining)
            break

        split_pos = -1
        for i in range(min(max_chars, len(remaining)) - 1, max_chars // 2, -1):
            if remaining[i] in SEGMENTATION_BATCH_DELIMITERS:
                split_pos = i + 1
                break

        if split_pos == -1:
            for i in range(min(max_chars, len(remaining)) - 1, max_chars // 2, -1):
                if remaining[i] in SEGMENTATION_BATCH_SOFT_DELIMITERS:
                    split_pos = i + 1
                    break

        if split_pos == -1:
            split_pos = max_chars

        batches.append(remaining[:split_pos])
        remaining = remaining[split_pos:]

    return batches

# 對齊演算法設定
NORMALIZE_IGNORE_CHARS = set(" \t\n\r，。？！：；、,.?!:;\"'()（）[]{}-—－~～《》")
ALIGNMENT_SEARCH_WINDOW = 50
ALIGNMENT_FALLBACK_CHAR_DURATION = 0.25
ALIGNMENT_FALLBACK_CHARS_PER_SEC = 4.0

def _content_length(text):
    """計算忽略標點後的實際內容字數"""
    return sum(1 for c in text if c not in NORMALIZE_IGNORE_CHARS)

def _normalize_text(text):
    """正規化文字用於比對（忽略標點、空白，統一小寫）"""
    return "".join(c for c in text if c not in NORMALIZE_IGNORE_CHARS).lower()

def validate_and_fix_lines(lines, max_chars, original_text, client, model, system_prompt, reasoning_effort=None):
    """後驗證：超長行送回 LLM 二次斷句，驗證字元完整性。回傳 (fixed_lines, extra_usage)"""
    extra_usage = {"input_tokens": 0, "output_tokens": 0}

    fixed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if _content_length(line) <= max_chars:
            fixed_lines.append(line)
            continue

        logger.info(f"Line too long ({_content_length(line)} chars > {max_chars}), re-segmenting with LLM: {line[:30]}...")
        try:
            reseg_lines, reseg_usage = call_llm_segmentation(
                client, model, system_prompt, line, reasoning_effort
            )
            extra_usage["input_tokens"] += reseg_usage["input_tokens"]
            extra_usage["output_tokens"] += reseg_usage["output_tokens"]

            still_too_long = any(_content_length(l) > max_chars for l in reseg_lines)
            if not reseg_lines or still_too_long:
                logger.warning("Re-segmentation still produced long lines, keeping original.")
                fixed_lines.append(line)
            else:
                fixed_lines.extend(reseg_lines)
        except Exception as e:
            logger.warning(f"Re-segmentation failed, keeping original line: {e}")
            fixed_lines.append(line)

    original_norm = _normalize_text(original_text)
    result_norm = _normalize_text("".join(fixed_lines))

    if original_norm != result_norm:
        diff = len(original_norm) - len(result_norm)
        logger.warning(f"Character mismatch after segmentation: original={len(original_norm)}, result={len(result_norm)}, diff={diff}")

    return fixed_lines, extra_usage

def build_segmentation_prompt(max_chars, subtitle_style, segmentation_prompt):
    """建構含 few-shot 範例的斷句 system prompt"""
    if subtitle_style == "tiktok":
        style_hint = f"每行字幕要短，通常 3-8 個字，不超過 {max_chars} 個字。"
    else:
        style_hint = f"每行字幕不超過 {max_chars} 個中文字。"

    example = FEW_SHOT_EXAMPLES[subtitle_style]
    example_output = json.dumps({"lines": example["output"]}, ensure_ascii=False)

    system_prompt = (
        "你是一個專業的字幕編輯員。你的任務是將輸入的逐字稿重新斷句，使其符合字幕閱讀習慣。\n"
        "規則：\n"
        f"1. {style_hint}\n"
        "2. 重要：請嚴格保持輸入文本的原始字元（包括繁簡體），不要進行繁簡轉換，僅進行斷句。絕對不要改寫文字內容，不要刪減字，不要增加字（除了標點符號）。\n"
        "3. 請依照語氣和語意進行換行。\n"
        "4. 如果遇到語氣詞（如：啦、喔、耶），請保留。\n"
        "5. 請以 JSON 格式輸出，將每行字幕放入 lines 陣列中。\n"
        "\n"
        "--- 範例 ---\n"
        f"輸入：{example['input']}\n"
        f"輸出：{example_output}\n"
        "--- 範例結束 ---"
    )

    if segmentation_prompt:
        system_prompt += f"\n額外指令: {segmentation_prompt}"

    return system_prompt

def call_llm_segmentation(client, model, system_prompt, text, reasoning_effort=None):
    """呼叫 LLM 進行單次斷句，含三層 fallback。回傳 (lines, usage)"""
    usage = {"input_tokens": 0, "output_tokens": 0}

    try:
        response_kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "segmentation_output",
                    "strict": True,
                    "schema": SEGMENTATION_SCHEMA
                }
            },
            "temperature": 0,
        }
        if reasoning_effort and model.startswith("gpt-5"):
            response_kwargs["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**response_kwargs)

        if hasattr(response, 'usage') and response.usage:
            usage["input_tokens"] += getattr(response.usage, 'input_tokens', 0)
            usage["output_tokens"] += getattr(response.usage, 'output_tokens', 0)

        lines = parse_lines_from_json(response.output_text)
        if lines is None:
            raise ValueError("Responses API JSON parsing failed")

        return lines, usage

    except Exception as e:
        logger.warning(f"Responses API failed, falling back to Responses JSON-object mode: {str(e)}")

        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                text={"format": {"type": "json_object"}}
            )

            if hasattr(response, 'usage') and response.usage:
                usage["input_tokens"] += getattr(response.usage, 'input_tokens', 0)
                usage["output_tokens"] += getattr(response.usage, 'output_tokens', 0)

            lines = parse_lines_from_json(response.output_text)
            if lines is None:
                raise ValueError("Responses JSON-object parsing failed")

            return lines, usage

        except Exception as e2:
            logger.warning(f"Responses JSON-object mode failed, using Chat plain text fallback: {str(e2)}")

            plain_prompt = system_prompt.replace(
                "5. 請以 JSON 格式輸出，將每行字幕放入 lines 陣列中。",
                "5. 輸出格式為純文字，行與行之間用換行符號分隔。"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": plain_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )

            if response.usage:
                usage["input_tokens"] += response.usage.prompt_tokens
                usage["output_tokens"] += response.usage.completion_tokens

            raw_lines = response.choices[0].message.content.strip().split("\n")
            return [l.strip() for l in raw_lines if l.strip()], usage

def segment_text_with_llm(full_text, api_key, model, max_chars, segmentation_prompt,
                          reasoning_effort="none", subtitle_style="youtube"):
    """請 LLM 將長文字切分為字幕行，含分批處理與後驗證。

    Returns:
        tuple: (segmented_text, usage_dict) - 斷句文字和使用量資訊
    """
    logger.info(f"Starting LLM segmentation. Model: {model}, Max chars: {max_chars}, Style: {subtitle_style}")
    logger.info(f"Reasoning effort: {reasoning_effort}")

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    if not full_text or not full_text.strip():
        return "", total_usage

    client = openai.OpenAI(api_key=api_key)
    system_prompt = build_segmentation_prompt(max_chars, subtitle_style, segmentation_prompt)

    batches = split_text_into_batches(full_text.strip())
    logger.info(f"Split text into {len(batches)} batch(es) for segmentation.")

    all_lines = []
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} chars)...")

        lines, usage = call_llm_segmentation(
            client, model, system_prompt, batch, reasoning_effort
        )

        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]
        all_lines.extend(lines)

    all_lines, fix_usage = validate_and_fix_lines(
        all_lines, max_chars, full_text, client, model, system_prompt, reasoning_effort
    )
    total_usage["input_tokens"] += fix_usage["input_tokens"]
    total_usage["output_tokens"] += fix_usage["output_tokens"]

    track_openai_cost(model, total_usage["input_tokens"], total_usage["output_tokens"])
    logger.info(f"Segmentation complete: {len(all_lines)} lines, tokens: in={total_usage['input_tokens']}, out={total_usage['output_tokens']}")

    return "\n".join(all_lines), total_usage

# --- 4. 核心邏輯：Alignment (對齊) ---
def align_transcript(raw_api_data, llm_segmented_text):
    """
    將 LLM 分好行的文字 (無時間) 與 ElevenLabs 的 Character (有時間) 進行對齊。
    使用強健的正規化錨點匹配 (Robust Normalized Anchor Matching)。
    """
    logger.info("Starting alignment process (Robust Logic)...")

    # Debug: 記錄 API 回傳的結構
    logger.info(f"API response keys: {list(raw_api_data.keys())}")

    # --- 1. 提取原始字元資訊 (Raw Characters Extraction) ---
    raw_chars = []
    
    # 方法 1: 嘗試從 words -> characters 結構取得
    words_data = raw_api_data.get('words', [])
    
    for word in words_data:
        if not isinstance(word, dict):
            continue

        # 檢查是否有 characters 陣列
        if 'characters' in word and word['characters'] is not None:
            for char_obj in word['characters']:
                if (
                    isinstance(char_obj, dict)
                    and 'text' in char_obj
                    and 'start' in char_obj
                    and 'end' in char_obj
                ):
                    raw_chars.append(char_obj)
        # 備用：如果沒有 characters，但 word 本身有時間資訊
        elif 'text' in word and 'start' in word and 'end' in word:
            word_text = word.get('text', '')
            try:
                word_start = float(word.get('start', 0) or 0)
                word_end = float(word.get('end', 0) or 0)
            except (TypeError, ValueError):
                continue
                
            if word_end < word_start:
                word_start, word_end = word_end, word_start

            if word_text and len(word_text) > 0:
                duration_per_char = (word_end - word_start) / len(word_text) if len(word_text) > 0 else 0
                for idx, char in enumerate(word_text):
                    raw_chars.append({
                        'text': char,
                        'start': word_start + idx * duration_per_char,
                        'end': word_start + (idx + 1) * duration_per_char if idx < len(word_text) - 1 else word_end
                    })

    logger.info(f"Total raw characters extracted: {len(raw_chars)}")

    # 如果無法取得字元級時間戳，使用備用方案：根據音訊長度估算
    if len(raw_chars) == 0:
        logger.warning("No character-level timestamps available. Using fallback time estimation.")
        lines = llm_segmented_text.split('\n')
        srt_output = []
        
        audio_duration = raw_api_data.get('audio_duration', None)
        if audio_duration is None:
            total_text_len = sum(len(line.strip()) for line in lines if line.strip())
            audio_duration = total_text_len / ALIGNMENT_FALLBACK_CHARS_PER_SEC if total_text_len > 0 else 60.0
        
        total_chars_count = sum(len(line.strip()) for line in lines if line.strip())
        char_duration = audio_duration / total_chars_count if total_chars_count > 0 else ALIGNMENT_FALLBACK_CHAR_DURATION

        current_time = 0.0
        line_index = 0
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            line_index += 1
            line_duration = len(clean_line) * char_duration
            srt_output.append({
                "index": line_index,
                "start": format_timestamp(current_time),
                "end": format_timestamp(current_time + line_duration),
                "text": clean_line
            })
            current_time += line_duration
            
        return srt_output, total_chars_count, total_chars_count

    # --- 2. 預處理：正規化原始序列 ---
    searchable_raw = []
    
    for rc in raw_chars:
        c = rc['text']
        if c not in NORMALIZE_IGNORE_CHARS:
            searchable_raw.append({
                'char': c.lower(),
                'start': rc['start'],
                'end': rc['end']
            })
    
    # --- 3. 對齊邏輯 (Robust Anchor Matching) ---
    lines = llm_segmented_text.split('\n')
    srt_output = []
    
    curr_search_idx = 0
    total_raw_len = len(searchable_raw)
    matched_count = 0
    total_llm_chars = 0
    last_valid_end = 0.0

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        line_chars = [c.lower() for c in clean_line if c not in NORMALIZE_IGNORE_CHARS]
        total_llm_chars += len(line_chars)
        
        if not line_chars:
            continue

        line_start_time = None
        line_end_time = None
        
        temp_idx = curr_search_idx
        line_matches = 0
        
        first_match_start = None
        last_match_end = None

        for lc in line_chars:
            # 視窗搜尋 (Window Search)
            search_window = ALIGNMENT_SEARCH_WINDOW
            found_at = -1
            
            for offset in range(search_window):
                if temp_idx + offset >= total_raw_len:
                    break
                
                if searchable_raw[temp_idx + offset]['char'] == lc:
                    found_at = temp_idx + offset
                    break
            
            if found_at != -1:
                if first_match_start is None:
                    first_match_start = searchable_raw[found_at]['start']
                last_match_end = searchable_raw[found_at]['end']
                
                temp_idx = found_at + 1
                line_matches += 1
                matched_count += 1
        
        if line_matches > 0:
            line_start_time = first_match_start
            line_end_time = last_match_end
            
            # 更新全域搜尋指標
            curr_search_idx = temp_idx
            last_valid_end = line_end_time
        else:
            # 該行完全未匹配 (Fallback)
            est_duration = len(line_chars) * ALIGNMENT_FALLBACK_CHAR_DURATION
            line_start_time = last_valid_end
            line_end_time = last_valid_end + est_duration
            last_valid_end = line_end_time

        srt_output.append({
            "index": len(srt_output) + 1,
            "start": format_timestamp(line_start_time),
            "end": format_timestamp(line_end_time),
            "text": clean_line
        })

    logger.info(f"Alignment completed. Matched {matched_count}/{total_llm_chars} characters.")
    return srt_output, matched_count, total_llm_chars

def parse_keyword_rules(raw_text):
    """解析 keyword 修正與保留詞"""
    replacements = []
    keep_terms = []

    if not raw_text:
        return replacements, keep_terms

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        separator = None
        for sep in ("=>", "->", "="):
            if sep in line:
                separator = sep
                break

        if separator:
            src, dst = line.split(separator, 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                replacements.append((src, dst))
        else:
            keep_terms.append(line)

    return replacements, keep_terms

def apply_replacements_to_text(text, replacements):
    if not replacements:
        return text

    updated = text
    for src, dst in replacements:
        if src:
            updated = updated.replace(src, dst)
    return updated

def apply_replacements_to_lines(lines, replacements):
    if not replacements:
        return lines
    return [apply_replacements_to_text(line, replacements) for line in lines]

def set_srt_texts(srt_data, lines):
    if len(srt_data) != len(lines):
        logger.warning(f"SRT line count mismatch: srt={len(srt_data)}, lines={len(lines)}. Updating available lines only.")
        st.warning(f"⚠️ 校正/翻譯後行數不一致 (原 {len(srt_data)} 行 → {len(lines)} 行)，部分字幕可能未更新。")

    for item, line in zip(srt_data, lines):
        item["text"] = line
    return srt_data

def split_lines_into_batches(lines, max_chars=3000, max_lines=40):
    batches = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        if current and (len(current) >= max_lines or current_len + line_len > max_chars):
            batches.append(current)
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        batches.append(current)
    return batches

def parse_json_data_list(raw_text):
    try:
        result = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(result, dict):
        return None

    data = result.get("data")
    if not isinstance(data, list):
        return None

    return [str(item) for item in data]

def build_glossary_instruction(replacements, keep_terms):
    lines = []
    if replacements:
        lines.append("請使用以下對應詞彙：")
        for src, dst in replacements:
            lines.append(f"{src} -> {dst}")
    if keep_terms:
        lines.append("請原樣保留以下詞彙：")
        lines.extend(keep_terms)
    return "\n".join(lines)

def llm_transform_lines(lines, api_key, model, system_prompt, temperature=0.2):
    if not lines:
        return lines, {"input_tokens": 0, "output_tokens": 0}

    client = openai.OpenAI(api_key=api_key)
    batches = split_lines_into_batches(lines)
    output_lines = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for batch in batches:
        payload = json.dumps({"data": batch}, ensure_ascii=False)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            if response.usage:
                usage["input_tokens"] += response.usage.prompt_tokens
                usage["output_tokens"] += response.usage.completion_tokens

            parsed = parse_json_data_list(response.choices[0].message.content)
            if not parsed or len(parsed) != len(batch):
                logger.warning("LLM output length mismatch; using original batch.")
                output_lines.extend(batch)
                continue

            output_lines.extend(parsed)

        except Exception as e:
            logger.error(f"LLM batch failed: {str(e)}")
            output_lines.extend(batch)

    if usage["input_tokens"] or usage["output_tokens"]:
        track_openai_cost(model, usage["input_tokens"], usage["output_tokens"])

    return output_lines, usage

def correct_lines_with_llm(lines, api_key, model, output_style, subtitle_style, replacements, keep_terms):
    style_hint = "忠實呈現：只修正明顯錯字，不改寫語句" if output_style == "faithful" else "通順自然：可小幅調整語序但不增刪資訊"
    length_hint = "保持短句、節奏快" if subtitle_style == "tiktok" else "保持字幕閱讀舒適"
    glossary = build_glossary_instruction(replacements, keep_terms)

    system_prompt = (
        "你是字幕校對員，請修正轉錄錯字與聽錯詞。\n"
        f"輸出風格：{style_hint}\n"
        f"字幕風格：{length_hint}\n"
        "規則：\n"
        "1. 每行獨立處理，不合併、不拆行。\n"
        "2. 保留原意，不增加或刪減資訊。\n"
        "3. 僅輸出 JSON 格式：{\"data\": [...]}，長度需與輸入一致。\n"
    )
    if glossary:
        system_prompt += f"\n{glossary}"

    return llm_transform_lines(lines, api_key, model, system_prompt, temperature=0.1)

def translate_lines_with_llm(lines, api_key, model, target_language, output_style, subtitle_style, replacements, keep_terms):
    style_hint = "忠實呈現：盡量直譯，保留語氣" if output_style == "faithful" else "通順自然：可適度潤飾但不改變意思"
    length_hint = "保持短句、節奏快" if subtitle_style == "tiktok" else "保持字幕閱讀舒適"
    glossary = build_glossary_instruction(replacements, keep_terms)

    system_prompt = (
        f"你是字幕翻譯員，請將每行字幕翻譯成{target_language}。\n"
        f"輸出風格：{style_hint}\n"
        f"字幕風格：{length_hint}\n"
        "規則：\n"
        "1. 每行獨立翻譯，不合併、不拆行。\n"
        "2. 保留原意，不增加或刪減資訊。\n"
        "3. 僅輸出 JSON 格式：{\"data\": [...]}，長度需與輸入一致。\n"
    )
    if glossary:
        system_prompt += f"\n{glossary}"

    return llm_transform_lines(lines, api_key, model, system_prompt, temperature=0.2)

def convert_lines_to_traditional(lines):
    """使用 OpenCC 將字幕轉換為台灣繁體中文"""
    try:
        converter = opencc.OpenCC('s2twp')
        return [converter.convert(line) for line in lines]
    except Exception as e:
        logger.error(f"OpenCC conversion error: {e}")
        return lines

def clean_subtitle_text(text):
    """清理字幕文字，移除不需要的標點符號"""
    if not text:
        return text

    # 移除開頭的 - 或 —
    text = text.lstrip('-—－')

    # 移除結尾的標點符號（逗號、句號、頓號、破折號等，保留問號和感歎號）
    trailing_punctuation = '，。、,.-—－;；:：'
    text = text.rstrip(trailing_punctuation)

    # 移除多餘的空白
    text = text.strip()

    return text

def generate_srt_string(srt_data, clean_text=True):
    parts = []
    for item in srt_data:
        cleaned_text = clean_subtitle_text(item['text']) if clean_text else item['text']
        parts.append(f"{item['index']}\n{item['start']} --> {item['end']}\n{cleaned_text}\n")
    return "\n".join(parts) + "\n" if parts else ""

# --- 5. Streamlit UI ---
st.set_page_config(page_title="AI 字幕生成器 (ElevenLabs + OpenAI)", layout="wide")

st.title("🎬 AI 字幕生成器 (Word-Level Timestamp)")
st.markdown("結合 **ElevenLabs Scribe** 的精準時間戳記與 **OpenAI** 的語意斷句能力。")

# 初始化費用追蹤器
init_cost_tracker()

# Sidebar: 設定
with st.sidebar:
    st.header("🔑 API Keys")
    el_key = st.text_input("ElevenLabs API Key", type="password")
    oa_key = st.text_input("OpenAI API Key", type="password")

    # API 驗證按鈕
    if el_key or oa_key:
        if st.button("🔍 驗證 API Keys"):
            with st.spinner("驗證中..."):
                # 驗證 ElevenLabs
                if el_key:
                    el_result = validate_elevenlabs_key(el_key)
                    if el_result["valid"]:
                        st.success(f"✅ ElevenLabs: {el_result['tier']}")
                        st.caption(f"剩餘額度: {el_result['remaining_characters']:,} 字元")
                    else:
                        st.error(f"❌ ElevenLabs: {el_result['error']}")

                # 驗證 OpenAI
                if oa_key:
                    oa_result = validate_openai_key(oa_key)
                    if oa_result["valid"]:
                        st.success("✅ OpenAI: API Key 有效")
                    else:
                        st.error(f"❌ OpenAI: {oa_result['error']}")

    st.header("⚙️ 參數設定")

    # 字幕風格選擇
    style_options = {
        "YouTube (完整語句)": "youtube",
        "TikTok (短句快節奏)": "tiktok",
    }
    selected_style = st.selectbox("字幕風格", list(style_options.keys()), index=0)
    subtitle_style = style_options[selected_style]

    # 根據風格調整預設字數
    default_chars = 8 if subtitle_style == "tiktok" else 16
    max_range = 15 if subtitle_style == "tiktok" else 30
    min_range = 3 if subtitle_style == "tiktok" else 10

    max_chars = st.slider("每行最大字數", min_range, max_range, default_chars)

    # 模型選擇（預設 GPT-4.1）
    model_options = {
        "GPT-4.1 (預設)": "gpt-4.1",
        "GPT-4.1 mini": "gpt-4.1-mini",
        "GPT-5.2": "gpt-5.2",
        "GPT-5-mini": "gpt-5-mini",
    }
    selected_model = st.selectbox("OpenAI Model", list(model_options.keys()), index=0)
    model_choice = model_options[selected_model]

    # 語言選擇
    language_options = {
        "自動偵測": "auto",
        "中文": "zho",
        "英文": "eng",
        "日文": "jpn",
        "韓文": "kor",
        "粵語": "yue",
    }
    selected_lang = st.selectbox("音訊語言", list(language_options.keys()))
    language_code = language_options[selected_lang]

    # ElevenLabs 模型選擇
    scribe_options = {
        "Scribe v2 (推薦)": "scribe_v2",
        "Scribe v1": "scribe_v1",
    }
    selected_scribe = st.selectbox("轉錄模型", list(scribe_options.keys()), index=0)
    scribe_model = scribe_options[selected_scribe]

    enable_diarize = st.checkbox("說話者辨識 (Diarize)", value=False, help="標註音訊中不同說話者，適合多人對話場景。")

    reasoning_effort = None
    with st.expander("進階設定"):
        custom_prompt = st.text_area("給斷句 LLM 的額外指令", value="保留語氣詞。")
        clean_punctuation = st.checkbox("清理字幕標點", value=True, help="移除行首破折號與行尾逗號/句號等。")
        show_debug = st.checkbox("顯示調試資訊", value=False, help="顯示 ElevenLabs API 原始回應。")

        st.markdown("---")
        st.markdown("**文字校正 / 翻譯**")

        enable_correction = st.checkbox("字幕校正（修正錯字/順句）", value=False)

        output_style_options = {
            "忠實呈現": "faithful",
            "通順自然": "fluent",
        }
        selected_output_style = st.selectbox("輸出風格", list(output_style_options.keys()), index=0)
        output_style = output_style_options[selected_output_style]

        translation_options = {
            "不翻譯（原文）": "source",
            "繁體中文": "繁體中文",
            "英文": "英文",
            "日文": "日文",
            "馬來文": "馬來文",
        }
        selected_translation = st.selectbox("翻譯語言", list(translation_options.keys()), index=0)
        target_language = translation_options[selected_translation]

        use_opencc = False
        if target_language == "繁體中文":
            use_opencc = st.checkbox("僅做繁簡轉換 (OpenCC)", value=True, help="僅適用中文來源，非翻譯。")

        keyword_rules = st.text_area(
            "Keyword 修正 / 詞彙表",
            value="",
            help="一行一組：原詞=修正詞 或 原詞=>目標詞；沒有等號視為強制保留詞。",
            placeholder="王小明=小明\nOpenAI=>OpenAI\nHokage"
        )

        st.markdown("---")
        st.markdown("**上傳與重試**")
        connect_timeout = st.number_input("ElevenLabs 連線逾時 (秒)", min_value=5, max_value=300, value=30, step=5)
        read_timeout = st.number_input("ElevenLabs 回應逾時 (秒)", min_value=300, max_value=7200, value=1800, step=60)
        retry_count = st.number_input("上傳重試次數", min_value=0, max_value=5, value=2, step=1)
        retry_backoff = st.number_input("重試等待秒數", min_value=0, max_value=60, value=5, step=1)

        st.markdown("---")
        st.markdown("**GPT-5 專屬設定**")

        if model_choice.startswith("gpt-5"):
            # 推理強度 (Reasoning Effort) - 根據模型動態調整選項
            # - gpt-5.2: none, low, medium, high, xhigh
            # - gpt-5-mini: minimal, low, medium, high
            if model_choice == "gpt-5.2":
                reasoning_options = {
                    "none (最快)": "none",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh (最深度)": "xhigh",
                }
                help_text = "控制模型推理深度。none 最快，xhigh 最深度思考。"
            else:  # gpt-5-mini
                reasoning_options = {
                    "minimal (最快)": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }
                help_text = "控制模型推理深度。minimal 最快，high 最深度思考。"

            selected_reasoning = st.selectbox(
                "推理強度 (Reasoning Effort)",
                list(reasoning_options.keys()),
                index=0,
                help=help_text
            )
            reasoning_effort = reasoning_options[selected_reasoning]
        else:
            st.caption("目前模型不支援 reasoning 設定。")

# Main Area
uploaded_file = st.file_uploader("上傳音訊檔案 (mp3, wav, m4a, flac, ogg)", type=SUPPORTED_AUDIO_TYPES)

# 初始化 session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'cached_transcript' not in st.session_state:
    st.session_state.cached_transcript = None
if 'cached_file_key' not in st.session_state:
    st.session_state.cached_file_key = None

def _file_cache_key(f):
    """產生檔案快取 key（名稱+大小+內容雜湊）以避免誤命中"""
    try:
        digest = hashlib.sha256(f.getbuffer()).hexdigest()[:16]
    except Exception:
        digest = "nohash"
    return f"{f.name}_{f.size}_{digest}"

def _run_pipeline(raw_transcript, uploaded_file, skip_transcribe=False):
    """執行 Step 2-4 的共用流程，回傳是否成功"""
    setup_logger()
    log_stream = get_log_stream()
    log_stream.truncate(0)
    log_stream.seek(0)

    st.session_state.result = None
    reset_cost_tracker()

    status = st.status("正在處理中...", expanded=True)
    error_occurred = False

    try:
        full_text = raw_transcript.get('text', '')

        if not skip_transcribe:
            audio_duration = raw_transcript.get('audio_duration', 0)
            if audio_duration:
                el_cost = track_elevenlabs_cost(audio_duration)
                logger.info(f"ElevenLabs cost: ${el_cost:.4f} for {audio_duration:.2f}s audio")

        if not full_text or not full_text.strip():
            status.update(label="❌ 轉錄結果為空", state="error")
            st.error("❌ 轉錄結果為空白，可能是靜音檔案或不支援的格式。請確認音訊內容後重試。")
            return

        status.write(f"📝 轉錄文字：共 {len(full_text)} 個字。")

        # Step 2: LLM Segmentation
        status.write(f"🧠 正在呼叫 {model_choice} 進行語意斷句 ({subtitle_style} 風格)...")
        segmented_text, seg_usage = segment_text_with_llm(
            full_text, oa_key, model_choice, max_chars, custom_prompt,
            reasoning_effort=reasoning_effort,
            subtitle_style=subtitle_style
        )
        status.write("✅ 斷句完成！")

        # Step 3: Alignment
        status.write("🔗 正在進行時間軸對齊 (Word/Char Level Alignment)...")
        srt_data, matched_cnt, total_cnt = align_transcript(raw_transcript, segmented_text)

        match_rate = (matched_cnt / total_cnt * 100) if total_cnt > 0 else 0
        status.write(f"📊 對齊匹配率: {match_rate:.2f}% ({matched_cnt}/{total_cnt})")
        logger.info(f"Match rate: {match_rate:.2f}%")

        low_match_rate = match_rate < 80

        # Step 4: Keyword 修正 / 校正 / 翻譯
        srt_lines = [item["text"] for item in srt_data]
        replacements, keep_terms = parse_keyword_rules(keyword_rules)

        if replacements:
            srt_lines = apply_replacements_to_lines(srt_lines, replacements)

        if enable_correction and srt_lines:
            status.write("🧹 正在校正字幕文字...")
            srt_lines, _ = correct_lines_with_llm(
                srt_lines, oa_key, model_choice, output_style,
                subtitle_style, replacements, keep_terms
            )
            status.write("✅ 字幕校正完成！")

        if target_language != "source" and srt_lines:
            if target_language == "繁體中文" and use_opencc:
                status.write("🇨🇳->🇹🇼 正在轉換為繁體中文 (OpenCC)...")
                srt_lines = convert_lines_to_traditional(srt_lines)
                status.write("✅ 繁體轉換完成！")
            else:
                status.write(f"🌐 正在翻譯字幕為 {target_language}...")
                srt_lines, _ = translate_lines_with_llm(
                    srt_lines, oa_key, model_choice, target_language,
                    output_style, subtitle_style, replacements, keep_terms
                )
                status.write("✅ 翻譯完成！")

        srt_data = set_srt_texts(srt_data, srt_lines)
        srt_string = generate_srt_string(srt_data, clean_text=clean_punctuation)
        cost_summary = get_cost_summary()

        status.update(label="🎉 任務完成！", state="complete", expanded=False)

        st.session_state.result = {
            'full_text': full_text,
            'segmented_text': segmented_text,
            'srt_string': srt_string,
            'srt_data': srt_data,
            'low_match_rate': low_match_rate,
            'filename': uploaded_file.name,
            'raw_api_response': raw_transcript,
            'cost_summary': cost_summary,
        }

    except Exception as e:
        error_occurred = True
        status.update(label="❌ 發生錯誤", state="error")
        st.error(f"Error Log: {str(e)}")
        logger.error(f"Critical error: {str(e)}")

    with st.expander("📝 執行日誌 (Logs)", expanded=error_occurred):
        st.code(get_log_stream().getvalue())

if uploaded_file and el_key and oa_key:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    file_ext = uploaded_file.name.rsplit('.', 1)[-1].upper() if '.' in uploaded_file.name else '?'

    # 檔案資訊顯示
    st.caption(f"📁 **{uploaded_file.name}** — {file_size_mb:.1f} MB · {file_ext}")

    if file_size_mb > UPLOAD_MAX_SIZE_MB:
        st.error(f"❌ 檔案過大 ({file_size_mb:.1f} MB)，上限為 {UPLOAD_MAX_SIZE_MB} MB。")
    else:
        # 檢查是否有同一檔案的快取轉錄
        current_key = _file_cache_key(uploaded_file)
        has_cache = (
            st.session_state.cached_transcript is not None
            and st.session_state.cached_file_key == current_key
        )

        if has_cache:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_full = st.button("🔄 重新轉錄 + 生成字幕", use_container_width=True)
            with col_btn2:
                btn_reseg = st.button("✂️ 重新斷句（使用快取轉錄）", use_container_width=True,
                                      help="跳過 ElevenLabs 轉錄，直接用上次的轉錄結果重新斷句。省時省錢。")
        else:
            btn_full = st.button("開始生成字幕", use_container_width=True)
            btn_reseg = False

        if btn_full:
            uploaded_file.seek(0)
            _, api_keep_terms = parse_keyword_rules(keyword_rules)
            est_minutes = max(1, file_size_mb * 0.5)
            scribe_label = scribe_model.replace("_", " ").title()
            with st.spinner(f"🎧 正在上傳至 ElevenLabs 進行轉錄 ({scribe_label})... 預估需要 {est_minutes:.0f}-{est_minutes * 2:.0f} 分鐘"):
                raw_transcript = transcribe_audio(
                    uploaded_file, el_key, language_code,
                    model_id=scribe_model,
                    diarize=enable_diarize,
                    keyterms=api_keep_terms if api_keep_terms else None,
                    timeout=(connect_timeout, read_timeout),
                    max_retries=int(retry_count),
                    retry_backoff=int(retry_backoff)
                )
            st.session_state.cached_transcript = raw_transcript
            st.session_state.cached_file_key = current_key
            _run_pipeline(raw_transcript, uploaded_file, skip_transcribe=False)

        elif btn_reseg:
            _run_pipeline(st.session_state.cached_transcript, uploaded_file, skip_transcribe=True)

    # 顯示結果
    if st.session_state.result:
        result = st.session_state.result

        st.markdown("---")

        # 費用摘要
        if 'cost_summary' in result:
            cost = result['cost_summary']
            openai_line = (
                f"- OpenAI: {cost['openai']['input_tokens']:,} input + {cost['openai']['output_tokens']:,} output tokens = ${cost['openai']['cost']:.4f}"
                if cost['openai']['pricing_available']
                else f"- OpenAI: {cost['openai']['input_tokens']:,} input + {cost['openai']['output_tokens']:,} output tokens = N/A（未設定 {cost['openai']['pricing_model']} 定價）"
            )
            total_line = f"- **總計: ${cost['total_cost']:.4f}**" if cost['openai']['pricing_available'] else "- **總計: N/A**"

            st.info(
                "💰 **費用估算**\n\n"
                f"- ElevenLabs: {cost['elevenlabs']['minutes']:.2f} 分鐘 = ${cost['elevenlabs']['cost']:.4f}\n"
                f"{openai_line}\n"
                f"{total_line}"
            )

        if result.get('low_match_rate'):
            st.warning("⚠️ 匹配率較低，可能是因為斷句時文字被修改了，或繁簡不一致。")

        with st.expander("查看原始轉錄文字"):
            st.text(result['full_text'])

        if show_debug:
            with st.expander("🔧 調試：ElevenLabs API 原始回應"):
                api_response = result.get('raw_api_response', {})
                st.write(f"**API 回應 keys:** {list(api_response.keys())}")
                if 'words' in api_response:
                    words = api_response['words']
                    st.write(f"**Words 數量:** {len(words)}")
                    if words:
                        st.write(f"**第一個 word 結構:** {words[0]}")
                st.json(api_response)

        # SRT 表格預覽
        st.subheader("字幕預覽")
        if result.get('srt_data'):
            df = pd.DataFrame([
                {"#": item["index"], "開始": item["start"], "結束": item["end"], "字幕": item["text"]}
                for item in result['srt_data']
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.text_area("SRT", result['srt_string'], height=300)

        with st.expander("LLM 斷句結果"):
            st.text_area("Segmented", result['segmented_text'], height=200, label_visibility="collapsed")

        with st.expander("SRT 原始文字"):
            st.text_area("SRT Raw", result['srt_string'], height=200, label_visibility="collapsed")

        # Download Button
        st.download_button(
            label="📥 下載 .srt 字幕檔",
            data=result['srt_string'],
            file_name=f"{Path(result['filename']).stem}.srt",
            mime="text/plain",
            use_container_width=True
        )

elif not (el_key and oa_key):
    st.info("👈 請先在左側輸入 API Keys")
