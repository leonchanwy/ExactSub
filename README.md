# ExactSub

AI-powered subtitle generator that combines **ElevenLabs Scribe** for precise character-level timestamps with **OpenAI LLMs** for intelligent sentence segmentation — producing accurately timed SRT subtitles from audio files.

[English](#why-exactsub) | [中文](#中文說明)

## Why ExactSub?

Traditional speech-to-text tools either give you a wall of text with no timing, or break lines at awkward points. ExactSub solves this by splitting the work:

1. **ElevenLabs Scribe v1** handles transcription with character-level timestamps
2. **OpenAI GPT** handles semantic segmentation — breaking text into natural, readable subtitle lines
3. A custom **alignment algorithm** merges the two, mapping each segmented line back to precise timestamps

The result: subtitles that are both **accurately timed** and **naturally segmented**.

## Features

- **Accurate timestamps** — Character-level alignment from ElevenLabs Scribe v1
- **Semantic segmentation** — LLM-powered line breaking that respects meaning and reading flow
- **Multiple subtitle styles** — YouTube (full sentences, 10–30 chars/line) or TikTok (short bursts, 3–8 chars/line)
- **Multi-language support** — Chinese, English, Japanese, Korean, Cantonese, or auto-detect
- **Text correction** — Optional LLM pass to fix transcription errors
- **Translation** — Translate subtitles into Traditional Chinese, English, Japanese, or Malay
- **Simplified → Traditional Chinese** — Lightweight OpenCC conversion (no LLM needed)
- **Keyword rules** — Define glossary replacements (e.g., `OpenAI=>OpenAI`) or terms to preserve
- **Punctuation cleaning** — Remove leading dashes, trailing commas/periods from subtitle lines
- **Cost tracking** — Real-time cost estimation for both ElevenLabs and OpenAI API usage
- **Multiple models** — GPT-4.1, GPT-4.1 mini, GPT-5.2, GPT-5-mini with configurable reasoning effort

## How It Works

```
Audio File (mp3/wav/m4a/flac/ogg)
        │
        ▼
┌─────────────────────┐
│  ElevenLabs Scribe   │  → Transcription with character-level timestamps
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  OpenAI LLM          │  → Semantic segmentation into subtitle lines
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Alignment Engine    │  → Maps each line to precise start/end times
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-Processing     │  → Correction, translation, OpenCC, cleanup
└─────────┬───────────┘
          │
          ▼
      SRT File
```

## Prerequisites

- Python 3.9+
- An [ElevenLabs](https://elevenlabs.io/) API key
- An [OpenAI](https://platform.openai.com/) API key

## Installation

```bash
git clone https://github.com/leonchanwy/ExactSub.git
cd ExactSub
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default: `http://localhost:8501`).

### Step-by-step

1. **Enter API keys** in the sidebar (ElevenLabs + OpenAI)
2. **Configure settings:**
   - Choose subtitle style (YouTube / TikTok)
   - Set max characters per line
   - Select OpenAI model
   - Choose audio language (or auto-detect)
3. **Upload an audio file** (mp3, wav, m4a, flac, or ogg)
4. **Click "開始生成字幕"** (Start generating subtitles)
5. **Download** the generated `.srt` file

### Advanced Options

| Option | Description |
|--------|-------------|
| Custom segmentation prompt | Additional instructions for the LLM segmenter |
| Punctuation cleaning | Strip leading dashes and trailing punctuation |
| Text correction | LLM pass to fix transcription typos |
| Translation | Translate to another language via LLM |
| OpenCC conversion | Lightweight Simplified → Traditional Chinese (no LLM) |
| Keyword rules | `source=>target` replacements or terms to preserve |
| Reasoning effort (GPT-5) | Control thinking depth: none → xhigh |
| Retry settings | Configure upload timeout and retry behavior |

### Keyword Rules Format

```
# Replacement rules (one per line)
王小明=>小明
OpenAI=>OpenAI

# Preserve terms (no arrow, kept as-is)
Hokage
```

## Cost Estimation

The app tracks API costs in real time:

| Service | Pricing |
|---------|---------|
| ElevenLabs Scribe | $0.008 / minute of audio |
| OpenAI GPT-4.1 | $2.00 / 1M input tokens, $8.00 / 1M output tokens |
| OpenAI GPT-5.2 | $1.75 / 1M input tokens, $14.00 / 1M output tokens |
| OpenAI GPT-5-mini | $0.25 / 1M input tokens, $2.00 / 1M output tokens |

## License

MIT

---

# 中文說明

結合 **ElevenLabs Scribe** 的字元級精準時間戳與 **OpenAI LLM** 的語意斷句能力，從音訊檔案自動生成高品質 SRT 字幕。

## 為什麼選擇 ExactSub？

傳統語音轉文字工具要嘛只給你一整段沒有時間軸的文字，要嘛在不自然的地方斷行。ExactSub 把工作分成兩步：

1. **ElevenLabs Scribe v1** 負責轉錄，提供字元級時間戳
2. **OpenAI GPT** 負責語意斷句，根據語意和語氣自然換行
3. 自製的**對齊演算法**將兩者合併，為每行字幕匹配精準的起訖時間

最終結果：**時間軸精準** 且 **斷句自然** 的字幕。

## 功能特色

- **精準時間戳** — 基於 ElevenLabs Scribe v1 的字元級對齊
- **語意斷句** — LLM 驅動的智慧換行，尊重語意和閱讀節奏
- **多種字幕風格** — YouTube（完整語句，每行 10–30 字）或 TikTok（短句快節奏，每行 3–8 字）
- **多語言支援** — 中文、英文、日文、韓文、粵語，或自動偵測
- **文字校正** — 可選的 LLM 校正，修正轉錄錯字
- **翻譯** — 可翻譯為繁體中文、英文、日文或馬來文
- **繁簡轉換** — 使用 OpenCC 輕量轉換，無需呼叫 LLM
- **詞彙表規則** — 定義替換規則（如 `OpenAI=>OpenAI`）或強制保留詞
- **標點清理** — 移除行首破折號、行尾逗號/句號
- **費用追蹤** — 即時顯示 ElevenLabs 和 OpenAI 的 API 使用成本
- **多模型支援** — GPT-4.1、GPT-4.1 mini、GPT-5.2、GPT-5-mini，可調整推理強度

## 運作原理

```
音訊檔案 (mp3/wav/m4a/flac/ogg)
        │
        ▼
┌─────────────────────┐
│  ElevenLabs Scribe   │  → 語音轉錄 + 字元級時間戳
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  OpenAI LLM          │  → 語意斷句，切分為字幕行
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  對齊引擎             │  → 為每行字幕匹配精準起訖時間
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  後處理               │  → 校正、翻譯、繁簡轉換、標點清理
└─────────┬───────────┘
          │
          ▼
      SRT 字幕檔
```

## 前置需求

- Python 3.9+
- [ElevenLabs](https://elevenlabs.io/) API Key
- [OpenAI](https://platform.openai.com/) API Key

## 安裝

```bash
git clone https://github.com/leonchanwy/ExactSub.git
cd ExactSub
pip install -r requirements.txt
```

## 使用方式

```bash
streamlit run app.py
```

然後開啟終端機顯示的網址（預設：`http://localhost:8501`）。

### 操作步驟

1. **輸入 API Keys** — 在左側欄輸入 ElevenLabs 和 OpenAI 的 API Key
2. **調整設定：**
   - 選擇字幕風格（YouTube / TikTok）
   - 設定每行最大字數
   - 選擇 OpenAI 模型
   - 選擇音訊語言（或自動偵測）
3. **上傳音訊檔案**（mp3、wav、m4a、flac 或 ogg）
4. **點擊「開始生成字幕」**
5. **下載**生成的 `.srt` 字幕檔

### 進階設定

| 選項 | 說明 |
|------|------|
| 自訂斷句指令 | 給 LLM 斷句的額外提示詞 |
| 標點清理 | 移除行首破折號與行尾標點 |
| 字幕校正 | LLM 修正轉錄錯字與聽錯詞 |
| 翻譯 | 透過 LLM 翻譯為其他語言 |
| OpenCC 繁簡轉換 | 輕量簡體→繁體轉換（不需 LLM） |
| 詞彙表規則 | `原詞=>目標詞` 替換或保留詞 |
| 推理強度（GPT-5） | 控制思考深度：none → xhigh |
| 重試設定 | 設定上傳逾時與重試行為 |

### 詞彙表格式

```
# 替換規則（一行一組）
王小明=>小明
OpenAI=>OpenAI

# 保留詞（不含箭頭，維持原樣）
Hokage
```

## 費用估算

應用程式會即時追蹤 API 費用：

| 服務 | 定價 |
|------|------|
| ElevenLabs Scribe | $0.008 / 每分鐘音訊 |
| OpenAI GPT-4.1 | $2.00 / 1M input tokens、$8.00 / 1M output tokens |
| OpenAI GPT-5.2 | $1.75 / 1M input tokens、$14.00 / 1M output tokens |
| OpenAI GPT-5-mini | $0.25 / 1M input tokens、$2.00 / 1M output tokens |
