# ExactSub

AI-powered subtitle generator that combines **ElevenLabs Scribe** for precise character-level timestamps with **OpenAI LLMs** for intelligent sentence segmentation — producing accurately timed SRT subtitles from audio files.

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
Audio File (mp3/wav/m4a)
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
3. **Upload an audio file** (mp3, wav, or m4a)
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
