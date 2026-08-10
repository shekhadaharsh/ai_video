# AI-Video — UGC Ad Generator

An AI-powered pipeline that automatically repurposes influencer/UGC product videos into multiple short-form ad variants using Gemini's native video understanding and FFmpeg rendering.

## What it does

1. **Analyzes** your source video with Google Gemini — understands speech, identifies hook/problem/demo/result segments, detects insert shots, reaction shots, cut types (J-cut, L-cut, etc.)
2. **Generates** 4–6 ad variants automatically — Problem Hook, Result Hook, Emotional Hook, Testimonial, Offer, Before/After
3. **Renders** with FFmpeg — frame-accurate clips, burn-in text overlays, cut-type-aware audio crossfades, direction-aware silence snapping
4. **QA Reviews** (optional) — Gemini compares rendered variants side-by-side with original to flag audio/video sync issues

## Tech Stack

- **Frontend:** Streamlit
- **AI:** Google Gemini (`gemini-3.1-pro-preview`) — native video + audio analysis
- **Rendering:** FFmpeg — single-pass `filter_complex` with J/L cut audio offsets
- **Frame Analysis:** FFmpeg `silencedetect` + `select` scene detection for timestamp accuracy

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd AI-Video

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
cp .env.example .env
# Edit .env and add:  GEMINI_API_KEY=your_key_here

# 5. Run
streamlit run app.py
```

## Project Structure

```
AI-Video/
├── app.py                  # Streamlit UI
├── run_pipeline.py         # CLI runner (no UI)
├── requirements.txt
├── modules/
│   ├── vision_analysis.py  # Gemini video analysis + prompt
│   ├── frame_analyzer.py   # FFmpeg silence/scene detection + timestamp reconciliation
│   ├── render.py           # FFmpeg rendering pipeline
│   ├── compressor.py       # Video compression before upload
│   ├── qa_reviewer.py      # Gemini comparative QA review
│   ├── cost_tracker.py     # API cost estimation
│   └── utils.py            # Shared utilities
└── data/                   # Generated outputs (gitignored)
```

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```
GEMINI_API_KEY=your_google_gemini_api_key
```

Get your API key at: https://ai.google.dev

## Notes

- Video files in `data/` are gitignored (large files, user-specific)
- `.env` is gitignored — never commit API keys
- FFmpeg must be installed and accessible in PATH
