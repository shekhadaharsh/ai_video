# Video-to-Ad-Hooks Generator — Full Implementation Spec

## 1. Project Goal

Build a system that takes **one uploaded influencer/UGC video** and automatically produces **multiple ready-to-post ad video variants**, each built around a different "hook" (the opening 4-8 seconds that grabs attention). The number of variants is **not fixed** — an AI model analyzes the source video and decides how many distinct hooks the content genuinely supports.

**Core principle: No new AI-generated voice, voice clone, or AI avatar is ever created.** Every output video uses 100% of the original creator's own footage and voice. Only the opening hook portion changes — by reusing a different existing moment from the same video and adding a new on-screen text/caption overlay.

---

## 2. High-Level Flow

```
1. Video Upload
       ↓
2. Transcription (Whisper — self-hosted, free) → timestamped transcript
       ↓
3. Frame Sampling (FFmpeg) → ~20-30 sampled frames across the video
       ↓
4. AI Analysis (Gemini 3.1 Pro — transcript + frames together)
   → Segments the video into Hook / Problem / Demo / Result (by meaning, not fixed time)
   → Evaluates all 6 candidate hook types and decides which ones the video
     actually supports with real evidence (dynamic count, not fixed at 6)
   → Outputs one structured JSON file
       ↓
5. Rendering Loop (Shotstack API) — one render job PER applicable hook
   → For each hook: cut the best-matching existing clip + overlay the new
     hook text + stitch with the reused Demo and Result footage
       ↓
6. Download & Store final videos (local folder for POC; cloud storage later)
       ↓
7. (Phase 2, later) Simple web UI for upload + result gallery
```

---

## 3. Why Hook Count Is Dynamic, Not Fixed

The six candidate hook categories are a **checklist**, not a requirement:

- Problem hook
- Result hook
- Emotional hook
- Testimonial hook
- Offer hook
- Before/After hook

For every video, the AI must independently check, for each category, whether the source video contains real evidence (spoken OR visual) supporting it. A category is only included if genuine evidence exists — never invent or force a hook type that isn't grounded in the actual footage.

Examples:
- No price/discount ever shown or mentioned → exclude "Offer".
- No visual before/after comparison and nothing spoken about a before/after change → exclude "Before/After".
- Influencer says something like a testimonial line ("this changed my skin in a week") → include "Testimonial".

Result: one video might produce 3 hooks, another might produce all 6. **Never hardcode the count.**

---

## 4. Tech Stack & Costs (confirmed choices)

| Component | Tool | Cost | Notes |
|---|---|---|---|
| Transcription | **Whisper — Open Source, Self-Hosted** (`openai-whisper` Python package, model size: `base` or `small` for POC) | **Free** (no API cost) — only cost is your own compute (CPU/GPU time) | Runs entirely on your own machine/server. No API key needed. Slower than the hosted API if you don't have a GPU, but zero per-minute cost. Supports 99+ languages, handles Hindi/English code-switching. |
| Frame extraction | **FFmpeg** (local, open-source) | Free | Sample 1 frame every ~2 seconds, cap total frames sent to Gemini at ~30 per video regardless of length. |
| Vision + reasoning | **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`) — PAID | $2.00 / million input tokens, $12.00 / million output tokens. No free tier. Long-context (>200K tokens) doubles to $4/$18. Batch mode (non-urgent, 50% off): $1.00/$6.00. | Native multimodal — accepts transcript text + multiple images (frames) in a single request. Estimated cost: **~$0.05–$0.15 per video** for the analysis step. |
| Rendering / assembly | **Shotstack** (cloud video-editing API) | ~$0.20–$0.30 per minute of rendered video | Fully automated cutting, text overlay, and stitching via JSON instructions — no manual editing, no FFmpeg-stitching code to maintain. |
| Orchestration | **Python 3.11+** | Free | Single script/small app is enough for POC. |
| Storage (POC) | Local filesystem (`/data`, `/outputs`) | Free | Upgrade to S3/Cloudinary only when moving past POC. |

**Model choice confirmed for this POC:**
- **Transcription = Whisper Open Source (self-hosted, free)** — chosen specifically to keep this stage at zero API cost while testing.
- **Vision/Reasoning = Gemini 3.1 Pro (paid)** — chosen for its strong multi-image (frame) handling and large context window; no free-tier equivalent is accurate enough for this stage, so this is the one paid component in the pipeline.

**Estimated total cost per source video (all applicable hooks rendered): roughly $0.25 – $1.40** (lower than before since transcription is now free; only Gemini + Shotstack cost money).

**Important:** Feed Gemini the **transcript text** produced by Whisper, not raw audio. Audio input tokens cost 2–7x more than text tokens on Gemini, and Whisper already gives more precise word-level timestamps than Gemini would from raw audio. Keep transcription (free, local) and vision reasoning (paid, cloud) as two separate, specialized steps.

---

## 5. Environment Setup

```bash
python -m venv venv
source venv/bin/activate

# Whisper (self-hosted, free) — installs the open-source model + its dependencies
pip install openai-whisper torch

# Gemini (paid, vision/reasoning)
pip install google-generativeai

# Rendering + utilities
pip install ffmpeg-python python-dotenv requests
```

**System-level requirement:** FFmpeg binary must also be installed on the machine itself (not just the Python wrapper) — Whisper's Python package uses it internally too:
```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

**Whisper model size — pick one for the POC:**

| Model size | Approx. speed on CPU | Accuracy | Recommended for |
|---|---|---|---|
| `tiny` | Fastest | Lowest | Quick sanity tests only |
| `base` | Fast | Good | **Recommended starting point for this POC** |
| `small` | Medium | Better | If `base` accuracy isn't good enough on your Hindi/English videos |
| `medium` / `large` | Slow on CPU, needs GPU | Best | Only if you have a GPU and accuracy really matters |

The model file downloads automatically the first time you run it (no API key, no account needed — it just needs internet access once to fetch the model weights, then it can run fully offline).

`.env` file (only Gemini and Shotstack need keys — Whisper does not):
```
GEMINI_API_KEY=your_gemini_key_here
SHOTSTACK_API_KEY=your_shotstack_key_here
```

**Getting the keys:**
- Gemini: aistudio.google.com → "Get API key" → Create API key (works with a normal Google account). Gemini 3.1 Pro itself is paid-only (no free tier), so billing must be enabled on the linked Google Cloud project.
- Shotstack: shotstack.io → sign up → dashboard has your API key (comes with some free trial credits).
- Whisper: **no key needed** — it's a local Python package, not an API.

---

## 6. Pipeline Steps in Detail

### Step 1 — Accept Video Input
- `python run_pipeline.py --video path/to/video.mp4`
- Validate file exists and is a supported format (mp4, mov, webm).
- Generate a unique `video_id` for this run (e.g. timestamp-based) and create `/data/<video_id>/`.

### Step 2 — Transcription (Whisper, self-hosted/free)
- Extract audio from the video first (Whisper's Python package can also read video files directly, but extracting audio first is more reliable and faster):
```bash
ffmpeg -i input.mp4 -vn -acodec libmp3lame -q:a 2 /data/<video_id>/audio.mp3
```
- Run local transcription with word/segment-level timestamps:
```python
import whisper

model = whisper.load_model("base")  # loads once, reuse across videos in the same run
result = model.transcribe("/data/<video_id>/audio.mp3", verbose=False)

# result["segments"] contains a list of {start, end, text} — this is what we need
```
- Save `result["segments"]` (with timestamps) to `/data/<video_id>/transcript.json`.
- **Note on first run:** the very first time `load_model("base")` runs, it downloads the model weights (~140MB for `base`) — this requires internet once. After that, it's cached locally and runs fully offline.
- **Note on speed:** on a normal laptop CPU (no GPU), expect roughly 1x–3x real-time (a 60-second video might take 1–3 minutes to transcribe with the `base` model). This is fine for POC testing; if it's too slow, drop to `tiny`, or use a machine with a GPU.

### Step 3 — Frame Sampling (FFmpeg)
```bash
ffmpeg -i input.mp4 -vf fps=1/2 /data/<video_id>/frames/frame_%04d.jpg
```
- Keep a mapping of `frame_filename → timestamp_seconds`.
- Cap total frames sent to Gemini at ~30 (sample evenly across full video length) to control cost.

### Step 4 — AI Analysis (Gemini 3.1 Pro)
Send ONE request containing:
- Full timestamped transcript (as text)
- All sampled frames (as images, each labeled with its timestamp)

**Prompt structure (use this JSON schema exactly):**

```
You are analyzing a short influencer/UGC product video to prepare it for ad repurposing.

You are given:
1. A timestamped transcript of everything spoken in the video.
2. A set of frames sampled from the video at regular timestamps.

TASK A — Segment the video by MEANING (not fixed time percentages):
- hook: the opening attention-grabbing moment
- problem: where the pain point / problem is described or shown
- demo: where the product is being used/demonstrated
- result: where the outcome/result and any call-to-action appears
Return start/end timestamps (seconds) for each, based on both what is said and shown.

TASK B — Hook-type evaluation. For EACH of these six categories:
Problem, Result, Emotional, Testimonial, Offer, Before/After

Decide whether the video contains genuine supporting evidence (spoken OR visual).
Only include a category if real evidence exists — never invent one.
For every included category, provide:
- a one-sentence justification citing the evidence found
- the timestamp range of the best EXISTING video moment to pair with a new hook
  message for that category (this clip will be reused as-is, not regenerated)
- a new short hook script (4-8 seconds if spoken aloud) in the style/language
  of the original video, to be overlaid as on-screen text/caption on that moment

Do not force a fixed number of categories. Videos may support 2 to 6 of them.

Respond ONLY with valid JSON in this exact structure:

{
  "segments": {
    "hook": {"start": 0.0, "end": 0.0},
    "problem": {"start": 0.0, "end": 0.0},
    "demo": {"start": 0.0, "end": 0.0},
    "result": {"start": 0.0, "end": 0.0}
  },
  "applicable_hooks": [
    {
      "type": "Problem",
      "evidence": "string explaining what was found",
      "best_clip": {"start": 0.0, "end": 0.0},
      "new_hook_script": "string"
    }
  ]
}
```

- Strip markdown code fences (` ```json `) from the response before parsing.
- Save to `/data/<video_id>/analysis.json`.

### Step 5 — Rendering Loop (Shotstack) — ONE JOB PER HOOK

Video rendering does **not** happen as a single combined request. For every entry in `applicable_hooks`, submit a **separate Shotstack render job**:

```python
render_jobs = []

for hook in analysis["applicable_hooks"]:
    render_request = build_shotstack_json(
        hook_clip=hook["best_clip"],          # cut from the SAME source video, reused
        hook_text=hook["new_hook_script"],     # new on-screen text overlay only
        demo_clip=analysis["segments"]["demo"],
        result_clip=analysis["segments"]["result"],
    )
    response = shotstack_api.submit_render(render_request)
    render_jobs.append({"hook_type": hook["type"], "render_id": response["render_id"]})

# Poll/wait for all jobs, then download each finished video
for job in render_jobs:
    result_url = wait_for_completion(job["render_id"])
    download_video(result_url, f"/data/<video_id>/outputs/{job['hook_type']}.mp4")
```

Key points:
- Each hook = its own independent render job with its own unique ID.
- Submit all jobs asynchronously first, then poll for completion — this lets Shotstack process them in parallel on their cloud, rather than waiting for one to finish before starting the next.
- The "Hook Assembly" step (cutting the existing clip + overlaying new text) happens **inside the Shotstack JSON instructions** — no new voice, no new footage, no avatar is generated at any point. Text overlay only.

### Step 6 — Output Storage
- Shotstack returns a temporary cloud-hosted URL per finished render. Download each to local storage:
```
/data/<video_id>/outputs/
  ├── Problem.mp4
  ├── Result.mp4
  ├── Emotional.mp4
  └── Testimonial.mp4          (however many hooks were applicable — not fixed)
```
- For POC purposes, this local folder is the final deliverable — no UI is needed yet. Just open/play the files directly to review.

---

## 7. What Is Explicitly Out of Scope for the POC

- Any new AI voice, voice cloning, or AI avatar generation — never used at any tier of this system.
- Web/app UI — POC output is just files in a local folder.
- Multi-user auth, permanent cloud storage, production deployment.
- Manual video editing of any kind — every step above must be automated end-to-end.

---

## 8. File/Folder Structure to Build

```
project/
├── .env
├── requirements.txt
├── run_pipeline.py            # main entry point, orchestrates steps 1-6
├── modules/
│   ├── transcribe.py          # Self-hosted Whisper (open-source, free) wrapper — no API key needed
│   ├── frame_sampler.py       # FFmpeg frame extraction wrapper
│   ├── vision_analysis.py     # Gemini API call + prompt + JSON parsing
│   ├── render.py              # Shotstack JSON builder + submit + poll + download
│   └── utils.py                # shared helpers (paths, JSON cleanup, etc.)
└── data/
    └── <video_id>/
        ├── transcript.json
        ├── frames/
        ├── analysis.json
        └── outputs/
            └── <hook_type>.mp4  (N files, N = however many hooks were applicable)
```

---

## 9. Acceptance Criteria for This POC

- [ ] Given any single video file (any length, any Whisper-supported language), the pipeline runs end-to-end with a single command — no manual steps in between.
- [ ] `analysis.json` is valid JSON matching the schema in Step 4.
- [ ] The number of entries in `applicable_hooks` varies based on actual video content — running the pipeline on two different videos should plausibly produce different counts (not always 6).
- [ ] For each applicable hook, a separate final `.mp4` is rendered and downloaded to the local `outputs/` folder.
- [ ] No new AI voice or avatar appears anywhere in any output video — 100% original footage and voice only.
- [ ] Transcription runs fully locally via self-hosted Whisper (`base` model) with zero API cost; Gemini 3.1 Pro is the only paid model used, for the vision/reasoning step.
- [ ] Total cost per test video stays under ~$1.40 (Gemini + Shotstack only; transcription is free).

---

## 10. Future Phases (after POC is validated)

**Phase 2 — Simple Web UI**
- Basic upload page → progress indicator while pipeline runs → gallery view of the resulting N videos with preview + download buttons.
- Move storage from local folder to cloud storage (S3/Cloudinary) so links are shareable.

**Phase 3 — Production Hardening**
- Queue-based processing for multiple videos at once.
- User accounts, history of past runs.
- Cost monitoring/alerts per video processed.
- Optional: allow manual review/edit of the AI's hook selection before rendering.
