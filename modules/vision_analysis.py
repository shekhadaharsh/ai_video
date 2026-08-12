"""
vision_analysis.py — Gemini native video analysis using the Google File API.

Uploads a compressed 720p copy of the video to Gemini.
Gemini natively listens to the audio/speech and watches the video frames.
Outputs: analysis.json and cost_report.json.
Automatically deletes the uploaded file from Google servers when finished.
"""

import os
import time
import json
import logging
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from modules.utils import clean_json_response, save_json, load_json, get_paths
from modules.cost_tracker import calculate_cost, save_cost_report, calculate_combined_cost

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-pro-preview"


# ── Prompt Builder Pass 1 (Segments, Hooks, and Source A/V Desync) ─────────────
def build_prompt_pass1(video_duration: float) -> str:
    return f"""You are analyzing a short influencer/UGC product video (duration: {video_duration:.1f} seconds)
to prepare it for ad repurposing. This prompt must generalize across ANY single-product
personal-care UGC/influencer video (soap, facewash, shampoo, skincare, haircare, etc.) —
do not assume a specific product category; base every decision only on what is actually
shown/said in THIS video.

Watch the video frames AND listen carefully to the spoken voice/audio track simultaneously.

IMPORTANT CONTEXT ABOUT THE SOURCE VIDEO:
- This video was likely composed from multiple separately recorded clips joined together.
- The audio track may be a separate voiceover recorded independently from the video.
- Therefore audio transitions and visual transitions may NOT occur at the exact same timestamp.
- The video may mix languages (e.g. Hindi/English code-switching) — this is normal, not an error.

════════════════════════════════════════════════════════
TASK A — Segment the video by MEANING (not fixed time percentages):
════════════════════════════════════════════════════════
- hook:    the opening attention-grabbing moment
- problem: where the pain point / problem is described or shown
- demo:    where the product is being used/demonstrated
- result:  where the outcome/result and any call-to-action appears

CRITICAL RULES FOR SEGMENT TIMESTAMPS:
1. Prefer timestamps where the AUDIO/SPEECH naturally transitions.
2. Choose cut points at natural pauses in speech (end of sentence, breath, pause).
3. If audio and video transitions differ, use the AUDIO timestamp as the source of truth.
4. Provide sub-second precision (e.g., 5.3, 14.7) — never rounded integers.
5. Segments must be contiguous (end of one = start of next) and cover the full video duration.

════════════════════════════════════════════════════════
TASK B — Hook-type evaluation
════════════════════════════════════════════════════════
Evaluate each of these six categories: Problem, Result, Emotional, Testimonial, Offer, Before/After.
Only include a category if the video contains genuine supporting evidence (spoken OR visual).
Never force a category that has to be invented — the number of applicable hooks is NOT fixed
and will vary by video (anywhere from 0 to 6).

For every included category, provide:
- "evidence": one sentence citing the specific spoken line or visual cue that justifies this hook
- "best_clip": the timestamp range of the best existing moment to pair with the new hook
- "new_hook_script": a new short hook message (MAX 8 WORDS for text overlay fit), written in
  the SAME language/style/register as the original video's spoken audio — preserve code-switching
  if present, do not translate to pure English or pure Hindi if the original mixes both.

STRICT CLIP BOUNDARY RULES (every rule applies to every best_clip):
1. best_clip.start MUST fall at a natural moment of silence or a clear pause in speech.
   If speech is playing at time X, never set start=X — listen for the gap BEFORE speech begins.
2. best_clip.end MUST fall at a natural moment of silence or a clear pause in speech.
   Never cut off mid-sentence or mid-word; the clip must be audibly complete if played standalone.
3. Duration should be as long as needed to capture one complete spoken thought — no shorter,
   no longer. Typical range is 3 to 12 seconds depending on the video's speech pace and
   how long the speaker takes to complete a sentence or idea. Do NOT cut mid-thought just
   to fit a fixed number; prioritize the natural end of a sentence over any target duration.
   The only hard limit: no single clip may exceed 50% of the total video duration.
4. SOURCE-SECTION RESTRICTIONS — which part of the video each hook type may draw its best_clip from:
   - Problem, Emotional  → hook or problem segment ONLY
   - Result, Before/After → result segment ONLY (transformation/outcome must be visible/spoken)
   - Testimonial         → problem, demo, or result — wherever the personal-experience statement
                           (day-by-day account, direct praise, honest review language) actually occurs
   - Offer               → wherever price/discount/promotional info is shown or spoken
5. No best_clip may cover more than 60% of the total duration of the segment it is drawn from.
   This prevents selecting an entire section as a clip instead of a focused moment within it.
   Example: if the result segment is 8 seconds long, the best_clip may be at most 4.8 seconds.
6. DEDUPLICATION: no two best_clips across all applicable_hooks may overlap by more than 30%
   of either clip's duration. If two strong candidate moments overlap by that much, keep only
   the stronger/more specific one and either find a different non-overlapping clip for the other
   hook type or exclude that hook type entirely from the output.

════════════════════════════════════════════════════════
TASK D — Audio/Video Sync Offset Detection:
════════════════════════════════════════════════════════
For EACH segment (hook, problem, demo, result), watch the speaker's lips/mouth
movement and compare it to the audio track. Determine whether audio and video appear
synced within that segment. If not synced, estimate the offset in seconds:
- Positive value = audio LAGS video (lips move first, sound arrives later)
- Negative value = audio LEADS video (sound arrives first, lips move later)
- If synced, set the value to 0.0
Provide this as "av_offset_seconds" on every segment object.

FALLBACK BEHAVIOR (important):
If no clip satisfying ALL of the above rules can be found for a given hook category,
DO NOT force a rule-breaking clip. Simply exclude that hook category from the output.
If the entire video is too short or lacks distinct content, return an empty applicable_hooks array.

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════
Respond ONLY with valid JSON — no markdown fences, no extra commentary:

{{
  "segments": {{
    "hook":    {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}},
    "problem": {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}},
    "demo":    {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}},
    "result":  {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}}
  }},
  "applicable_hooks": [
    {{
      "type": "Problem",
      "evidence": "specific spoken line or visual cue",
      "best_clip": {{"start": 0.0, "end": 0.0}},
      "new_hook_script": "max 8 word text overlay"
    }}
  ]
}}"""


# ── Prompt Builder Pass 2 (Deep Cut Analysis) ──────────────────────────────────
def build_prompt_pass2(video_duration: float) -> str:
    return f"""You are analyzing the editing cuts and B-roll of a short product video (duration: {video_duration:.1f} seconds).
Please watch the video frames AND listen to the audio to identify editing structures.

════════════════════════════════════════════════════════
TASK C — Cut Analysis (required for accurate video editing)
════════════════════════════════════════════════════════
Analyze the editing style of this video and identify:

C1. SEGMENT BOUNDARY CUT TYPES — At each boundary (hook->problem, problem->demo, demo->result):
  - "hard_cut":      instant visual change, no transition
  - "jump_cut":      same scene/speaker, time is skipped (speaker repositions)
  - "j_cut":         audio of NEXT segment starts BEFORE the video switches (audio leads)
  - "l_cut":         audio of CURRENT segment continues AFTER video switches (audio lags)
  - "action_cut":    cut occurs while subject is mid-movement (hand raise, walking, turning)
  - "smash_cut":     sudden jarring cut for dramatic/comedic effect
  - "invisible_cut": seamless cut hidden by camera motion or object passing in front
  - "montage_cut":   cut between frames of a rapid-cut montage sequence
  For j_cut and l_cut, provide audio_offset_seconds (how many seconds audio leads/lags).
  For action_cut, provide the safe_timestamp (0.2-0.5s later, when motion completes).

C2. INSERT SHOTS — close-up shots of the product, packaging, results, or key details.
  Provide start/end timestamps and what is shown.

C3. JUMP CUT LOCATIONS — timestamps of all same-scene time-skip edits (bad cut points).

C4. ACTION CUT RISKS — timestamps where the subject is mid-motion.
  Provide the safe_timestamp (when motion completes).

C5. MATCH CUTS — cuts connected by visual similarity (same shape, motion, or composition).

C6. INVISIBLE CUTS — seamless cuts hidden by camera motion, color match, or object obstruction.
  These must NOT be modified — adding any fade would destroy the effect.

C7. CUTAWAY SHOTS — brief cuts to a related scene/object, then back to main subject.
  Provide: start, end, shows, returns_to_timestamp.

C8. CROSS-CUT / PARALLEL EDITING — does the video alternate between two simultaneous scenes?
  If yes, identify both threads and which is the PRIMARY thread (about the product).
  If not detected, set detected=false.

C9. REACTION SHOTS — moments where the subject shows a clear emotional reaction.
  Powerful for social proof in Before/After and Testimonial hooks.

C10. MONTAGE SEQUENCES — rapid-cut sequences showing time passage, transformation, or results.
  The ENTIRE montage must be treated as ONE UNIT — never cut mid-montage.

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════
Respond ONLY with valid JSON — no markdown fences, no extra commentary:

{{
  "cut_analysis": {{
    "segment_boundaries": {{
      "hook_end":    {{"cut_type": "hard_cut", "audio_offset_seconds": 0.0, "is_mid_action": false, "safe_timestamp": 0.0}},
      "problem_end": {{"cut_type": "hard_cut", "audio_offset_seconds": 0.0, "is_mid_action": false, "safe_timestamp": 0.0}},
      "demo_end":    {{"cut_type": "hard_cut", "audio_offset_seconds": 0.0, "is_mid_action": false, "safe_timestamp": 0.0}}
    }},
    "insert_shots": [
      {{"start": 0.0, "end": 0.0, "description": "what is shown", "hook_potential": "high"}}
    ],
    "jump_cut_locations": [],
    "action_cut_risks": [
      {{"timestamp": 0.0, "description": "subject is mid-motion", "safe_timestamp": 0.0}}
    ],
    "match_cuts": [
      {{"timestamp": 0.0, "connected_by": "connected by shape/motion"}}
    ],
    "invisible_cuts": [
      {{"timestamp": 0.0, "hidden_by": "camera motion / passing object"}}
    ],
    "cutaway_shots": [
      {{"start": 0.0, "end": 0.0, "shows": "what is shown", "returns_to_timestamp": 0.0}}
    ],
    "cross_cut_threads": {{
      "detected": false,
      "thread_a": [],
      "thread_b": [],
      "primary_thread": "a"
    }},
    "reaction_shots": [
      {{"start": 0.0, "end": 0.0, "emotion": "emotion", "context": "context"}}
    ],
    "montage_sequences": [
      {{"start": 0.0, "end": 0.0, "description": "description"}}
    ]
  }}
}}"""


# ── JSON Validator ─────────────────────────────────────────────────────────────
def validate_analysis_schema(data: dict) -> tuple[bool, str]:
    """
    Validate that Gemini response matches expected schema.
    Returns (is_valid, error_message).
    """
    if "segments" not in data:
        return False, "Missing 'segments' key"
    if "applicable_hooks" not in data:
        return False, "Missing 'applicable_hooks' key"

    required_segments = {"hook", "problem", "demo", "result"}
    missing = required_segments - set(data["segments"].keys())
    if missing:
        return False, f"Missing segment keys: {missing}"

    for seg_name, seg_val in data["segments"].items():
        if "start" not in seg_val or "end" not in seg_val:
            return False, f"Segment '{seg_name}' missing start/end"
        # Ensure av_offset_seconds exists (default to 0.0 if prompt failed to include it)
        seg_val.setdefault("av_offset_seconds", 0.0)

    if not isinstance(data["applicable_hooks"], list):
        return False, "'applicable_hooks' must be a list"

    valid_hook_types = {"Problem", "Result", "Emotional", "Testimonial", "Offer", "Before/After"}
    for i, hook in enumerate(data["applicable_hooks"]):
        # Hard-required fields — fail if missing
        if "type" not in hook:
            return False, f"Hook {i} missing 'type' key"
        if "best_clip" not in hook:
            return False, f"Hook {i} missing 'best_clip' key"
        if "start" not in hook["best_clip"] or "end" not in hook["best_clip"]:
            return False, f"Hook {i} best_clip missing start/end"
        if hook["type"] not in valid_hook_types:
            return False, f"Hook {i} has invalid type: {hook['type']}"

        # Soft-optional fields — default if missing rather than failing
        # Gemini sometimes omits these in fallback/compressed responses
        hook.setdefault("new_hook_script", "")
        hook.setdefault("evidence", "")

    return True, ""


def _retry_api_call(fn, max_retries=3, delay=3):
    """Retry API call on temporary network or HTTP errors (e.g. getaddrinfo failed, 503, 429)."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries and ("ConnectError" in err_str or "getaddrinfo" in err_str or "503" in err_str or "429" in err_str or "httpx" in err_str):
                logger.warning(f"  Network/API glitch (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


# ── Main Analysis (Two-Pass Orchestrator) ──────────────────────────────────────
def run_vision_analysis(
    video_id: str,
    video_duration: float,
    compressed_video_path: str
) -> dict:
    """
    Full Step 4 orchestration using native Gemini File API in TWO sequential calls.
      Pass 1: Segments, Hooks, and Source A/V Offset (Task A, B, D)
      Pass 2: Detailed Cut Analysis (Task C)
    Resilient design: If Pass 2 fails, we fallback to default cut analysis
    rather than failing the entire run.
    """
    paths = get_paths(video_id)
    client = genai.Client()
    uploaded_file = None

    try:
        # 1. Upload compressed video to Gemini File API (with retry)
        logger.info(f"Uploading compressed video to Gemini File API...")
        uploaded_file = _retry_api_call(lambda: client.files.upload(file=Path(compressed_video_path)))
        logger.info(f"  File name on server: {uploaded_file.name}")

        # 2. Wait for Google server processing to finish (status ACTIVE)
        poll_count = 0
        while uploaded_file.state.name == "PROCESSING":
            poll_count += 1
            logger.info(f"  Processing on Google servers... (poll #{poll_count})")
            time.sleep(5)
            uploaded_file = _retry_api_call(lambda: client.files.get(name=uploaded_file.name))

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Gemini File API processing failed on Google servers.")

        logger.info("✓ Video processed and ACTIVE on Google servers!")

        # ──────────────────────────────────────────────────────────────────────
        # CALL 1: Task A + B + D (Segments, Hooks, Desync)
        # ──────────────────────────────────────────────────────────────────────
        prompt1 = build_prompt_pass1(video_duration)
        logger.info(f"Calling Gemini Pass 1 (Task A+B+D)...")

        resp1 = _retry_api_call(lambda: client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, prompt1],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            )
        ))
        logger.info("Pass 1 response received. Parsing JSON...")

        raw_text1 = resp1.text
        cleaned_text1 = clean_json_response(raw_text1)
        analysis = None
        last_error = None

        try:
            analysis = json.loads(cleaned_text1)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"Pass 1 response parse failed: {e}. Retrying with fallback...")

        # Fallback for Pass 1: simplified JSON structure
        if analysis is None:
            simplified_prompt = f"""You are analyzing a short product video ({video_duration:.1f}s).
Provide ONLY Tasks A and B from this analysis. Output valid JSON with:
{{"segments": {{"hook": {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}}, "problem": {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}}, "demo": {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}}, "result": {{"start": 0.0, "end": 0.0, "av_offset_seconds": 0.0}}}}, "applicable_hooks": [{{"type": "Problem", "evidence": "", "best_clip": {{"start": 0.0, "end": 0.0}}, "new_hook_script": ""}}]}}"""

            logger.info("Retrying Gemini Pass 1 with simplified fallback...")
            retry_resp = _retry_api_call(lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[uploaded_file, simplified_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            ))
            retry_text = clean_json_response(retry_resp.text)
            try:
                analysis = json.loads(retry_text)
                logger.info("Pass 1 fallback succeeded.")
            except json.JSONDecodeError as e2:
                raise ValueError(
                    f"Both Gemini Pass 1 attempts returned invalid JSON.\n"
                    f"Attempt 1 error: {last_error}\n"
                    f"Attempt 2 error: {e2}\n"
                    f"Raw response 1:\n{raw_text1[:500]}"
                )

        # Validate Pass 1 schema
        is_valid, error_msg = validate_analysis_schema(analysis)
        if not is_valid:
            raise ValueError(f"Gemini response schema validation failed: {error_msg}")

        # ──────────────────────────────────────────────────────────────────────
        # CALL 2: Task C (Deep Cut Analysis)
        # ──────────────────────────────────────────────────────────────────────
        prompt2 = build_prompt_pass2(video_duration)
        logger.info(f"Calling Gemini Pass 2 (Task C — Cut Analysis)...")

        # Keep metadata track in case call 2 is successful
        resp2_metadata = None
        try:
            resp2 = _retry_api_call(lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[uploaded_file, prompt2],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            ))
            resp2_metadata = resp2.usage_metadata
            logger.info("Pass 2 response received. Merging JSON...")

            raw_text2 = resp2.text
            cleaned_text2 = clean_json_response(raw_text2)
            cut_data = json.loads(cleaned_text2)

            # Merge cut analysis into final dict
            analysis["cut_analysis"] = cut_data.get("cut_analysis", {})
        except Exception as e_cut:
            logger.warning(f"⚠️ Pass 2 (Cut Analysis) failed or returned invalid JSON: {e_cut}. Continuing with empty cut_analysis.")
            analysis["cut_analysis"] = {
                "segment_boundaries": {
                    "hook_end": {"cut_type": "default", "audio_offset_seconds": 0.0, "is_mid_action": False, "safe_timestamp": 0.0},
                    "problem_end": {"cut_type": "default", "audio_offset_seconds": 0.0, "is_mid_action": false, "safe_timestamp": 0.0},
                    "demo_end": {"cut_type": "default", "audio_offset_seconds": 0.0, "is_mid_action": False, "safe_timestamp": 0.0}
                }
            }

        # 3. Save combined analysis.json
        save_json(analysis, paths["analysis"])
        logger.info(f"✓ Analysis saved: {paths['analysis']}")

        # 4. Save aggregated cost report
        metadata_list = [resp1.usage_metadata]
        if resp2_metadata:
            metadata_list.append(resp2_metadata)

        cost_report = calculate_combined_cost(metadata_list)
        save_cost_report(cost_report, paths["cost_report"])

        return analysis

    finally:
        # ALWAYS delete the video file from Google servers
        if uploaded_file:
            try:
                logger.info(f"Cleaning up: deleting video file {uploaded_file.name} from Google servers...")
                client.files.delete(name=uploaded_file.name)
                logger.info("✓ Google server file deleted.")
            except Exception as e:
                logger.warning(f"Failed to delete file from Google servers: {e}")
