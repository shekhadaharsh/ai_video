"""
qa_reviewer.py -- Comparative Gemini QA review of rendered ad videos.

After FFmpeg renders each output video, this module uploads:
  1. The ORIGINAL source video (once, reused for all reviews)
  2. The RENDERED ad variant video
Gemini compares both side-by-side to identify:
  - Audio/video sync drift (lips moving before/after voice)
  - Abrupt cuts (mid-sentence, mid-word, or missing start/end words)
  - Visual errors (duplicate frames, visual jumps, clashing transitions)
  - Text overlay contrast and placement
"""

import os
import time
import json
import logging
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from modules.utils import clean_json_response, load_json

load_dotenv()
logger = logging.getLogger(__name__)

# Model to use for QA -- same premium model for best comparison
QA_MODEL = "gemini-3.1-pro-preview"



def robust_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        import ast
        try:
            # Clean python values and fallback to ast.literal_eval
            cleaned = text.replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')
            # Normalize newlines
            cleaned = cleaned.replace('\n', ' ')
            return ast.literal_eval(cleaned)
        except Exception as e:
            # If everything fails, try basic regex cleanups or raise the json error
            raise ValueError(f'Failed to parse JSON: {e}. Raw: {text[:200]}')


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    return genai.Client(api_key=api_key)


def build_qa_prompt(hook_type: str, hook_script: str, video_duration: float, segments_info: str) -> str:
    return f"""You are a professional video editor and ad specialist performing a QUALITY CONTROL review.

You have two videos uploaded to watch:
1. ORIGINAL SOURCE VIDEO (raw unedited footage)
2. GENERATED AD VARIANT (the final stitched ad variant)

METADATA OF AD VARIANT:
- Hook Type: {hook_type}
- Intended Hook Script (text overlay): "{hook_script}"
- Duration: {video_duration:.1f}s

STITCHED SEGMENTS STRUCTURE (times refer to timestamps in the ORIGINAL SOURCE VIDEO):
{segments_info}

YOUR TASK:
Compare the Generated Ad Variant side-by-side against the Original Source Video to find any bugs. 
Be extremely critical. Your goal is to spot any errors in editing, timing, or sync.

Evaluate the following:

1. AUDIO/VIDEO SYNC AND MISMATCH (CRITICAL):
   - Listen to the voice in the Generated Ad. Does the audio line up perfectly with the speaker's mouth movements in the video?
   - Compare the voice in the Generated Ad to the original footage. Is the audio shifted, delayed, or out-of-sync?
   - Check if there is any sync drift (e.g. lips move before voice starts or vice versa).

2. ABRUPT CUTS / MISSING AUDIO / GAPS:
   - Check if any clip in the Generated Ad starts mid-word or cuts off mid-sentence.
   - Specifically, check the beginning (0:00) of the Generated Ad. Does the audio start mid-word, or did we lose the first word of the hook sentence?
   - Compare the start/end of each segment in the Generated Ad with its source range in the Original video. Did we lose any critical starting or ending words?

3. VISUAL / FRAME MISMATCHES:
   - Are there any duplicate frames, repeating frames, or extra clips inserted at the wrong place?
   - Does the visual cut happen smoothly at the transition points, or is there a visual jump/mismatch?

4. TEXT OVERLAYS:
   - Is the text block placed at the bottom-third of the screen?
   - Does it block the presenter's face or look cluttered?
   - Does it fade out correctly?

Respond ONLY with valid JSON in this exact structure:

{{
  "quality_score": 7,
  "verdict": "brief overall verdict based on comparing both videos",
  "issues": [
    {{
      "category": "av_sync",
      "severity": "critical",
      "timestamp_approx": 3.5,
      "description": "Speech is shifted by 0.3s -- lips move before voice starts compared to original",
      "fix_suggestion": "Shift audio track forward by 0.3s"
    }}
  ],
  "biggest_problem": "single most critical issue in one sentence",
  "would_stop_scrolling": true,
  "audio_smooth": false,
  "visual_smooth": true,
  "text_readable": true,
  "av_sync_perfect": true
}}

Severity levels: "critical" | "major" | "minor"
Categories: "audio_cut" | "visual_cut" | "text_overlay" | "hook_effectiveness" | "timing" | "av_sync" | "other"

If no issues found in a category, omit those issues.
"""


def review_single_video(
    original_file_obj,
    video_path: str,
    hook_type: str,
    hook_script: str,
    segments_info: str
) -> dict:
    """
    Upload a rendered video and compare it side-by-side with the original file object.
    """
    client = get_gemini_client()
    rendered_file = None

    try:
        video_duration = _get_video_duration(video_path)
        prompt = build_qa_prompt(hook_type, hook_script, video_duration, segments_info)

        logger.info(f"  Uploading rendered {Path(video_path).name} for comparative review...")
        rendered_file = client.files.upload(file=Path(video_path))

        # Wait for processing
        poll_count = 0
        while rendered_file.state.name == "PROCESSING":
            poll_count += 1
            if poll_count > 30:
                raise TimeoutError("Rendered QA upload timed out")
            time.sleep(5)
            rendered_file = client.files.get(name=rendered_file.name)

        if rendered_file.state.name == "FAILED":
            raise RuntimeError("Rendered QA upload failed on Google servers.")

        logger.info("  Running side-by-side comparison...")
        response = client.models.generate_content(
            model=QA_MODEL,
            # Pass BOTH original and rendered files to allow side-by-side comparison
            contents=[original_file_obj, rendered_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
            )
        )

        raw = clean_json_response(response.text)
        result = robust_json_loads(raw)
        result["video_path"] = video_path
        result["hook_type"] = hook_type
        return result

    except Exception as e:
        logger.warning(f"  QA review failed for {Path(video_path).name}: {e}")
        return {
            "error": str(e),
            "video_path": video_path,
            "hook_type": hook_type,
            "quality_score": None,
            "issues": []
        }
    finally:
        if rendered_file:
            try:
                client.files.delete(name=rendered_file.name)
            except Exception:
                pass


def review_rendered_videos(
    rendered_videos: list,
    source_video_path: str,
    analysis_json_path: str,
    max_reviews: int = 5
) -> list:
    """
    Upload original video once, and perform side-by-side comparative review on rendered variants.
    """
    client = get_gemini_client()
    original_file = None

    # Load analysis for segment ranges
    try:
        analysis = load_json(analysis_json_path)
    except Exception as err:
        logger.error(f"Failed to load analysis.json: {err}")
        analysis = {}

    # Only review actual ad variants (not reference/B-roll clips)
    is_ref = lambda t: any(kw in t for kw in ["Reference Only", "Insert Shot", "Cutaway", "Reaction Shot"])
    ad_videos = [v for v in rendered_videos if not is_ref(v.get("type", ""))]

    # Limit reviews
    to_review = ad_videos[:max_reviews]

    if not to_review:
        logger.info("No rendered variants to review.")
        return []

    logger.info(f"\n{'='*60}")
    logger.info(f"[QA COMPARISON] Starting side-by-side QA on {len(to_review)} variants")
    logger.info(f"Source video: {Path(source_video_path).name}")
    logger.info(f"{'='*60}\n")

    all_results = []

    try:
        # Upload original video once
        logger.info(f"Uploading original source video ({Path(source_video_path).name})...")
        original_file = client.files.upload(file=Path(source_video_path))

        # Wait for processing
        poll_count = 0
        while original_file.state.name == "PROCESSING":
            poll_count += 1
            if poll_count > 40:
                raise TimeoutError("Original video upload timed out")
            time.sleep(5)
            original_file = client.files.get(name=original_file.name)

        if original_file.state.name == "FAILED":
            raise RuntimeError("Original video upload failed on Google servers.")

        # Review each variant side-by-side
        for idx, vid in enumerate(to_review):
            video_path  = vid.get("path", "")
            hook_type   = vid.get("type", "Unknown")
            hook_script = vid.get("hook_text", "")

            if not Path(video_path).exists():
                logger.warning(f"  Skipping {hook_type}: file not found at {video_path}")
                continue

            # Build segment info description for this hook
            segments_info = _extract_segments_info(analysis, hook_type)

            logger.info(f"[QA {idx+1}/{len(to_review)}] Reviewing: {hook_type}")
            logger.info(f"  Script: \"{hook_script}\"")
            logger.info(f"  Expected structure: {segments_info.replace(chr(10), ' | ')}")

            result = review_single_video(original_file, video_path, hook_type, hook_script, segments_info)
            all_results.append(result)

            # Print structured terminal output for easy copy-paste
            _print_qa_result(result, idx + 1, len(to_review))

    except Exception as main_err:
        logger.error(f"QA comparison pipeline failed: {main_err}")
        # Return fallback results
        for vid in to_review:
            all_results.append({
                "error": str(main_err),
                "video_path": vid.get("path", ""),
                "hook_type": vid.get("type", "Unknown"),
                "quality_score": None,
                "issues": []
            })
    finally:
        if original_file:
            try:
                logger.info("Cleaning up original video upload from Google servers...")
                client.files.delete(name=original_file.name)
            except Exception:
                pass

    # Print final summary
    _print_qa_summary(all_results)

    return all_results


def _extract_segments_info(analysis: dict, hook_type: str) -> str:
    """Helper to format segments timings used to build the variant."""
    if not analysis:
        return "Segments timings not available."

    segments = analysis.get("segments", {})
    demo     = segments.get("demo", {})
    result   = segments.get("result", {})

    # Find the corresponding hook best_clip
    hook_clip = None
    for hook in analysis.get("applicable_hooks", []):
        if hook.get("type") == hook_type:
            hook_clip = hook.get("best_clip", {})
            break

    if not hook_clip:
        # Fallback if type name mismatch
        hook_clip = {"start": 0.0, "end": 4.0}

    info = (
        f"- Hook segment: {hook_clip.get('start', 0.0):.2f}s to {hook_clip.get('end', 4.0):.2f}s\n"
        f"- Demo segment: {demo.get('start', 0.0):.2f}s to {demo.get('end', 0.0):.2f}s\n"
        f"- Result segment: {result.get('start', 0.0):.2f}s to {result.get('end', 0.0):.2f}s\n"
        f"Note: Duplicate footage from the hook range has been subtracted from Demo and Result segments."
    )
    return info


def _print_qa_result(result: dict, idx: int, total: int):
    """Print a formatted QA result to terminal for easy reading."""
    sep = "-" * 55
    logger.info(f"\n{sep}")
    logger.info(f"QA RESULT [{idx}/{total}]: {result.get('hook_type', 'Unknown')}")
    logger.info(sep)

    if "error" in result:
        logger.info(f"  STATUS: ERROR -- {result['error']}")
        logger.info(sep)
        return

    score = result.get("quality_score", "?")
    verdict = result.get("verdict", "")
    biggest = result.get("biggest_problem", "")

    logger.info(f"  QUALITY SCORE : {score}/10")
    logger.info(f"  VERDICT       : {verdict}")
    logger.info(f"  BIGGEST ISSUE : {biggest}")
    logger.info(f"  AV SYNC       : {'PERFECT' if result.get('av_sync_perfect', True) else 'OUT OF SYNC  <-- CRITICAL'}")
    logger.info(f"  AUDIO SMOOTH  : {'YES' if result.get('audio_smooth') else 'NO  <-- PROBLEM'}")
    logger.info(f"  VISUAL SMOOTH : {'YES' if result.get('visual_smooth') else 'NO  <-- PROBLEM'}")
    logger.info(f"  TEXT READABLE : {'YES' if result.get('text_readable') else 'NO  <-- PROBLEM'}")
    logger.info(f"  SCROLL STOPPER: {'YES' if result.get('would_stop_scrolling') else 'NO  <-- PROBLEM'}")

    issues = result.get("issues", [])
    if issues:
        logger.info(f"\n  ISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            sev  = issue.get("severity", "?").upper()
            cat  = issue.get("category", "?")
            ts   = issue.get("timestamp_approx", "?")
            desc = issue.get("description", "")
            fix  = issue.get("fix_suggestion", "")
            logger.info(f"    [{i}] [{sev}] {cat} @ ~{ts}s")
            logger.info(f"        Problem: {desc}")
            logger.info(f"        Fix:     {fix}")
    else:
        logger.info("  ISSUES FOUND: None -- clean output!")

    logger.info(sep)


def _print_qa_summary(results: list):
    """Print an overall summary of all QA results."""
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("QA REVIEW SUMMARY (COMPARATIVE)")
    logger.info(sep)

    valid = [r for r in results if "error" not in r and r.get("quality_score") is not None]
    errors = [r for r in results if "error" in r]

    if valid:
        avg_score = sum(r.get("quality_score", 0) for r in valid) / len(valid)
        logger.info(f"  Videos reviewed : {len(results)}")
        logger.info(f"  Successful QA   : {len(valid)}")
        logger.info(f"  Failed QA       : {len(errors)}")
        logger.info(f"  Average score   : {avg_score:.1f}/10")

        all_issues = []
        for r in valid:
            all_issues.extend(r.get("issues", []))

        critical = [i for i in all_issues if i.get("severity") == "critical"]
        major    = [i for i in all_issues if i.get("severity") == "major"]
        minor    = [i for i in all_issues if i.get("severity") == "minor"]

        logger.info(f"\n  Total issues    : {len(all_issues)}")
        logger.info(f"    CRITICAL       : {len(critical)}")
        logger.info(f"    MAJOR          : {len(major)}")
        logger.info(f"    MINOR          : {len(minor)}")

        if critical:
            logger.info("\n  CRITICAL ISSUES TO FIX FIRST:")
            for i, issue in enumerate(critical, 1):
                logger.info(f"    {i}. [{issue.get('category', '?')}] {issue.get('description', '')}")
                logger.info(f"       Fix: {issue.get('fix_suggestion', '')}")
    else:
        logger.info("  No valid QA results (all failed or no videos reviewed).")

    logger.info(sep)
    logger.info("END OF COMPARATIVE QA REPORT -- Copy issues above to fix sync/editing bugs")
    logger.info(sep + "\n")


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    import subprocess
    from modules.utils import find_ffmpeg

    ffmpeg_path = find_ffmpeg()
    ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if not Path(ffprobe).exists():
        ffprobe = str(Path(ffmpeg_path).parent / "ffprobe.exe")

    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
