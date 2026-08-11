"""
frame_analyzer.py -- FFmpeg-based scene detection and silence detection.

DIRECTION-AWARE TIMESTAMP RECONCILIATION:
  - For clip STARTS: snap to silence.END (speech just resumed -- clean entry)
  - For clip ENDS:   snap to silence.START (speech just stopped -- clean exit)
  - Snapping to silence MIDPOINT is FORBIDDEN -- it puts cuts inside silence
    causing the "no audio for N seconds then abrupt start" bug.
"""

import re
import copy
import subprocess
import logging
from pathlib import Path

from modules.utils import find_ffmpeg

logger = logging.getLogger(__name__)

# Safe snap windows -- tight enough to only catch real Gemini imprecision
MAX_SILENCE_SNAP = 0.7   # Max distance to snap to silence boundary
MAX_SCENE_SNAP   = 0.9   # Max distance to snap to scene change


def detect_scene_changes(video_path, threshold=0.30):
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
        "-an", "-vsync", "vfr", "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", timeout=120)
        output = result.stdout + result.stderr
        timestamps = []
        for line in output.splitlines():
            m = re.search(r"pts_time[=:](\d+\.?\d*)", line)
            if m:
                timestamps.append(float(m.group(1)))
        timestamps = sorted(set(round(t, 3) for t in timestamps))
        logger.info(f"  Scene detection: {len(timestamps)} visual cuts found")
        return timestamps
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        return []


def measure_mean_volume(video_path) -> float:
    """Measure the average loudness of a video file using FFmpeg volumedetect."""
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-i", video_path,
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", timeout=30)
        output = result.stdout + result.stderr
        m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", output)
        if m:
            val = float(m.group(1))
            logger.info(f"  Measured mean volume: {val:.1f} dB")
            return val
    except Exception as e:
        logger.warning(f"Loudness detection failed: {e}")
    return -20.0


def detect_silence_periods(video_path, noise_db=-28.0, min_duration=0.08):
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-i", video_path,
        "-af", f"silencedetect=noise={int(noise_db)}dB:d={min_duration:.2f}",
        "-vn", "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", timeout=120)
        periods = []
        current_start = None
        for line in result.stderr.splitlines():
            sm = re.search(r"silence_start: (\d+\.?\d*)", line)
            em = re.search(r"silence_end: (\d+\.?\d*)", line)
            if sm:
                current_start = float(sm.group(1))
            if em and current_start is not None:
                end = float(em.group(1))
                periods.append({
                    "start": round(current_start, 3),
                    "end": round(end, 3),
                    "midpoint": round((current_start + end) / 2, 3),
                    "duration": round(end - current_start, 3)
                })
                current_start = None
        periods.sort(key=lambda p: p["start"])
        logger.info(f"  Silence detection: {len(periods)} natural pauses found")
        return periods
    except Exception as e:
        logger.warning(f"Silence detection failed: {e}")
        return []


def reconcile_timestamp(timestamp, scene_changes, silence_periods,
                        max_silence_snap=MAX_SILENCE_SNAP,
                        max_scene_snap=MAX_SCENE_SNAP,
                        direction="start"):
    """
    Snap a Gemini timestamp to the nearest real video event.

    direction="start" -> for clip STARTS: prefer silence.END (speech resumes here).
    direction="end"   -> for clip ENDS:   prefer silence.START (speech ends here).
    direction="any"   -> for non-clip timestamps: use nearest silence boundary.
    """
    if direction == "start":
        silence_candidates = [(p["end"], p) for p in silence_periods]
    elif direction == "end":
        silence_candidates = [(p["start"], p) for p in silence_periods]
    else:
        silence_candidates = []
        for p in silence_periods:
            silence_candidates.append((p["start"], p))
            silence_candidates.append((p["end"], p))

    best_silence = None
    best_silence_dist = float("inf")
    for candidate, _ in silence_candidates:
        d = abs(timestamp - candidate)
        if d < best_silence_dist:
            best_silence_dist = d
            best_silence = candidate

    if best_silence is not None and best_silence_dist <= max_silence_snap:
        snapped = round(best_silence, 3)
        dir_label = "silence.end" if direction == "start" else "silence.start"
        logger.info(
            f"  {timestamp:.3f}s -> {snapped:.3f}s ({dir_label} snap, d={best_silence_dist:.3f}s)"
        )
        return snapped, f"{dir_label}_snap (d={best_silence_dist:.3f}s)"

    best_scene = None
    best_scene_dist = float("inf")
    for sc in scene_changes:
        d = abs(timestamp - sc)
        if d < best_scene_dist:
            best_scene_dist = d
            best_scene = sc

    if best_scene is not None and best_scene_dist <= max_scene_snap:
        snapped = round(best_scene, 3)
        logger.info(f"  {timestamp:.3f}s -> {snapped:.3f}s (scene snap, d={best_scene_dist:.3f}s)")
        return snapped, f"scene_snap (d={best_scene_dist:.3f}s)"

    return round(timestamp, 3), "original"


def analyze_video_structure(video_path):
    """Run scene + silence detection with dynamic loudness-aware threshold."""
    logger.info(f"[frame_analyzer] Analyzing: {Path(video_path).name}")
    scene_changes = detect_scene_changes(video_path, threshold=0.30)
    
    # Measure average loudness and compute dynamic silence threshold
    mean_vol = measure_mean_volume(video_path)
    noise_db = mean_vol - 12.0
    # Guardrail threshold to a safe range
    noise_db = max(-45.0, min(-18.0, noise_db))
    logger.info(f"  Dynamic silence threshold calculated: {noise_db:.1f} dB (mean volume: {mean_vol:.1f} dB)")
    
    silence_periods = detect_silence_periods(video_path, noise_db=noise_db, min_duration=0.08)
    logger.info(f"  Done: {len(scene_changes)} scene changes, {len(silence_periods)} silences")
    
    return {
        "scene_changes": scene_changes,
        "silence_periods": silence_periods,
        "video_path": video_path
    }


def validate_post_reconciliation(analysis: dict, original_analysis: dict) -> dict:
    """
    Validate post-reconciliation contiguity and section boundaries.
    """
    segs = analysis.get("segments", {})
    seg_names = ["hook", "problem", "demo", "result"]
    
    # 1. Contiguity Check
    for i in range(len(seg_names) - 1):
        curr_name = seg_names[i]
        next_name = seg_names[i+1]
        curr_seg = segs.get(curr_name)
        next_seg = segs.get(next_name)
        if curr_seg and next_seg:
            if abs(curr_seg["end"] - next_seg["start"]) > 0.001:
                logger.warning(
                    f"  [Reconcile Validation] Contiguity broken: '{curr_name}' end ({curr_seg['end']:.3f}s) "
                    f"!= '{next_name}' start ({next_seg['start']:.3f}s). Re-aligning..."
                )
                curr_seg["end"] = next_seg["start"]

    # 2. Section Restrictions Check for best_clips
    original_hooks = original_analysis.get("applicable_hooks", [])
    for idx, hook in enumerate(analysis.get("applicable_hooks", [])):
        h_type = hook.get("type", "")
        clip = hook.get("best_clip", {})
        
        allowed_start = 0.0
        allowed_end = 999.0
        
        if h_type in ("Problem", "Emotional"):
            allowed_start = segs.get("hook", {}).get("start", 0.0)
            allowed_end = segs.get("problem", {}).get("end", 999.0)
        elif h_type in ("Result", "Before/After"):
            allowed_start = segs.get("result", {}).get("start", 0.0)
            allowed_end = segs.get("result", {}).get("end", 999.0)
            
        if clip.get("start", 0.0) < allowed_start or clip.get("end", 0.0) > allowed_end:
            logger.warning(
                f"  [Reconcile Validation] Snapped hook '{h_type}' clip [{clip.get('start', 0.0):.2f}s - {clip.get('end', 0.0):.2f}s] "
                f"crossed disallowed boundaries [{allowed_start:.2f}s - {allowed_end:.2f}s]."
            )
            # Revert to original valid timestamp
            orig_clip = original_hooks[idx].get("best_clip", {}) if idx < len(original_hooks) else {}
            if orig_clip.get("start", 0.0) >= allowed_start and orig_clip.get("end", 0.0) <= allowed_end:
                logger.info(f"    Reverting to pre-snap: {orig_clip.get('start', 0.0):.2f}s - {orig_clip.get('end', 0.0):.2f}s")
                clip["start"] = orig_clip.get("start", 0.0)
                clip["end"] = orig_clip.get("end", 0.0)
            else:
                logger.warning(f"    Original clip also out of bounds. Excluding hook '{h_type}'.")
                hook["_exclude"] = True

    analysis["applicable_hooks"] = [h for h in analysis["applicable_hooks"] if not h.pop("_exclude", False)]
    return analysis


def reconcile_analysis_timestamps(analysis, video_structure):
    """
    Walk through Gemini analysis dict and snap ALL timestamps to real video events.
    """
    original_analysis = copy.deepcopy(analysis)
    analysis = copy.deepcopy(analysis)
    sc = video_structure.get("scene_changes", [])
    sp = video_structure.get("silence_periods", [])
    snap_log = []

    def snap(ts, label, direction="start"):
        snapped, reason = reconcile_timestamp(float(ts), sc, sp, direction=direction)
        if round(snapped, 3) != round(float(ts), 3):
            snap_log.append(f"{label}: {float(ts):.3f}s -> {snapped:.3f}s ({reason})")
        return snapped

    # Segment boundaries
    for name, seg in analysis.get("segments", {}).items():
        seg["start"] = snap(seg["start"], f"segments.{name}.start", direction="start")
        seg["end"]   = snap(seg["end"],   f"segments.{name}.end",   direction="end")

    # Hook best_clips
    for i, hook in enumerate(analysis.get("applicable_hooks", [])):
        clip = hook.get("best_clip", {})
        clip["start"] = snap(clip["start"], f"hook[{i}].best_clip.start", direction="start")
        clip["end"]   = snap(clip["end"],   f"hook[{i}].best_clip.end",   direction="end")

    cut = analysis.get("cut_analysis", {})

    # Shot groups
    for group in ("insert_shots", "cutaway_shots", "reaction_shots", "montage_sequences"):
        for j, item in enumerate(cut.get(group, [])):
            item["start"] = snap(item["start"], f"{group}[{j}].start", direction="start")
            item["end"]   = snap(item["end"],   f"{group}[{j}].end",   direction="end")

    # Action cut risks
    for j, risk in enumerate(cut.get("action_cut_risks", [])):
        risk["timestamp"]      = snap(risk["timestamp"],      f"action_risk[{j}].ts",      direction="any")
        risk["safe_timestamp"] = snap(risk["safe_timestamp"], f"action_risk[{j}].safe_ts", direction="end")

    # Point timestamps
    for j, mc in enumerate(cut.get("match_cuts", [])):
        mc["timestamp"] = snap(mc["timestamp"], f"match_cut[{j}]", direction="any")
    for j, ic in enumerate(cut.get("invisible_cuts", [])):
        ic["timestamp"] = snap(ic["timestamp"], f"invisible_cut[{j}]", direction="any")

    # Jump cut locations
    jcl = cut.get("jump_cut_locations", [])
    if jcl and isinstance(jcl[0], (int, float)):
        cut["jump_cut_locations"] = [
            snap(float(t), f"jump_cut[{j}]", direction="any") for j, t in enumerate(jcl)
        ]

    # Run Post-Reconciliation Validation Check
    analysis = validate_post_reconciliation(analysis, original_analysis)

    # Summary
    if snap_log:
        logger.info(f"  Reconciled {len(snap_log)} timestamps:")
        for entry in snap_log:
            logger.info(f"    -- {entry}")
    else:
        logger.info("  All Gemini timestamps were accurate -- no snapping needed")

    analysis["_frame_analysis"] = {
        "scene_changes_count": len(sc),
        "silence_periods_count": len(sp),
        "timestamps_reconciled": len(snap_log),
        "snap_log": snap_log
    }
    return analysis
