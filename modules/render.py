"""
render.py — Production-Grade FFmpeg rendering for AI Video Hook Generator.

Uses FFmpeg filter_complex to process ALL segments in a single pass:
  - Zero encoder delay gaps between clips (all trimmed in memory)
  - Audio crossfade (0.3s) at every clip boundary → completely seamless transitions
  - aresample=async=1 as final safety net for any residual audio/video drift
  - drawtext applied inline to hook segment only
  - Scale all output to 1080x1920 (vertical HD) in the same pass
  - Dynamic interval subtraction to prevent any repeated footage across segments

Single FFmpeg invocation per variant = no temp files, no pops, no drift.
"""

import os
import json
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

from modules.utils import find_ffmpeg, save_json

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
# Font for text overlay — Arial Bold is available on all Windows systems
FONT_PATHS = [
    r"C:\Windows\Fonts\arialbd.ttf",    # Arial Bold (best for readability)
    r"C:\Windows\Fonts\arial.ttf",      # Arial Regular
    r"C:\Windows\Fonts\calibri.ttf",    # Calibri fallback
]

# Text overlay style
TEXT_FONT_SIZE   = 48                    # Slightly smaller to prevent overflow
TEXT_COLOR       = "white"
TEXT_BOX_COLOR   = "black@0.75"          # Darker background for high contrast
TEXT_BOX_PADDING = 12
TEXT_POSITION_Y  = "h*0.72"             # Bottom-third — avoids face overlap

# Audio crossfade duration at every clip boundary (seconds)
AUDIO_FADE_DURATION = 0.05

# ── Cut-Type Aware Fade Map ────────────────────────────────────────────────────
# Duration of audio fade (seconds) at EACH boundary, based on the cut type
# detected by Gemini in the original video.
CUT_FADE_MAP: dict[str, float] = {
    "hard_cut":      0.02,  # Very short fade to prevent pops without volume dips
    "smash_cut":     0.0,   # Zero — preserve dramatic energy completely
    "invisible_cut": 0.0,   # Zero — NEVER modify, cut is already seamless
    "jump_cut":      0.05,  # Short fade
    "j_cut":         0.1,   # Keep J-cut crossover short to avoid speech overlaps
    "l_cut":         0.1,   # Keep L-cut crossover short to avoid speech overlaps
    "action_cut":    0.05,  # Smooth but fast transition
    "match_cut":     0.02,  # Keep match cuts crisp
    "montage_cut":   0.05,  # Snappy montage cuts
    "default":       0.05,  # Standard UGC-style short fade — prevents volume dips
}

# Video crossfade duration (seconds) - added to create visual flow
VIDEO_FADE_DURATION = 0.15


# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_font() -> str:
    """Find the best available font for text overlay on this system."""
    for path in FONT_PATHS:
        if Path(path).exists():
            return path
    return ""


def wrap_text(text: str, max_chars: int = 28) -> str:
    """Wrap text to keep line widths suitable for 9:16 vertical video layout."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + (1 if current_line else 0) <= max_chars:
            current_line.append(word)
            current_length += len(word) + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def _run_ffmpeg(cmd: list, step_name: str) -> None:
    """Run an FFmpeg command and raise RuntimeError if it fails."""
    ffmpeg = find_ffmpeg()
    full_cmd = [ffmpeg] + cmd

    logger.info(f"  FFmpeg [{step_name}]: running...")

    result = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed at [{step_name}]:\n{result.stderr[-2000:]}"
        )
    logger.info(f"  FFmpeg [{step_name}]: done ✓")


# ── Boundary Cut Info Helper ──────────────────────────────────────────────────
def get_boundary_cut_info(actual_timestamp, hook_end_ref, demo_end_ref, hook_bnd, demo_bnd, tolerance=0.3):
    if abs(actual_timestamp - hook_end_ref) <= tolerance:
        return hook_bnd.get("cut_type", "default"), abs(hook_bnd.get("audio_offset_seconds", 0.0))
    if abs(actual_timestamp - demo_end_ref) <= tolerance:
        return demo_bnd.get("cut_type", "default"), abs(demo_bnd.get("audio_offset_seconds", 0.0))
    return "default", 0.0


# ── Segment Helper for AV Offset ──────────────────────────────────────────────
def get_segment_for_clip(clip_start: float, clip_end: float, segments_meta: dict | None) -> dict:
    if not segments_meta:
        return {"av_offset_seconds": 0.0}
    for name, seg in segments_meta.items():
        if seg.get("start", 0.0) <= clip_start + 0.1 < seg.get("end", 999.0):
            return seg
    return {"av_offset_seconds": 0.0}


def get_corrected_audio_trim(video_start: float, video_end: float, av_offset_seconds: float):
    audio_start = video_start + av_offset_seconds
    audio_end = video_end + av_offset_seconds
    if audio_start < 0.0:
        logger.warning(
            f"AV-offset correction clamped: audio_start {audio_start:.2f}s -> 0.0s "
            f"(offset {av_offset_seconds:.2f}s may be too large for this clip's position)"
        )
        audio_start = 0.0
    return audio_start, audio_end


# ── Interval Subtraction Helper ────────────────────────────────────────────────
def subtract_interval(
    base_intervals: list[tuple[float, float]],
    subtractor: tuple[float, float]
) -> list[tuple[float, float]]:
    """
    Subtract a subtractor interval [sub_start, sub_end] from a list of base intervals.
    Returns a list of remaining intervals, filtering out any parts shorter than 0.2s.
    """
    sub_start, sub_end = subtractor
    result = []

    for start, end in base_intervals:
        if end <= sub_start or start >= sub_end:
            result.append((start, end))
        elif start < sub_start and end > sub_end:
            result.append((start, sub_start))
            result.append((sub_end, end))
        elif start >= sub_start and start < sub_end and end > sub_end:
            result.append((sub_end, end))
        elif start < sub_start and end > sub_start and end <= sub_end:
            result.append((start, sub_start))

    return [inv for inv in result if (inv[1] - inv[0]) >= 0.2]


# ── Boundary Snapping ──────────────────────────────────────────────────────────
def snap_to_clean_boundary(timestamp: float, cut_analysis: dict) -> float:
    """
    Adjust a timestamp to avoid bad cut points identified by Gemini:
      - Jump cut locations (jarring if we cut mid-sequence)
      - Action cut risks (subject mid-motion)
      - Montage sequences (never cut mid-montage — snap to before/after)
      - Invisible cuts (preserve seamlessness — don't alter their timing)
    """
    if not cut_analysis:
        return timestamp

    # Avoid jump cut locations (within 0.5s snap window)
    jump_cuts = cut_analysis.get("jump_cut_locations", [])
    for jc_time in jump_cuts:
        if abs(timestamp - jc_time) < 0.5:
            snapped = round(jc_time + 0.08, 3)
            logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (jump cut at {jc_time:.2f}s)")
            return snapped

    # Avoid action cut risks (within 0.3s snap window)
    action_risks = cut_analysis.get("action_cut_risks", [])
    for risk in action_risks:
        if abs(timestamp - risk.get("timestamp", -999)) < 0.3:
            snapped = round(risk.get("safe_timestamp", timestamp), 3)
            logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (mid-action: {risk.get('description', '')})")
            return snapped

    # Never cut mid-montage — snap to before start or after end
    montages = cut_analysis.get("montage_sequences", [])
    for m in montages:
        m_start = m.get("start", -1)
        m_end   = m.get("end",   -1)
        if m_start < timestamp < m_end:
            if (timestamp - m_start) < (m_end - timestamp):
                snapped = round(max(0.0, m_start - 0.05), 3)
                logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (before montage {m_start:.2f}s–{m_end:.2f}s)")
            else:
                snapped = round(m_end + 0.05, 3)
                logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (after montage {m_start:.2f}s–{m_end:.2f}s)")
            return snapped

    # Invisible cuts — move AWAY from the cut point.
    invisible = cut_analysis.get("invisible_cuts", [])
    for ic in invisible:
        ic_ts = ic.get("timestamp", -999)
        if abs(timestamp - ic_ts) < 0.3:
            snapped = round(ic_ts + 0.4, 3)
            logger.info(f"  Boundary moved AWAY from invisible cut: {timestamp:.2f}s → {snapped:.2f}s")
            return snapped

    # Match cuts — snap TO the exact timestamp for frame-perfect visual match
    match_cuts = cut_analysis.get("match_cuts", [])
    for mc in match_cuts:
        mc_ts = mc.get("timestamp", -999)
        if abs(timestamp - mc_ts) < 0.4:
            snapped = round(mc_ts, 3)
            logger.info(f"  Boundary snapped TO match cut: {timestamp:.2f}s → {snapped:.2f}s ({mc.get('connected_by', '')})")
            return snapped

    return timestamp


# ── Core: Filter Complex Single-Pass Renderer ──────────────────────────────────────────────
def render_variant_filter_complex(
    source_video: str,
    segments: list[dict],
    output_path: str,
    txt_dir: str
) -> None:
    """
    Render a complete ad variant in a SINGLE FFmpeg filter_complex pass.

    Real J-Cut / L-Cut via independent audio trim timestamps:
      J-Cut: seg[i] audio ends early, seg[i+1] audio starts early from source
      L-Cut: seg[i] audio ends late,  seg[i+1] audio starts late from source
    Video and Audio use SEPARATE concat chains for true AV boundary independence.

    Segment dict fields:
      start (float), end (float), audio_start (float), audio_end (float),
      text (str|None), cut_type (str), audio_offset (float)
    """
    font_path    = _find_font()
    n            = len(segments)
    filter_parts = []

    # -- Phase 1: Pre-compute per-segment audio trim points --
    audio_starts = [seg.get("audio_start", seg["start"]) for seg in segments]
    audio_ends   = [seg.get("audio_end", seg["end"]) for seg in segments]

    for i in range(n):
        if i > 0:
            # Symmetrical J/L cuts use cut_type of the boundary between i-1 and i (stored on segments[i-1])
            cut_type = segments[i-1].get("cut_type", "default")
            aud_offset = float(segments[i-1].get("audio_offset", 0.0))
            if cut_type in ("j_cut", "l_cut") and aud_offset < 0.1:
                aud_offset = 0.35

            if cut_type == "j_cut" and aud_offset > 0:
                original_start = audio_starts[i]
                new_start = max(0.0, original_start - aud_offset)
                actual_shift = original_start - new_start
                audio_starts[i] = new_start
                audio_ends[i - 1] = audio_ends[i - 1] - actual_shift
                logger.info(f"  Boundary J-cut applied between Seg {i-1} and {i} with shift {actual_shift:.2f}s")
            elif cut_type == "l_cut" and aud_offset > 0:
                audio_starts[i] = audio_starts[i] + aud_offset
                audio_ends[i - 1] = audio_ends[i - 1] + aud_offset
                logger.info(f"  Boundary L-cut applied between Seg {i-1} and {i} with shift {aud_offset:.2f}s")

    # Round all values safely to 3 decimal places
    audio_starts = [max(0.0, round(t, 3)) for t in audio_starts]
    audio_ends   = [max(audio_starts[k] + 0.1, round(t, 3)) for k, t in enumerate(audio_ends)]

    # ── Safety Check: Verify total AV duration match ──
    total_v_dur = sum(seg["end"] - seg["start"] for seg in segments)
    total_a_dur = sum(audio_ends[k] - audio_starts[k] for k in range(n))
    if abs(total_v_dur - total_a_dur) > 0.05:
        logger.warning(
            f"  ⚠️ Audio/Video duration mismatch detected after J/L shifts: "
            f"Video={total_v_dur:.3f}s, Audio={total_a_dur:.3f}s. "
            f"Falling back to default unshifted audio trims to prevent desync."
        )
        audio_starts = [seg.get("audio_start", seg["start"]) for seg in segments]
        audio_ends   = [seg.get("audio_end", seg["end"]) for seg in segments]
        audio_starts = [max(0.0, round(t, 3)) for t in audio_starts]
        audio_ends   = [max(audio_starts[k] + 0.1, round(t, 3)) for k, t in enumerate(audio_ends)]

    # -- Phase 2: Build per-segment video + audio filter chains --
    for i, seg in enumerate(segments):
        start    = seg["start"]
        end      = seg["end"]
        duration = max(0.1, round(end - start, 3))

        # Outgoing boundary cut type (boundary after this segment) is segments[i]["cut_type"]
        # Incoming boundary cut type (boundary before this segment) is segments[i-1]["cut_type"] if i > 0 else "default"
        out_cut_type = seg.get("cut_type", "default")
        in_cut_type  = segments[i-1].get("cut_type", "default") if i > 0 else "default"

        # ── Video filter chain for this segment ──
        vchain = (
            f"[0:v]trim=start={start}:end={end},"
            f"setpts=PTS-STARTPTS,"
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        )

        if seg.get("text"):
            txt_path = str(Path(txt_dir) / f"seg_{i}.txt")
            wrapped  = wrap_text(seg["text"], max_chars=28)
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(wrapped)

            escaped_txt  = txt_path.replace("\\", "/").replace(":", "\\:")
            escaped_font = font_path.replace("\\", "/").replace(":", "\\:") if font_path else ""

            # Dynamic fade-out: start fade-out exactly 0.5s before segment ends
            fade_start = max(0.5, duration - 0.5)
            drawtext = (
                f"drawtext="
                f"textfile='{escaped_txt}':"
                f"fontsize={TEXT_FONT_SIZE}:"
                f"fontcolor={TEXT_COLOR}:"
                f"x=(w-text_w)/2:"
                f"y={TEXT_POSITION_Y}:"
                f"box=1:"
                f"boxcolor={TEXT_BOX_COLOR}:"
                f"boxborderw={TEXT_BOX_PADDING}:"
                f"line_spacing=8:"
                f"bordercolor=black@0.8:borderw=2:"
                f"alpha='if(lt(t,{fade_start:.2f}),1,max(0,1-(t-{fade_start:.2f})/0.5))'"
            )
            if escaped_font:
                drawtext = f"drawtext=fontfile='{escaped_font}':" + drawtext.split("drawtext=", 1)[1]

            vchain += f",{drawtext}"

        # ── Video fade at segment boundaries (symmetrical) ──
        vid_fade_d = 0.08
        if i < n - 1 and out_cut_type not in ("smash_cut", "invisible_cut", "hard_cut"):
            vchain += f",fade=t=out:st={max(0.0, duration - vid_fade_d):.3f}:d={vid_fade_d:.3f}"
        if i > 0 and in_cut_type not in ("smash_cut", "invisible_cut", "hard_cut"):
            vchain += f",fade=t=in:st=0:d={vid_fade_d:.3f}"

        filter_parts.append(f"{vchain}[v{i}]")

        # -- Audio chain (J/L-adjusted trim points with symmetrical fades) --
        a_start  = audio_starts[i]
        a_end    = audio_ends[i]
        a_dur    = max(0.1, round(a_end - a_start, 3))
        
        out_base_fade = CUT_FADE_MAP.get(out_cut_type, CUT_FADE_MAP["default"])
        out_fade_d    = min(out_base_fade, a_dur / 3)
        
        in_base_fade  = CUT_FADE_MAP.get(in_cut_type, CUT_FADE_MAP["default"])
        in_fade_d     = min(in_base_fade, a_dur / 3)

        achain = (
            f"[0:a]atrim=start={a_start}:end={a_end},"
            f"asetpts=PTS-STARTPTS"
        )
        if out_cut_type not in ("j_cut", "l_cut") and i < n - 1:
            achain += f",afade=t=out:st={max(0.0, a_dur - out_fade_d):.3f}:d={out_fade_d:.3f}"
        
        if in_cut_type not in ("j_cut", "l_cut") and i > 0:
            achain += f",afade=t=in:st=0:d={in_fade_d:.3f}"
        elif i == 0:
            achain += f",afade=t=in:st=0:d=0.02"

        filter_parts.append(f"{achain}[a{i}]")

    # -- Phase 3: Separate video and audio concat chains --
    video_inputs = "".join(f"[v{i}]" for i in range(n))
    audio_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(f"{video_inputs}concat=n={n}:v=1:a=0[outv]")
    filter_parts.append(f"{audio_inputs}concat=n={n}:v=0:a=1[outa_raw]")
    filter_parts.append("[outa_raw]aresample=async=1[outa]")

    filter_complex = ";\n".join(filter_parts)

    cmd = [
        "-i", source_video,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]

    _run_ffmpeg(cmd, f"filter_complex → {Path(output_path).name}")


# ── Reference Clip Cutter (for user preview — plain, no text) ─────────────────
def cut_reference_clip(
    source_video: str,
    start: float,
    end: float,
    output_path: str,
    label: str = "clip"
) -> None:
    """
    Cut a plain reference clip (no text) for user preview/download.
    Uses filter_complex for consistent quality and zero encoder delay.
    """
    segments = [{"start": start, "end": end, "text": None}]
    txt_dir  = str(Path(output_path).parent)
    render_variant_filter_complex(
        source_video=source_video,
        segments=segments,
        output_path=output_path,
        txt_dir=txt_dir
    )
    logger.info(f"  Reference clip [{label}] saved: {output_path}")


# ── Single Hook Renderer ───────────────────────────────────────────────────────
def render_single_hook(
    source_video: str,
    hook: dict,
    demo_clip: dict,
    result_clip: dict,
    output_path: str,
    txt_dir: str,
    cut_analysis: dict | None = None,
    segments_meta: dict | None = None
) -> str:
    """
    Render one complete ad variant in a single FFmpeg filter_complex pass.
    Subtracts hook range from demo/result to prevent duplicate footage.
    """
    hook_type    = hook["type"]
    hook_text    = hook.get("new_hook_script", "")
    best_clip    = hook["best_clip"]
    cut_analysis = cut_analysis or {}

    logger.info(f"Rendering hook variant: {hook_type}")
    logger.info(f"  Hook clip: {best_clip['start']:.1f}s → {best_clip['end']:.1f}s")
    logger.info(f"  Text: \"{hook_text}\"")

    # Snap best_clip boundaries to avoid jump cuts and mid-action points
    snapped_start = snap_to_clean_boundary(best_clip["start"], cut_analysis)
    snapped_end   = snap_to_clean_boundary(best_clip["end"],   cut_analysis)

    # ── Safety guard: dynamic warning only ──
    clip_duration = snapped_end - snapped_start
    try:
        from modules.utils import find_ffmpeg
        ffprobe_path = find_ffmpeg().replace("ffmpeg", "ffprobe")
        r = subprocess.run(
            [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", source_video],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", timeout=10
        )
        total_duration = float(r.stdout.strip() or "0")
        max_allowed = total_duration * 0.5
        if total_duration > 0 and clip_duration > max_allowed:
            logger.warning(
                f"  ⚠️ Hook clip {clip_duration:.1f}s is > 50% of total video ({total_duration:.1f}s). "
                f"This may make the ad look identical to the source. "
            )
    except Exception:
        pass

    # Resolve cut type + audio offset for every named boundary
    boundaries = cut_analysis.get("segment_boundaries", {})
    hook_bnd   = boundaries.get("hook_end", {})
    demo_bnd   = boundaries.get("demo_end", {})

    hook_end_ref = demo_clip["start"]
    demo_end_ref = result_clip["start"]

    # Check boundary 0 (end of hook segment)
    cut0_type, cut0_off = get_boundary_cut_info(
        actual_timestamp=snapped_end,
        hook_end_ref=hook_end_ref,
        demo_end_ref=demo_end_ref,
        hook_bnd=hook_bnd,
        demo_bnd=demo_bnd
    )

    # Apply source-level A/V offset correction first
    seg_info = get_segment_for_clip(snapped_start, snapped_end, segments_meta)
    av_offset = seg_info.get("av_offset_seconds", 0.0)
    a_start, a_end = get_corrected_audio_trim(snapped_start, snapped_end, av_offset)

    # Segment 0: hook clip with text overlay
    segments = [{
        "start":        snapped_start,
        "end":          snapped_end,
        "audio_start":  a_start,
        "audio_end":    a_end,
        "text":         hook_text,
        "cut_type":     cut0_type,
        "audio_offset": cut0_off
    }]

    # Subtract hook range from demo + result intervals
    base_intervals = [
        (demo_clip["start"],   demo_clip["end"]),
        (result_clip["start"], result_clip["end"])
    ]
    remaining = subtract_interval(base_intervals, (snapped_start, snapped_end))

    # Assign cut type and audio offset to each remaining segment
    for seg_idx, (start, end) in enumerate(remaining):
        snapped_s = snap_to_clean_boundary(start, cut_analysis)
        snapped_e = snap_to_clean_boundary(end,   cut_analysis)

        # Cut info at the end of this segment
        this_cut_type, this_aud_off = get_boundary_cut_info(
            actual_timestamp=snapped_e,
            hook_end_ref=hook_end_ref,
            demo_end_ref=demo_end_ref,
            hook_bnd=hook_bnd,
            demo_bnd=demo_bnd
        )

        seg_info = get_segment_for_clip(snapped_s, snapped_e, segments_meta)
        av_offset = seg_info.get("av_offset_seconds", 0.0)
        rem_a_start, rem_a_end = get_corrected_audio_trim(snapped_s, snapped_e, av_offset)

        segments.append({
            "start":        snapped_s,
            "end":          snapped_e,
            "audio_start":  rem_a_start,
            "audio_end":    rem_a_end,
            "text":         None,
            "cut_type":     this_cut_type,
            "audio_offset": this_aud_off
        })

    render_variant_filter_complex(
        source_video=source_video,
        segments=segments,
        output_path=output_path,
        txt_dir=txt_dir
    )

    logger.info(f"✓ Rendered: {output_path}")
    return output_path


# ── Main Orchestrator ──────────────────────────────────────────────────────────
def run_rendering_pipeline(
    analysis_json_path: str,
    source_video_path: str,
    output_dir: str,
    progress_callback=None
) -> list[dict]:
    """
    Main rendering orchestrator — reads analysis.json and renders
    one .mp4 per applicable hook using filter_complex single-pass.
    """
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    hooks        = analysis.get("applicable_hooks", [])
    segments     = analysis.get("segments", {})
    demo_clip    = segments.get("demo",   {"start": 0.0, "end": 0.0})
    result_clip  = segments.get("result", {"start": 0.0, "end": 0.0})
    cut_analysis = analysis.get("cut_analysis", {})

    if cut_analysis:
        jc_count = len(cut_analysis.get("jump_cut_locations", []))
        ac_count = len(cut_analysis.get("action_cut_risks", []))
        ins_count = len(cut_analysis.get("insert_shots", []))
        logger.info(f"Cut analysis found: {jc_count} jump cuts, {ac_count} action risks, {ins_count} insert shots")

    if not hooks:
        raise ValueError("No applicable_hooks found in analysis.json")

    logger.info(f"Starting rendering pipeline: {len(hooks)} hooks to render")
    logger.info(f"Source video: {source_video_path}")
    logger.info(f"Output dir:   {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    txt_dir = tempfile.mkdtemp(prefix="ai_video_txt_")
    rendered_videos = []

    try:
        demo_ref_path   = str(Path(output_dir) / "reused_demo_clip.mp4")
        result_ref_path = str(Path(output_dir) / "reused_result_clip.mp4")

        logger.info("\n[Prep] Cutting reference Demo clip...")
        if progress_callback:
            progress_callback(0, len(hooks) + 2, "Prep: Demo reference clip")
        cut_reference_clip(
            source_video=source_video_path,
            start=demo_clip["start"],
            end=demo_clip["end"],
            output_path=demo_ref_path,
            label="reused_demo"
        )

        logger.info("\n[Prep] Cutting reference Result clip...")
        if progress_callback:
            progress_callback(1, len(hooks) + 2, "Prep: Result reference clip")
        cut_reference_clip(
            source_video=source_video_path,
            start=result_clip["start"],
            end=result_clip["end"],
            output_path=result_ref_path,
            label="reused_result"
        )

        for idx, hook in enumerate(hooks):
            hook_type     = hook.get("type", f"hook_{idx}")
            best_clip     = hook.get("best_clip", {})

            # Cross-Cut Validation: Drop hook if best_clip overlaps secondary thread
            cross_cut = cut_analysis.get("cross_cut_threads", {})
            if cross_cut.get("detected"):
                thread_b = cross_cut.get("thread_b", [])
                overlaps_secondary = False
                for interval in thread_b:
                    b_start = interval.get("start", interval[0] if isinstance(interval, list) else 0.0)
                    b_end = interval.get("end", interval[1] if isinstance(interval, list) else 0.0)
                    if not (best_clip.get("end", 0.0) <= b_start or best_clip.get("start", 0.0) >= b_end):
                        overlaps_secondary = True
                        break
                if overlaps_secondary:
                    logger.warning(f"  ⚠️ Skipping hook variant '{hook_type}' because it overlaps secondary cross-cut thread.")
                    continue

            safe_name     = hook_type.lower().replace("/", "_").replace(" ", "_")
            out_filename  = f"ad_variant_{safe_name}_hook.mp4"
            output_path   = str(Path(output_dir) / out_filename)

            if progress_callback:
                progress_callback(idx + 2, len(hooks) + 2, hook_type)

            logger.info(f"\n[{idx+1}/{len(hooks)}] Rendering: {hook_type}")

            render_single_hook(
                source_video=source_video_path,
                hook=hook,
                demo_clip=demo_clip,
                result_clip=result_clip,
                output_path=output_path,
                txt_dir=txt_dir,
                cut_analysis=cut_analysis,
                segments_meta=segments
            )

            rendered_videos.append({
                "type":      hook_type,
                "path":      output_path,
                "filename":  out_filename,
                "hook_text": hook.get("new_hook_script", ""),
            })

        rendered_videos.insert(0, {
            "type":      "Reused Result Clip (Reference Only)",
            "path":      result_ref_path,
            "filename":  "reused_result_clip.mp4",
            "hook_text": (
                f"[Segment: {result_clip['start']:.1f}s → {result_clip['end']:.1f}s] "
                "Full result segment — hook range dynamically removed in each variant."
            )
        })
        rendered_videos.insert(0, {
            "type":      "Reused Demo Clip (Reference Only)",
            "path":      demo_ref_path,
            "filename":  "reused_demo_clip.mp4",
            "hook_text": (
                f"[Segment: {demo_clip['start']:.1f}s → {demo_clip['end']:.1f}s] "
                "Full demo segment — hook range dynamically removed in each variant."
            )
        })

        # ── Insert shots ────
        insert_shots = cut_analysis.get("insert_shots", [])
        for ins_idx, shot in enumerate(insert_shots):
            ins_start = shot.get("start", 0.0)
            ins_end   = shot.get("end",   0.0)
            ins_desc  = shot.get("description", f"insert_{ins_idx}")
            ins_potential = shot.get("hook_potential", "unknown")

            if ins_end - ins_start < 0.2:
                continue

            ins_safe = ins_desc.lower().replace(" ", "_")[:30]
            ins_filename = f"insert_shot_{ins_idx}_{ins_safe}.mp4"
            ins_path = str(Path(output_dir) / ins_filename)

            try:
                cut_reference_clip(
                    source_video=source_video_path,
                    start=ins_start,
                    end=ins_end,
                    output_path=ins_path,
                    label=f"insert_{ins_idx}"
                )
                rendered_videos.append({
                    "type":      f"🔍 Insert Shot (hook_potential={ins_potential})",
                    "path":      ins_path,
                    "filename":  ins_filename,
                    "hook_text": f"[{ins_start:.1f}s → {ins_end:.1f}s] {ins_desc}"
                })
                logger.info(f"  Insert shot saved: {ins_path}")
            except Exception as e:
                logger.warning(f"  Insert shot {ins_idx} failed: {e}")

        # ── Cutaway shots ────
        cutaway_shots = cut_analysis.get("cutaway_shots", [])
        for cut_idx, shot in enumerate(cutaway_shots):
            c_start = shot.get("start", 0.0)
            c_end   = shot.get("end",   0.0)
            c_shows = shot.get("shows", f"cutaway_{cut_idx}")

            if c_end - c_start < 0.2:
                continue

            c_safe     = c_shows.lower().replace(" ", "_")[:30]
            c_filename = f"cutaway_{cut_idx}_{c_safe}.mp4"
            c_path     = str(Path(output_dir) / c_filename)

            try:
                cut_reference_clip(
                    source_video=source_video_path,
                    start=c_start,
                    end=c_end,
                    output_path=c_path,
                    label=f"cutaway_{cut_idx}"
                )
                rendered_videos.append({
                    "type":      "✂️ Cutaway Shot (B-Roll)",
                    "path":      c_path,
                    "filename":  c_filename,
                    "hook_text": f"[{c_start:.1f}s → {c_end:.1f}s] {c_shows}"
                })
                logger.info(f"  Cutaway shot saved: {c_path}")
            except Exception as e:
                logger.warning(f"  Cutaway shot {cut_idx} failed: {e}")

        # ── Reaction shots ────
        reaction_shots = cut_analysis.get("reaction_shots", [])
        for r_idx, shot in enumerate(reaction_shots):
            r_start   = shot.get("start", 0.0)
            r_end     = shot.get("end",   0.0)
            r_emotion = shot.get("emotion",  f"reaction_{r_idx}")
            r_context = shot.get("context",  "")

            if r_end - r_start < 0.2:
                continue

            r_safe     = r_emotion.lower().replace(" ", "_")[:20]
            r_filename = f"reaction_{r_idx}_{r_safe}.mp4"
            r_path     = str(Path(output_dir) / r_filename)

            try:
                cut_reference_clip(
                    source_video=source_video_path,
                    start=r_start,
                    end=r_end,
                    output_path=r_path,
                    label=f"reaction_{r_idx}"
                )
                rendered_videos.append({
                    "type":      f"😮 Reaction Shot ({r_emotion})",
                    "path":      r_path,
                    "filename":  r_filename,
                    "hook_text": f"[{r_start:.1f}s → {r_end:.1f}s] {r_context}"
                })
                logger.info(f"  Reaction shot saved: {r_path}")
            except Exception as e:
                logger.warning(f"  Reaction shot {r_idx} failed: {e}")

        cross_cut = cut_analysis.get("cross_cut_threads", {})
        if cross_cut.get("detected"):
            primary = cross_cut.get("primary_thread", "a")
            logger.warning(
                f"⚠️  Cross-cut (parallel editing) detected in source video! "
                f"Primary thread '{primary}' was used for segment boundaries. "
                f"The other thread has been excluded from output variants."
            )

        if progress_callback:
            progress_callback(len(hooks) + 2, len(hooks) + 2, "done")

    finally:
        shutil.rmtree(txt_dir, ignore_errors=True)
        logger.info("Temp text files cleaned up")

    manifest_path = str(Path(output_dir) / "render_manifest.json")
    save_json(rendered_videos, manifest_path)

    logger.info(f"\n✅ Rendering complete! {len(rendered_videos)} videos saved to {output_dir}")
    return rendered_videos
