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
TEXT_TIME_LIMIT  = 4.0                  # Max seconds text stays on screen

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
        # else: interval is completely inside subtractor → discard

    # Filter out extremely short clips to avoid FFmpeg glitches
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
            # Choose the nearer edge
            if (timestamp - m_start) < (m_end - timestamp):
                snapped = round(max(0.0, m_start - 0.05), 3)
                logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (before montage {m_start:.2f}s–{m_end:.2f}s)")
            else:
                snapped = round(m_end + 0.05, 3)
                logger.info(f"  Boundary snapped: {timestamp:.2f}s → {snapped:.2f}s (after montage {m_start:.2f}s–{m_end:.2f}s)")
            return snapped

    # FIXED: Invisible cuts — move AWAY from the cut point.
    # Our cut placed near an invisible cut would destroy its seamlessness.
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
      start (float), end (float), text (str|None),
      cut_type (str), audio_offset (float)
    """
    font_path    = _find_font()
    n            = len(segments)
    filter_parts = []

    # -- Phase 1: Pre-compute per-segment audio trim points --
    audio_starts = [seg["start"] for seg in segments]
    audio_ends   = [seg["end"] for seg in segments]

    for i in range(n):
        if i > 0:
            cut_type = segments[i].get("cut_type", "default")
            aud_offset = float(segments[i].get("audio_offset", 0.0))
            if cut_type in ("j_cut", "l_cut") and aud_offset < 0.1:
                aud_offset = 0.35

            if cut_type == "j_cut" and aud_offset > 0:
                original_start = segments[i]["start"]
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

    # -- Phase 2: Build per-segment video + audio filter chains --
    for i, seg in enumerate(segments):
        start    = seg["start"]
        end      = seg["end"]
        duration = max(0.1, round(end - start, 3))
        cut_type = seg.get("cut_type", "default")

        # ── Video filter chain for this segment ──
        vchain = (
            f"[0:v]trim=start={start}:end={end},"
            f"setpts=PTS-STARTPTS,"
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        )

        if seg.get("text"):
            # Write text to temp UTF-8 file (avoids CLI encoding issues)
            txt_path = str(Path(txt_dir) / f"seg_{i}.txt")
            wrapped  = wrap_text(seg["text"], max_chars=28)
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(wrapped)

            escaped_txt  = txt_path.replace("\\", "/").replace(":", "\\:")
            escaped_font = font_path.replace("\\", "/").replace(":", "\\:") if font_path else ""

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
                f"bordercolor=black@0.8:borderw=2:"              # Text border for high contrast
                f"alpha='if(lt(t,3.0),1,max(0,1-(t-3.0)/0.5))'"  # Fade out from 3.0s-3.5s
            )
            if escaped_font:
                drawtext = f"fontfile='{escaped_font}':" + drawtext.replace("drawtext=", "drawtext=")
                drawtext = f"drawtext=fontfile='{escaped_font}':" + drawtext.split("drawtext=", 1)[1]

            vchain += f",{drawtext}"

        # ── Video fade at segment boundaries for smooth transitions ──
        vid_fade_d = 0.08  # 80ms video fade — subtle, not a full dissolve
        if cut_type not in ("smash_cut", "invisible_cut", "hard_cut"):
            if i < n - 1:
                vchain += f",fade=t=out:st={max(0.0, duration - vid_fade_d):.3f}:d={vid_fade_d:.3f}"
            if i > 0:
                vchain += f",fade=t=in:st=0:d={vid_fade_d:.3f}"

        filter_parts.append(f"{vchain}[v{i}]")

        # -- Audio chain (J/L-adjusted trim points) --
        a_start  = audio_starts[i]
        a_end    = audio_ends[i]
        a_dur    = max(0.1, round(a_end - a_start, 3))
        base_fade = CUT_FADE_MAP.get(cut_type, CUT_FADE_MAP["default"])
        fade_d   = min(base_fade, a_dur / 3)

        achain = (
            f"[0:a]atrim=start={a_start}:end={a_end},"
            f"asetpts=PTS-STARTPTS"
        )
        # J/L cuts handled via timestamp offset -- no extra fade needed
        if cut_type not in ("j_cut", "l_cut"):
            if i < n - 1:
                achain += f",afade=t=out:st={max(0.0, a_dur - fade_d):.3f}:d={fade_d:.3f}"
            # Apply minimal 0.02s fade-in to the first segment (i=0) to prevent pop.
            # Longer fade-in would mute the start of the first word since frame_analyzer
            # already snaps accurately to silence.end.
            fade_in_d = 0.02 if i == 0 else fade_d
            achain += f",afade=t=in:st=0:d={fade_in_d:.3f}"

        filter_parts.append(f"{achain}[a{i}]")

    # -- Phase 3: Separate video and audio concat chains --
    # Video switches at natural timestamps; audio at J/L-adjusted timestamps.
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
    cut_analysis: dict | None = None
) -> str:
    """
    Render one complete ad variant in a single FFmpeg filter_complex pass.
    Subtracts hook range from demo/result to prevent duplicate footage.
    Uses cut_analysis to:
      - Snap boundaries away from jump cuts and mid-action points
      - Apply cut-type-aware audio fade durations per boundary

    Structure:
      [New hook text overlay clip] + [Remaining demo/result segments]
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

    # Safety guard: detect if hook clip is unreasonably long (> 50% of total video)
    # This catches the edge case where Gemini ignores the 60% segment rule and
    # selects the full video as a hook, making the ad look identical to the source.
    # We do NOT clamp to a fixed duration — the clip length is determined by natural
    # speech boundaries and can be anywhere from 3s to 15s depending on the video.
    clip_duration = snapped_end - snapped_start
    try:
        import subprocess
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
                f"  Hook clip {clip_duration:.1f}s is > 50% of total video ({total_duration:.1f}s). "
                f"This may make the ad look identical to the source. "
                f"Re-analyse the video with a stricter prompt if this persists."
            )
    except Exception:
        pass  # Guard is best-effort; never block rendering

    # Resolve cut type + audio offset for every named boundary
    boundaries = cut_analysis.get("segment_boundaries", {})

    hook_bnd      = boundaries.get("hook_end",    {})
    hook_cut_type = hook_bnd.get("cut_type",           "default")
    hook_aud_off  = abs(hook_bnd.get("audio_offset_seconds", 0.0))

    demo_bnd      = boundaries.get("demo_end",    {})
    demo_cut_type = demo_bnd.get("cut_type",           "default")
    demo_aud_off  = abs(demo_bnd.get("audio_offset_seconds", 0.0))

    # Segment 0: hook clip with text overlay + cut info at its END boundary
    segments = [{
        "start":        snapped_start,
        "end":          snapped_end,
        "text":         hook_text,
        "cut_type":     hook_cut_type,
        "audio_offset": hook_aud_off
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

        # First remaining segment: transition from hook (cut type already on hook seg)
        # Subsequent segments: use demo_end cut type (demo→result boundary)
        this_cut_type = "default" if seg_idx == 0 else demo_cut_type
        this_aud_off  = 0.0       if seg_idx == 0 else demo_aud_off

        segments.append({
            "start":        snapped_s,
            "end":          snapped_e,
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

    Args:
        analysis_json_path:  Path to analysis.json from Gemini step
        source_video_path:   Path to the original source video
        output_dir:          Directory to save final output videos
        progress_callback:   Optional callable(current, total, hook_type)

    Returns:
        List of dicts: [{"type": str, "path": str, "filename": str, "hook_text": str}]
    """
    with open(analysis_json_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    hooks       = analysis.get("applicable_hooks", [])
    segments    = analysis.get("segments", {})
    demo_clip   = segments.get("demo",   {"start": 0.0, "end": 0.0})
    result_clip = segments.get("result", {"start": 0.0, "end": 0.0})
    cut_analysis = analysis.get("cut_analysis", {})

    # Log cut analysis summary if available
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

    # Temp dir for drawtext text files (auto-cleaned)
    txt_dir = tempfile.mkdtemp(prefix="ai_video_txt_")
    rendered_videos = []

    try:
        # ── Reference clips for user preview ─────────────────────────────────
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

        # ── Render each hook variant ──────────────────────────────────────────
        for idx, hook in enumerate(hooks):
            hook_type     = hook.get("type", f"hook_{idx}")
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
                cut_analysis=cut_analysis
            )

            rendered_videos.append({
                "type":      hook_type,
                "path":      output_path,
                "filename":  out_filename,
                "hook_text": hook.get("new_hook_script", ""),
            })

        # ── Add reference clips to manifest ──────────────────────────────────
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

        # ── Insert shots from cut_analysis → cut them as reference clips ────
        insert_shots = cut_analysis.get("insert_shots", [])
        for ins_idx, shot in enumerate(insert_shots):
            ins_start = shot.get("start", 0.0)
            ins_end   = shot.get("end",   0.0)
            ins_desc  = shot.get("description", f"insert_{ins_idx}")
            ins_potential = shot.get("hook_potential", "unknown")

            if ins_end - ins_start < 0.2:
                continue  # Too short to be useful

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

        # ── Cutaway shots → cut as B-roll reference clips ─────────────────────
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

        # ── Reaction shots → cut as social proof reference clips ───────────────
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

        # ── Cross-cut threads: log warning if detected ─────────────────────────
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
        # Always clean up temp text files
        shutil.rmtree(txt_dir, ignore_errors=True)
        logger.info("Temp text files cleaned up")

    # Save render manifest
    manifest_path = str(Path(output_dir) / "render_manifest.json")
    save_json(rendered_videos, manifest_path)

    logger.info(f"\n✅ Rendering complete! {len(rendered_videos)} videos saved to {output_dir}")
    return rendered_videos
