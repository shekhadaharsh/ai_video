"""
run_pipeline.py — CLI entry point for the AI Video Hook Generator pipeline (Production-Grade).

Usage:
    python run_pipeline.py --video path/to/video.mp4

Steps:
    1. Validate video input
    2. Compress copy locally to 720p (bandwidth optimization)
    3. Upload to Gemini File API and run native analysis
    4. Run frame-accurate FFmpeg editing & concatenation to render final ad variants
"""

import argparse
import sys
import logging
from pathlib import Path

from modules.utils import (
    generate_video_id,
    setup_data_dirs,
    validate_video_file,
    load_json,
    get_video_duration
)
from modules.compressor      import compress_video_to_720p
from modules.vision_analysis import run_vision_analysis
from modules.render          import run_rendering_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[AI-Video] %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    print("\n" + "="*55)
    print("  🎬 AI Video Hook Generator — Pipeline Runner")
    print("="*55 + "\n")


def run_pipeline(video_path: str):
    """
    Run the full end-to-end pipeline (Steps 1–5).

    Args:
        video_path: Path to input video file
    """
    print_banner()

    # ── Step 1: Validate video ────────────────────────────────────────────────
    logger.info(f"Input video: {video_path}")
    is_valid, error_msg = validate_video_file(video_path)
    if not is_valid:
        logger.error(f"Video validation failed: {error_msg}")
        sys.exit(1)

    # Generate unique ID + create directories
    video_id = generate_video_id()
    paths    = setup_data_dirs(video_id)
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Data directory: {paths['base']}\n")

    # Get duration early (needed for compression + prompt)
    try:
        video_duration = get_video_duration(video_path)
    except Exception as e:
        logger.error(f"Could not read video duration: {e}")
        sys.exit(1)

    # ── Step 2: Compress locally ──────────────────────────────────────────────
    logger.info("Step 1/4: Compressing video copy locally to 720p...")
    try:
        compressed_path = str(Path(paths["base"]) / "compressed_720p.mp4")
        compress_video_to_720p(video_path, compressed_path)
        logger.info("✓ Local compression complete\n")
    except Exception as e:
        logger.error(f"Local compression failed: {e}")
        sys.exit(1)

    # ── Step 3: Gemini Native Video Analysis ──────────────────────────────────
    logger.info("Step 2/4: Running Gemini native video analysis...")
    logger.info("  (Uploading copy and analysis. This may take 30-45 seconds...)")
    try:
        analysis = run_vision_analysis(
            video_id=video_id,
            video_duration=video_duration,
            compressed_video_path=compressed_path
        )
        logger.info("✓ Gemini analysis complete!\n")
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        sys.exit(1)

    # ── Step 4: Render Final Ad Videos ────────────────────────────────────────
    logger.info("Step 3/4: Rendering final ad variant videos...")
    output_dir = str(Path(paths["base"]) / "outputs")
    try:
        rendered = run_rendering_pipeline(
            analysis_json_path=paths["analysis"],
            source_video_path=video_path,
            output_dir=output_dir
        )
        logger.info("✓ Rendering complete!\n")
    except Exception as e:
        logger.error(f"Rendering pipeline failed: {e}")
        sys.exit(1)

    # ── Results Summary ───────────────────────────────────────────────────────
    cost_report = load_json(paths["cost_report"])

    print("\n" + "="*55)
    print("  ✅ Pipeline Complete!")
    print("="*55)
    
    print(f"\n  🎯 Generated Files in {output_dir}:")
    for vid in rendered:
        print(f"     • {vid['filename']} (Type: {vid['type']})")

    print(f"\n  💰 Gemini Cost:")
    print(f"     Input:  {cost_report.get('input_tokens', 0):,} tokens  = ${cost_report.get('input_cost_usd', 0):.4f}")
    print(f"     Output: {cost_report.get('output_tokens', 0):,} tokens = ${cost_report.get('output_cost_usd', 0):.4f}")
    print(f"     Total:  ${cost_report.get('total_cost_usd', 0):.4f}  (₹{cost_report.get('total_cost_inr', 0):.2f})")

    print(f"\n  📁 Data Directories:")
    print(f"     Base data folder: {paths['base']}")
    print(f"     Analysis JSON:    {paths['analysis']}")
    print(f"     Cost Report:      {paths['cost_report']}")
    print()

    return analysis


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Video Hook Generator — Generate ad hook variants from a source video natively."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the high-quality source video file."
    )

    args = parser.parse_args()
    run_pipeline(video_path=args.video)
