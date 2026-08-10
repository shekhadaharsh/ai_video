"""
test_render.py — Standalone test runner for video rendering (Step 5).
Allows testing FFmpeg cutting, text overlay, and stitching locally using
pre-existing analysis JSON and video without calling Gemini or Whisper.
"""

import os
import sys
import json
import logging
from pathlib import Path
from modules.render import run_rendering_pipeline

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[AI-Video-Test] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    json_path  = "test_input.json"
    video_path = "test_video.mp4"
    output_dir = "test_outputs"

    print("\n" + "="*60)
    print("  [AI-Video] Standalone Renderer Test")
    print("="*60 + "\n")

    # 1. Check for test_input.json
    if not os.path.exists(json_path):
        logger.error(f"'{json_path}' not found!")
        print(f"\nPlease create a file named '{json_path}' in this folder.")
        print("Example content:")
        print("""{
  "segments": {
    "demo": {"start": 5.7, "end": 14.3},
    "result": {"start": 14.3, "end": 33.9}
  },
  "applicable_hooks": [
    {
      "type": "Problem",
      "best_clip": {"start": 0.0, "end": 5.7},
      "new_hook_script": "Tired of stubborn tanning that just won't fade away?"
    }
  ]
}""")
        sys.exit(1)

    # 2. Check for test_video.mp4
    if not os.path.exists(video_path):
        logger.error(f"'{video_path}' not found!")
        print(f"\nPlease place your source video file named '{video_path}' in this folder.")
        sys.exit(1)

    # 3. Start rendering
    logger.info("Initializing rendering pipeline...")
    logger.info(f"Input JSON: {json_path}")
    logger.info(f"Input Video: {video_path}")
    logger.info(f"Output Dir: {output_dir}/")

    try:
        # Run rendering using the optimized 1080p rendering module
        rendered = run_rendering_pipeline(
            analysis_json_path=json_path,
            source_video_path=video_path,
            output_dir=output_dir
        )
        print("\n" + "="*60)
        print("  Rendering Successful!")
        print("="*60)
        print(f"\nGenerated {len(rendered)} file(s) in '{output_dir}/':")
        for vid in rendered:
            print(f"  - {vid['filename']} (Type: {vid['type']})")
        print()

    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
