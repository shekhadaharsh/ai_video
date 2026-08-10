"""
compressor.py — local video compression for Gemini upload.
Shrinks the source video to a 720p copy to minimize upload bandwidth and time,
while retaining clear audio and visual context for the Gemini model.
"""

import subprocess
import logging
from pathlib import Path
from modules.utils import find_ffmpeg

logger = logging.getLogger(__name__)


def compress_video_to_720p(video_path: str, output_path: str) -> str:
    """
    Compress source video to 720p resolution (max 1280px on the larger dimension).
    Maintains vertical (9:16) or horizontal (16:9) aspect ratios.
    Uses lower video/audio bitrates to optimize file size (aims for 1-5MB).

    Args:
        video_path:  Path to the high-quality source video
        output_path: Path where the compressed video should be saved

    Returns:
        output_path
    """
    ffmpeg_bin = find_ffmpeg()
    
    # Generic scale filter:
    # If width > height (landscape): scale width to 1280, height auto-calculate (divisible by 2)
    # If height > width (portrait): scale height to 1280, width auto-calculate (divisible by 2)
    # This ensures a clean 720p HD output for both vertical and horizontal layouts.
    scale_filter = "scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'"
    
    cmd = [
        ffmpeg_bin,
        "-i", video_path,
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",           # Standard compression CRF (higher = smaller size)
        "-c:a", "aac",
        "-b:a", "64k",          # 64kbps is perfect for LLM vocal analysis
        "-ac", "1",             # Downmix to mono for even smaller size
        "-y",
        output_path
    ]
    
    logger.info(f"Compressing video copy for Gemini upload...")
    logger.info(f"  Source: {video_path} ({Path(video_path).stat().st_size / 1024 / 1024:.1f} MB)")
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Video compression failed:\n{result.stderr[-1000:]}"
        )
        
    compressed_size = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f"✓ Compression complete: {output_path} ({compressed_size:.2f} MB)")
    return output_path
