"""
utils.py — Shared helper functions for the AI Video Pipeline.
Handles: video ID generation, directory setup, JSON I/O, video validation.
"""

import os
import json
import shutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[AI-Video] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".webm"}
DATA_DIR = Path(__file__).parent.parent / "data"

# Common Windows FFmpeg install locations (winget, chocolatey, manual)
FFMPEG_SEARCH_PATHS = [
    r"C:\Users\harsh\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]
FFPROBE_SEARCH_PATHS = [
    r"C:\Users\harsh\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe",
    r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
]


# ── FFmpeg Binary Finder ───────────────────────────────────────────────────────
def find_ffmpeg() -> str:
    """
    Find the ffmpeg binary path.
    Checks PATH first, then common Windows install locations.
    Returns the full path string.
    Raises RuntimeError if not found.
    """
    # Try PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Try known install paths
    for p in FFMPEG_SEARCH_PATHS:
        if Path(p).exists():
            return p
    raise RuntimeError(
        "ffmpeg not found! Please restart your terminal so PATH is updated, "
        "or install ffmpeg via: winget install ffmpeg"
    )


def find_ffprobe() -> str:
    """
    Find the ffprobe binary path.
    Checks PATH first, then common Windows install locations.
    Returns the full path string.
    Raises RuntimeError if not found.
    """
    found = shutil.which("ffprobe")
    if found:
        return found
    for p in FFPROBE_SEARCH_PATHS:
        if Path(p).exists():
            return p
    raise RuntimeError(
        "ffprobe not found! Please restart your terminal so PATH is updated."
    )


# ── Video Duration ─────────────────────────────────────────────────────────────
def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    """
    ffprobe_bin = find_ffprobe()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    logger.info(f"Video duration: {duration:.1f}s")
    return duration


# ── Video ID ───────────────────────────────────────────────────────────────────
def generate_video_id() -> str:
    """Generate a unique, human-readable video ID based on current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Directory Setup ────────────────────────────────────────────────────────────
def setup_data_dirs(video_id: str) -> dict:
    """
    Create all required directories for a pipeline run.
    Returns a dict of important paths for this video_id.
    """
    base = DATA_DIR / video_id
    frames_dir = base / "frames"

    base.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "base":          str(base),
        "audio":         str(base / "audio.mp3"),
        "transcript":    str(base / "transcript.json"),
        "frames_dir":    str(frames_dir),
        "frame_mapping": str(base / "frame_mapping.json"),
        "analysis":      str(base / "analysis.json"),
        "cost_report":   str(base / "cost_report.json"),
    }
    logger.info(f"Data directory created: {base}")
    return paths


def get_paths(video_id: str) -> dict:
    """Return path dict for an existing video_id (no directory creation)."""
    base = DATA_DIR / video_id
    frames_dir = base / "frames"
    return {
        "base":          str(base),
        "audio":         str(base / "audio.mp3"),
        "transcript":    str(base / "transcript.json"),
        "frames_dir":    str(frames_dir),
        "frame_mapping": str(base / "frame_mapping.json"),
        "analysis":      str(base / "analysis.json"),
        "cost_report":   str(base / "cost_report.json"),
    }


# ── Video Validation ───────────────────────────────────────────────────────────
def validate_video_file(video_path: str) -> tuple[bool, str]:
    """
    Validate that the video file exists and is a supported format.
    Returns (is_valid, error_message). error_message is empty string if valid.
    """
    path = Path(video_path)

    if not path.exists():
        return False, f"File not found: {video_path}"

    if not path.is_file():
        return False, f"Path is not a file: {video_path}"

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        return False, (
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )

    if path.stat().st_size == 0:
        return False, f"File is empty: {video_path}"

    return True, ""


# ── JSON Helpers ───────────────────────────────────────────────────────────────
def save_json(data: dict | list, file_path: str) -> None:
    """Save data as pretty-printed JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON saved: {file_path}")


def load_json(file_path: str) -> dict | list:
    """Load and parse a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_json_response(text: str) -> str:
    """
    Strip markdown code fences and sanitize JSON from Gemini API responses.
    Handles: ```json ... ```, unquoted keys, single quotes, trailing commas, and boundary extraction.
    """
    import re

    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Extract JSON object boundaries { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    # Try standard json parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Sanitize common dirty JSON syntax issues
    # 1. Fix missing commas between properties/objects separated by newlines
    text = re.sub(r'("|\d|true|false|null|\}|\])\s*\n\s*("|\{)', r'\1,\n\2', text)
    # 2. Remove trailing commas before } or ]
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    # 3. Fix unquoted keys: { key: -> { "key":
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    # 4. Fix single-quoted strings: 'val' -> "val"
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)

    return text


# ── File Size Helper ───────────────────────────────────────────────────────────
def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
