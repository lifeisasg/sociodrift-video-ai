"""Lossless video and audio multiplexing module.

Merges generated AI soundtrack waveforms with source video files while preserving
100% original video quality bit-for-bit using stream copying (zero generational loss).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_ffmpeg_binary() -> str:
    """Resolve the best available FFmpeg binary (bundled or system)."""
    # 1. Check bundled imageio-ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass

    # 2. Check system PATH
    system_exe = shutil.which("ffmpeg")
    if system_exe:
        return system_exe

    raise RuntimeError(
        "FFmpeg is required to merge audio with video. "
        "Install via `pip install imageio-ffmpeg` or `brew install ffmpeg`."
    )


def merge_video_and_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    audio_bitrate: str = "320k",
) -> Path:
    """Merge an audio file into a video container while preserving original video stream.

    Uses stream-copying (`-c:v copy`) for the video channel so that no re-encoding,
    compression artifacts, or quality degradation occurs.

    Args:
        video_path: Source input video file (.mp4, .mov, .avi, etc.).
        audio_path: Synchronized AI soundtrack (.wav).
        output_path: Destination path for the final synced video.
        audio_bitrate: AAC audio encoding bitrate (default: 320k for studio quality).

    Returns:
        Path to the generated output video file.

    Raises:
        FileNotFoundError: If input video or audio files are missing.
        RuntimeError: If FFmpeg muxing fails.
    """
    v_path = Path(video_path).resolve()
    a_path = Path(audio_path).resolve()
    out = Path(output_path).resolve()

    if not v_path.exists():
        raise FileNotFoundError(f"Source video file not found: {v_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Soundtrack audio file not found: {a_path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = resolve_ffmpeg_binary()

    command = [
        ffmpeg_exe,
        "-y",               # Overwrite destination if exists
        "-i", str(v_path),  # Input 0: original video
        "-i", str(a_path),  # Input 1: AI music soundtrack
        "-c:v", "copy",     # Stream-copy video: 100% BIT-FOR-BIT LOSSLESS (no re-encoding)
        "-c:a", "aac",      # Encode audio as high-definition AAC
        "-b:a", audio_bitrate,
        "-map", "0:v:0",    # Take primary video stream from input 0
        "-map", "1:a:0",    # Take primary audio stream from input 1
        "-shortest",        # Truncate to whichever stream finishes first
        str(out),
    ]

    logger.info("Muxing video with lossless stream copy: %s + %s -> %s", v_path.name, a_path.name, out.name)
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("FFmpeg error: %s", result.stderr)
        raise RuntimeError(f"FFmpeg muxing failed: {result.stderr.strip() or 'Unknown error'}")

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced an empty or missing output video: {out}")

    logger.info("Successfully merged video and audio: %s (%.1f KB)", out.name, out.stat().st_size / 1024.0)
    return out
