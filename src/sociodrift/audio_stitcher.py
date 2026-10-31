"""Audio stitching and crossfading module using PyDub.

Provides smooth inter-scene crossfading, loudness normalization,
and natural ending fade-outs to prevent abrupt audio cutoffs.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CROSSFADE_MS = 450
DEFAULT_ENDING_FADE_OUT_MS = 900
TARGET_SEGMENT_DBFS = -16.0
TARGET_PEAK_DBFS = -1.0


def normalize_loudness(segment, target_dbfs: float = TARGET_SEGMENT_DBFS):
    """Normalize average audio level to prevent volume jumps between scenes."""
    if len(segment) <= 0 or segment.dBFS == float("-inf"):
        return segment
    gain = max(-8.0, min(8.0, target_dbfs - segment.dBFS))
    return segment.apply_gain(gain)


def prevent_clipping(segment, target_peak: float = TARGET_PEAK_DBFS):
    """Limit peak amplitude to avoid digital clipping/distortion."""
    if len(segment) <= 0 or segment.max_dBFS == float("-inf"):
        return segment
    if segment.max_dBFS > target_peak:
        return segment.apply_gain(target_peak - segment.max_dBFS)
    return segment


def apply_segment_edges(segment, edge_fade_ms: int = 150):
    """Apply gentle fade-in and fade-out at segment boundaries."""
    if len(segment) <= (edge_fade_ms * 2):
        return segment
    return segment.fade_in(edge_fade_ms).fade_out(edge_fade_ms)


def stitch_audio_with_crossfade(
    scene_audio_paths: list[Path | str],
    output_path: Path | str,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    ending_fade_out_ms: int = DEFAULT_ENDING_FADE_OUT_MS,
) -> Path:
    """Stitch multiple scene audio files into a seamless soundtrack with crossfades.

    Args:
        scene_audio_paths: List of filepaths to scene .wav files in order.
        output_path: Destination path for the continuous .wav soundtrack.
        crossfade_ms: Duration in milliseconds for overlapping scene crossfades.
        ending_fade_out_ms: Duration in milliseconds for smooth final fade-out.

    Returns:
        Path to the saved soundtrack.
    """
    from pydub import AudioSegment

    if not scene_audio_paths:
        raise ValueError("No audio paths provided to stitch.")

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    loaded_segments: list[AudioSegment] = []
    for p in scene_audio_paths:
        path_obj = Path(p).resolve()
        if not path_obj.exists():
            logger.warning("Audio file missing: %s", path_obj)
            continue
        seg = AudioSegment.from_file(str(path_obj), format="wav")
        seg = normalize_loudness(seg)
        seg = apply_segment_edges(seg, edge_fade_ms=120)
        loaded_segments.append(seg)

    if not loaded_segments:
        raise ValueError("Failed to load any valid audio segments for stitching.")

    # Combine with equal-power crossfades between scenes
    combined = loaded_segments[0]
    for seg in loaded_segments[1:]:
        effective_crossfade = min(crossfade_ms, len(combined) // 4, len(seg) // 4)
        if effective_crossfade > 50:
            combined = combined.append(seg, crossfade=effective_crossfade)
        else:
            combined = combined + seg

    # Apply smooth natural ending fade-out so it never cuts abruptly
    fade_tail = min(ending_fade_out_ms, len(combined) // 3)
    if fade_tail > 100:
        combined = combined.fade_out(fade_tail)

    combined = prevent_clipping(combined)
    combined.export(str(out), format="wav")
    logger.info(
        "Successfully stitched %d scenes into seamless soundtrack (%.2fs): %s",
        len(loaded_segments),
        len(combined) / 1000.0,
        out.name,
    )
    return out
