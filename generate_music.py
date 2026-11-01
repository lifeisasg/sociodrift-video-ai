"""Generate real cinematic music waveforms from Video AI prompts using Meta's MusicGen.

Uses Meta's official open-source musicgen-small model via Hugging Face Transformers.
Leverages Apple Silicon GPU acceleration (MPS) when available, or CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from sociodrift.audio_stitcher import stitch_audio_with_crossfade
    from sociodrift.video_audio_merger import merge_video_and_audio
except ImportError:
    stitch_audio_with_crossfade = None
    merge_video_and_audio = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("musicgen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real .wav music tracks from SocioDrift Video AI prompts using Meta MusicGen."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="outputs/sample_video.json",
        help="Path to pipeline output JSON file (default: outputs/sample_video.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/audio",
        help="Directory to save generated .wav files (default: outputs/audio)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/musicgen-small",
        help="Hugging Face model ID (default: facebook/musicgen-small)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=500,
        help="Crossfade overlap in milliseconds between adjacent scenes (default: 500)",
    )
    parser.add_argument(
        "--fade-out-ms",
        type=int,
        default=1000,
        help="Ending fade-out in milliseconds to prevent abrupt ending (default: 1000)",
    )
    parser.add_argument(
        "--merge-video",
        action="store_true",
        help="Automatically merge generated audio back into the source video using FFmpeg",
    )
    return parser.parse_args()


def generate_procedural_scene_audio(scene: dict, duration: float, sampling_rate: int = 32000):
    """Generate harmonic procedural cinematic audio for cloud hosts with constrained RAM (Render Free)."""
    import numpy as np

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    mood = scene.get("mood", "calm").lower()
    cue = scene.get("music_cue", {})
    tempo = float(cue.get("tempo", 110))
    beat_freq = tempo / 60.0

    if "suspense" in mood or "horror" in mood:
        # Dark low drone + dissonant tension chords
        f1, f2, f3 = 55.0, 58.27, 110.0
        wave = 0.4 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.2 * t)
        wave = wave * envelope
    elif "energetic" in mood or "action" in mood:
        # Driving synth bass + rhythm pulse
        f1, f2 = 110.0, 220.0
        rhythm = (0.5 * (1 + np.sin(2 * np.pi * beat_freq * t))) ** 2
        wave = (0.4 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)) * rhythm
    else:
        # Calm / Peaceful warm ambient chord
        f1, f2, f3 = 130.81, 164.81, 196.00
        wave = 0.35 * np.sin(2 * np.pi * f1 * t) + 0.25 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)
        wave = wave * (0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t))

    fade_len = int(sampling_rate * 0.05)
    if len(wave) > 2 * fade_len:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        wave[:fade_len] *= fade_in
        wave[-fade_len:] *= fade_out

    return wave.astype(np.float32)


def main() -> int:
    args = parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        logger.error("JSON file not found: %s", json_path)
        return 1

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    if not scenes:
        logger.error("No scenes found in %s", json_path)
        return 1

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    processor = None
    device = "cpu"
    sampling_rate = 32000

    try:
        import numpy as np
        import scipy.io.wavfile
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        # Determine hardware acceleration
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon Metal Performance Shaders (MPS) GPU acceleration!")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using NVIDIA CUDA GPU acceleration!")
        else:
            device = "cpu"
            logger.info("Using CPU for neural music generation.")

        logger.info("Loading Meta MusicGen model: %s ...", args.model)
        try:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
            model = MusicgenForConditionalGeneration.from_pretrained(args.model, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(args.model)
            model = MusicgenForConditionalGeneration.from_pretrained(args.model)
        model.to(device)
        sampling_rate = model.config.audio_encoder.sampling_rate
    except Exception as exc:
        import numpy as np
        import scipy.io.wavfile
        logger.warning(
            "Neural audio synthesis not available (%s). "
            "Using harmonic procedural audio synthesizer (lightweight cloud mode).",
            exc,
        )
    generated_audio_paths: list[Path] = []
    all_audio_chunks: list[np.ndarray] = []

    logger.info("Generating music for %d scenes...", len(scenes))

    for scene in scenes:
        scene_id = scene["scene_id"]
        mood = scene["mood"]
        duration = float(scene.get("duration", 3.0))

        # We construct a dense, effective prompt for MusicGen from the cue
        music_cue = scene.get("music_cue", {})
        style = music_cue.get("music_style", "cinematic")
        instrument = music_cue.get("primary_instrument", "piano")
        tempo = music_cue.get("tempo", 110)
        energy = music_cue.get("energy", 50)
        description = scene.get("description", "")

        # Mood-specific sonic conditioning for Meta MusicGen
        if mood == "suspense":
            prompt_text = (
                f"dark horror cinematic suspense score, eerie low sub-bass drones, "
                f"tense ticking clock, unsettling dissonant strings, ominous atmosphere, "
                f"slow creeping dread, {tempo} BPM, dark horror film soundtrack"
            )
        elif mood == "mysterious":
            prompt_text = (
                f"mysterious ambient cinematic soundtrack, glassy eerie pads, sparse dark piano, "
                f"low drones, enigmatic subtle textures, {tempo} BPM, mystery film score"
            )
        elif mood == "action":
            prompt_text = (
                f"epic action cinematic soundtrack, driving heavy drums, pulsing bass, "
                f"fast rhythmic strings ostinato, intense dynamic chase, {tempo} BPM, high energy"
            )
        elif mood == "romantic":
            prompt_text = (
                f"warm romantic cinematic soundtrack, expressive emotional piano, "
                f"legato strings, soft acoustic harp, heartwarming melody, {tempo} BPM"
            )
        elif mood == "calm":
            prompt_text = (
                f"peaceful ambient daylight cinematic music, soft warm felt piano, "
                f"gentle airy pads, quiet relaxing background, {tempo} BPM, clean mix"
            )
        else:
            prompt_text = (
                f"{style} soundtrack, {mood} mood, {instrument}, "
                f"{tempo} BPM, {energy}% intensity, {description}, high quality instrumental"
            )

        logger.info(
            "--- Synthesizing %s | Mood: %s | Style: %s | %d BPM (%.1fs) ---",
            scene_id.upper(),
            mood.upper(),
            style,
            tempo,
            duration,
        )

        if model is not None and processor is not None:
            # 50 tokens per second of audio
            max_tokens = int(max(duration, 1.5) * 50)

            inputs = processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

            # Convert tensor to numpy 1D waveform
            waveform = audio_values[0, 0].cpu().numpy().astype(np.float32)
        else:
            waveform = generate_procedural_scene_audio(scene, duration, sampling_rate)

        # Truncate or match exact segment duration
        target_samples = int(duration * sampling_rate)
        if len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        elif len(waveform) < target_samples:
            pad = np.zeros(target_samples - len(waveform), dtype=np.float32)
            waveform = np.concatenate([waveform, pad])

        # Normalize audio volume to prevent digital clipping
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = (waveform / max_val) * 0.95

        # Convert to 16-bit PCM for universal player compatibility
        pcm16 = (waveform * 32767).astype(np.int16)

        stem = Path(data.get("filepath", "")).stem if data.get("filepath") else json_path.stem
        wav_path = out_dir / f"{stem}_{scene_id}.wav"
        scipy.io.wavfile.write(str(wav_path), sampling_rate, pcm16)
        # Also maintain generic scene file for backward compatibility
        scipy.io.wavfile.write(str(out_dir / f"{scene_id}.wav"), sampling_rate, pcm16)
        logger.info("Saved scene audio: %s", wav_path)

        generated_audio_paths.append(wav_path)
        all_audio_chunks.append(pcm16)

    # Stitch all scene audio chunks into a complete continuous soundtrack with crossfades
    stem = Path(data.get("filepath", "")).stem if data.get("filepath") else json_path.stem
    full_soundtrack_path = out_dir / f"{stem}_soundtrack.wav"
    legacy_soundtrack_path = out_dir / "final_soundtrack.wav"
    if generated_audio_paths and stitch_audio_with_crossfade:
        logger.info(
            "Stitching %d scenes with %dms crossfades and %dms smooth ending fade-out...",
            len(generated_audio_paths),
            args.crossfade_ms,
            args.fade_out_ms,
        )
        try:
            stitch_audio_with_crossfade(
                scene_audio_paths=generated_audio_paths,
                output_path=full_soundtrack_path,
                crossfade_ms=args.crossfade_ms,
                ending_fade_out_ms=args.fade_out_ms,
            )
            # Copy to generic final_soundtrack.wav
            import shutil
            shutil.copyfile(full_soundtrack_path, legacy_soundtrack_path)
        except Exception as exc:
            logger.warning("PyDub crossfading fallback to simple concat: %s", exc)
            if all_audio_chunks:
                full_soundtrack = np.concatenate(all_audio_chunks)
                scipy.io.wavfile.write(str(full_soundtrack_path), sampling_rate, full_soundtrack)
                scipy.io.wavfile.write(str(legacy_soundtrack_path), sampling_rate, full_soundtrack)
    elif all_audio_chunks:
        full_soundtrack = np.concatenate(all_audio_chunks)
        scipy.io.wavfile.write(str(full_soundtrack_path), sampling_rate, full_soundtrack)
        scipy.io.wavfile.write(str(legacy_soundtrack_path), sampling_rate, full_soundtrack)
        logger.info("Saved stitched complete soundtrack: %s", full_soundtrack_path)

    # Merge audio back into original input video with 100% lossless video stream copying
    final_video_path = None
    input_video_path = Path(data.get("filepath", "")).resolve()
    if input_video_path.exists() and merge_video_and_audio:
        final_video_path = out_dir.parent / f"{input_video_path.stem}_with_ai_music.mp4"
        try:
            merge_video_and_audio(
                video_path=input_video_path,
                audio_path=full_soundtrack_path,
                output_path=final_video_path,
            )
            logger.info("Created synchronized final video: %s", final_video_path)
        except Exception as exc:
            logger.warning("Could not merge audio with video: %s", exc)

    print("\n" + "=" * 60)
    print("      META MUSICGEN AI AUDIO GENERATION COMPLETE!      ")
    print("=" * 60)
    for p in generated_audio_paths:
        print(f"  * Scene Track : {p}")
    print(f"  * Full Score  : {full_soundtrack_path}")
    if final_video_path and final_video_path.exists():
        print(f"  * FINAL VIDEO : {final_video_path}")
        print("    (Original video quality 100% preserved bit-for-bit)")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
