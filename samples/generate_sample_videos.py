"""Generates synthetic, deterministic sample MP4 videos for testing and demonstration.

Creates:
1. sample_video.mp4 (9 seconds, 24 fps):
   - Scene 1 (0-3s): Bright, calm landscape (blue sky, soft clouds, calm gradient)
   - Scene 2 (3-6s): Warm, high-motion sunset with rotating golden element (dynamic motion)
   - Scene 3 (6-9s): Dark, high-contrast, suspenseful nocturnal scene with flickering bars
2. sample_video_2.mp4 (6 seconds, 24 fps):
   - Scene 1 (0-3s): Forest nature green scene
   - Scene 2 (3-6s): Action / sports vibrant scene
3. corrupt_video.mp4 (0.5KB corrupt header file for failure isolation testing)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def generate_videos(output_dir: str | Path | None = None) -> None:
    """Generate deterministic sample video files."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV or NumPy not installed. Install requirements first.", file=sys.stderr)
        return

    dest = Path(output_dir or Path(__file__).resolve().parent)
    dest.mkdir(parents=True, exist_ok=True)

    width, height = 640, 360
    fps = 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 1. Generate sample_video.mp4 (9.0 seconds = 216 frames)
    sample_1_path = dest / "sample_video.mp4"
    out1 = cv2.VideoWriter(str(sample_1_path), fourcc, fps, (width, height))
    total_frames_1 = int(9.0 * fps)

    for i in range(total_frames_1):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if t < 3.0:
            # Scene 1: Calm blue daytime gradient
            # Brightness ~150-180, low motion
            frame[:, :, 0] = np.uint8(np.linspace(200, 240, height)[:, None])  # Blue
            frame[:, :, 1] = np.uint8(np.linspace(160, 200, height)[:, None])  # Green
            frame[:, :, 2] = np.uint8(np.linspace(80, 120, height)[:, None])   # Red
            # Subtle floating circle
            cx = int(width / 2 + math.sin(t * 0.5) * 20)
            cy = int(height / 2 + math.cos(t * 0.5) * 10)
            cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)

        elif t < 6.0:
            # Scene 2: Warm energetic sunset orange with active camera motion
            frame[:, :, 0] = np.uint8(30)                                       # Blue
            frame[:, :, 1] = np.uint8(np.linspace(100, 180, width)[None, :])   # Green
            frame[:, :, 2] = np.uint8(np.linspace(200, 255, height)[:, None])  # Red
            # Moving sun element
            sweep = (t - 3.0) / 3.0
            cx = int(width * 0.2 + sweep * (width * 0.6))
            cy = int(height * 0.7 - math.sin(sweep * math.pi) * 120)
            cv2.circle(frame, (cx, cy), 50, (0, 215, 255), -1)

        else:
            # Scene 3: Dark suspenseful nocturnal scene with rapid motion
            frame[:, :, :] = np.uint8(15)  # Very dark base
            # Rapid scanning beam
            beam_x = int((i * 18) % width)
            cv2.rectangle(frame, (beam_x, 0), (beam_x + 30, height), (90, 80, 120), -1)
            # Center mysterious pulsing symbol
            pulse = int(25 + 15 * math.sin(t * 8.0))
            cv2.circle(frame, (width // 2, height // 2), pulse, (160, 140, 220), 2)

        out1.write(frame)

    out1.release()
    print(f"Generated: {sample_1_path} ({sample_1_path.stat().st_size} bytes, 9.0s)")

    # 2. Generate sample_video_2.mp4 (6.0 seconds = 144 frames)
    sample_2_path = dest / "sample_video_2.mp4"
    out2 = cv2.VideoWriter(str(sample_2_path), fourcc, fps, (width, height))
    total_frames_2 = int(6.0 * fps)

    for i in range(total_frames_2):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if t < 3.0:
            # Green peaceful nature landscape
            frame[:, :, 0] = 40
            frame[:, :, 1] = 160
            frame[:, :, 2] = 60
            # Horizon line
            cv2.line(frame, (0, height // 2), (width, height // 2), (30, 120, 40), 4)
        else:
            # Dynamic magenta/violet action scene
            frame[:, :, 0] = 160
            frame[:, :, 1] = 40
            frame[:, :, 2] = 180
            # Moving action polygon
            shift = int((i * 12) % width)
            pts = np.array([[shift, 100], [shift + 60, 200], [shift - 30, 260]], np.int32)
            cv2.fillPoly(frame, [pts], (220, 200, 255))

        out2.write(frame)

    out2.release()
    print(f"Generated: {sample_2_path} ({sample_2_path.stat().st_size} bytes, 6.0s)")

    # 3. Create corrupt_video.mp4 for failure isolation tests
    corrupt_path = dest / "corrupt_video.mp4"
    corrupt_path.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00corrupted-junk-payload")
    print(f"Generated: {corrupt_path} (Corrupt test fixture)")


if __name__ == "__main__":
    generate_videos()
