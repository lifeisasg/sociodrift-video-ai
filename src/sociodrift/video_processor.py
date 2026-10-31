"""Video processing and segmentation module.

Handles video file validation, metadata extraction, duration calculation,
and segment partitioning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSegment:
    """Represents a continuous time slice within a video."""

    segment_id: str
    index: int
    start_time: float
    end_time: float
    duration: float

    def to_dict(self) -> dict[str, float | str | int]:
        """Convert segment to dictionary representation."""
        return {
            "segment_id": self.segment_id,
            "index": self.index,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "duration": round(self.duration, 2),
        }


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from a video container and stream."""

    filepath: str
    filename: str
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int

    def to_dict(self) -> dict[str, float | str | int]:
        """Convert metadata to dictionary representation."""
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "duration_seconds": round(self.duration_seconds, 2),
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
        }


class VideoProcessor:
    """Extracts metadata, extracts frames, and divides videos into time segments."""

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

    def __init__(self) -> None:
        self._cv2 = None

    def _ensure_cv2(self):
        """Lazy load OpenCV to maintain fast imports."""
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError as err:
                raise ImportError(
                    "OpenCV is required for video processing. "
                    "Install it via `pip install opencv-python-headless`."
                ) from err
        return self._cv2

    def load_video(self, video_path: str | Path) -> VideoMetadata:
        """Validate and extract metadata from a video file.

        Args:
            video_path: Path to the target video file.

        Returns:
            VideoMetadata instance with duration, fps, frame count, dimensions.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError: If the file format is unsupported, empty, or unreadable.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Target path is not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension '{path.suffix}'. "
                f"Supported extensions: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {path.name}")

        cv2 = self._ensure_cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file (corrupted or unreadable codec): {path.name}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            # Some containers return 0 for FPS or frame count; compute or fallback
            if fps <= 0.0:
                fps = 24.0

            if total_frames > 0:
                duration = total_frames / fps
            else:
                # Count frames manually if container metadata is incomplete
                frames_counted = 0
                while True:
                    success, _ = capture.read()
                    if not success:
                        break
                    frames_counted += 1
                total_frames = frames_counted
                duration = (frames_counted / fps) if fps > 0 else 0.0

            if duration <= 0.0 or total_frames == 0:
                raise ValueError(f"Video file contains no readable video frames: {path.name}")

            return VideoMetadata(
                filepath=str(path),
                filename=path.name,
                duration_seconds=round(duration, 3),
                fps=round(fps, 2),
                total_frames=total_frames,
                width=width,
                height=height,
            )
        finally:
            capture.release()

    @staticmethod
    def calculate_segments(
        duration: float, segment_duration: float
    ) -> list[tuple[float, float]]:
        """Calculate continuous non-overlapping segment intervals.

        Divides total duration into windows of length `segment_duration`.
        The final segment captures any remaining partial interval cleanly.

        Args:
            duration: Total video duration in seconds (must be > 0).
            segment_duration: Desired segment length in seconds (must be > 0).

        Returns:
            List of (start_time, end_time) tuples rounded to 2 decimal places.

        Raises:
            ValueError: If duration or segment_duration are invalid (<= 0).
        """
        if duration <= 0:
            raise ValueError(
                f"Video duration must be strictly positive (> 0), got: {duration}"
            )
        if segment_duration <= 0:
            raise ValueError(
                f"Segment duration must be strictly positive (> 0), got: {segment_duration}"
            )

        segments: list[tuple[float, float]] = []
        current_start = 0.0

        while current_start < duration:
            current_end = min(duration, current_start + segment_duration)
            # Ensure float rounding precision
            start_rounded = round(current_start, 2)
            end_rounded = round(current_end, 2)
            if end_rounded > start_rounded:
                segments.append((start_rounded, end_rounded))
            current_start += segment_duration

        return segments

    def divide_into_segments(
        self, video_metadata: VideoMetadata, segment_duration: float = 3.0
    ) -> list[VideoSegment]:
        """Divide a video into structured VideoSegment instances.

        Args:
            video_metadata: VideoMetadata instance.
            segment_duration: Desired segment duration in seconds.

        Returns:
            List of VideoSegment instances.
        """
        intervals = self.calculate_segments(
            video_metadata.duration_seconds, segment_duration
        )
        segments: list[VideoSegment] = []

        for index, (start, end) in enumerate(intervals, start=1):
            seg_len = round(end - start, 2)
            segments.append(
                VideoSegment(
                    segment_id=f"scene-{index}",
                    index=index - 1,
                    start_time=start,
                    end_time=end,
                    duration=seg_len,
                )
            )

        logger.debug(
            "Divided %s (%.2fs) into %d segments of ~%.1fs",
            video_metadata.filename,
            video_metadata.duration_seconds,
            len(segments),
            segment_duration,
        )
        return segments

    def extract_segment_frames(
        self,
        video_path: str | Path,
        segment: VideoSegment,
        sample_count: int = 6,
    ) -> list["np.ndarray"]:
        """Extract representative frames from a video segment.

        Args:
            video_path: Path to the video file.
            segment: The VideoSegment defining the time window.
            sample_count: Number of equidistant frames to sample.

        Returns:
            List of sampled frames as numpy arrays (BGR format).
        """
        cv2 = self._ensure_cv2()
        import numpy as np

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.warning("Could not open video %s for frame extraction", video_path)
            return []

        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
            start_sec = segment.start_time
            end_sec = max(segment.start_time, segment.end_time - 0.05)

            actual_samples = max(2, min(sample_count, int(segment.duration * 4)))
            timestamps = np.linspace(start_sec, end_sec, actual_samples)
            frames: list[np.ndarray] = []

            for ts in timestamps:
                frame_no = int(max(0, ts * fps))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                success, frame = capture.read()
                if success and frame is not None:
                    frames.append(frame)

            return frames
        finally:
            capture.release()
