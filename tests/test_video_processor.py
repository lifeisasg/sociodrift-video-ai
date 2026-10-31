"""Unit tests for the VideoProcessor module.

Covers:
- Happy path: video segmentation with valid duration (10s divided by 3s -> 4 segments)
- Non-overlapping intervals and exact final partial segment capture
- Edge cases: zero segment duration, negative segment duration, invalid duration
- Segment duration greater than video duration
- Metadata extraction and error handling for missing/empty files
"""

import pytest

from sociodrift.video_processor import VideoMetadata, VideoProcessor, VideoSegment


class TestVideoProcessorSegmentation:
    """Tests for segment interval calculation logic."""

    def test_segmentation_happy_path(self):
        """Happy Path: Duration = 10.0s, Segment Length = 3.0s.

        Expected segments:
        - Segment 1: (0.0, 3.0)
        - Segment 2: (3.0, 6.0)
        - Segment 3: (6.0, 9.0)
        - Segment 4: (9.0, 10.0) -> partial segment capturing remaining 1.0s
        """
        duration = 10.0
        segment_duration = 3.0

        intervals = VideoProcessor.calculate_segments(duration, segment_duration)

        # 1. Validate correct number of segments
        assert len(intervals) == 4, f"Expected 4 segments, got {len(intervals)}"

        # 2. Validate exact timestamps
        expected_intervals = [
            (0.0, 3.0),
            (3.0, 6.0),
            (6.0, 9.0),
            (9.0, 10.0),
        ]
        assert intervals == expected_intervals, f"Intervals mismatch: {intervals}"

        # 3. Validate no overlapping segments and strict continuity
        for i in range(len(intervals) - 1):
            current_end = intervals[i][1]
            next_start = intervals[i + 1][0]
            assert current_end == next_start, (
                f"Discontinuity or overlap between segment {i} ({intervals[i]}) "
                f"and segment {i+1} ({intervals[i+1]})"
            )

        # 4. Validate start and end boundaries
        assert intervals[0][0] == 0.0, "First segment must start at 0.0s"
        assert intervals[-1][1] == duration, f"Final segment must end exactly at {duration}s"

        # 5. Validate final partial segment duration
        final_start, final_end = intervals[-1]
        final_duration = round(final_end - final_start, 2)
        assert final_duration == 1.0, f"Expected partial segment duration 1.0s, got {final_duration}s"

    def test_segmentation_exact_multiple(self):
        """Video duration exactly divisible by segment duration (12s / 4s = 3 segments)."""
        intervals = VideoProcessor.calculate_segments(duration=12.0, segment_duration=4.0)
        assert len(intervals) == 3
        assert intervals == [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)]

    def test_segmentation_shorter_than_segment_length(self):
        """Video duration shorter than segment duration (e.g. 2.5s duration, 5.0s segment)."""
        intervals = VideoProcessor.calculate_segments(duration=2.5, segment_duration=5.0)
        assert len(intervals) == 1
        assert intervals == [(0.0, 2.5)]

    def test_divide_into_segments_dataclasses(self):
        """Verify divide_into_segments returns well-formed VideoSegment dataclasses."""
        processor = VideoProcessor()
        meta = VideoMetadata(
            filepath="/mock/test.mp4",
            filename="test.mp4",
            duration_seconds=10.0,
            fps=30.0,
            total_frames=300,
            width=1920,
            height=1080,
        )
        segments = processor.divide_into_segments(meta, segment_duration=3.0)
        assert len(segments) == 4
        assert segments[0].segment_id == "scene-1"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 3.0
        assert segments[0].duration == 3.0
        assert segments[-1].segment_id == "scene-4"
        assert segments[-1].start_time == 9.0
        assert segments[-1].end_time == 10.0
        assert segments[-1].duration == 1.0


class TestVideoProcessorEdgeCases:
    """Tests for edge cases and input validation failures."""

    def test_zero_segment_duration_raises_value_error(self):
        """Zero segment duration must raise ValueError."""
        with pytest.raises(ValueError, match="Segment duration must be strictly positive"):
            VideoProcessor.calculate_segments(duration=10.0, segment_duration=0.0)

    def test_negative_segment_duration_raises_value_error(self):
        """Negative segment duration must raise ValueError."""
        with pytest.raises(ValueError, match="Segment duration must be strictly positive"):
            VideoProcessor.calculate_segments(duration=10.0, segment_duration=-3.0)

    def test_zero_video_duration_raises_value_error(self):
        """Zero video duration must raise ValueError."""
        with pytest.raises(ValueError, match="Video duration must be strictly positive"):
            VideoProcessor.calculate_segments(duration=0.0, segment_duration=3.0)

    def test_negative_video_duration_raises_value_error(self):
        """Negative video duration must raise ValueError."""
        with pytest.raises(ValueError, match="Video duration must be strictly positive"):
            VideoProcessor.calculate_segments(duration=-10.0, segment_duration=3.0)

    def test_missing_video_file_raises_file_not_found(self):
        """Non-existent video file path must raise FileNotFoundError."""
        processor = VideoProcessor()
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            processor.load_video("non_existent_file.mp4")

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        """Unsupported video extension must raise ValueError."""
        fake_file = tmp_path / "document.pdf"
        fake_file.write_text("not a video")
        processor = VideoProcessor()
        with pytest.raises(ValueError, match="Unsupported video extension"):
            processor.load_video(fake_file)

    def test_empty_video_file_raises_value_error(self, tmp_path):
        """Zero-byte video file must raise ValueError."""
        empty_file = tmp_path / "empty.mp4"
        empty_file.touch()
        processor = VideoProcessor()
        with pytest.raises(ValueError, match="Video file is empty"):
            processor.load_video(empty_file)
