"""Integration tests for the BatchProcessor extension.

Validates:
- Discovery of supported video files
- Processing multiple inputs in one execution run
- Generation of individual JSON files for each video
- Failure isolation: corrupted/failing inputs do not halt batch execution
- Accurate batch summary statistics (total, success, failed) and summary JSON
- Empty directory and non-video directory handling
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sociodrift.batch_processor import BatchProcessor, BatchResult
from sociodrift.pipeline import VideoAIPipeline, VideoAnalysisResult
from sociodrift.video_processor import VideoMetadata


class TestBatchProcessor:
    """Tests for multi-file batch processing and fault tolerance."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline that succeeds for normal files and fails on 'corrupt.mp4'."""
        pipeline = MagicMock(spec=VideoAIPipeline)

        def side_effect(video_path, output_dir=None):
            path_str = str(video_path)
            if "corrupt" in path_str:
                raise ValueError("Corrupted video container: invalid header")

            filename = video_path.name
            result = VideoAnalysisResult(
                filename=filename,
                filepath=path_str,
                video_duration=6.0,
                total_frames=180,
                fps=30.0,
                resolution="1920x1080",
                segment_duration_target=3.0,
                total_segments=2,
                status="SUCCESS",
                scenes=[],
            )
            if output_dir:
                result.save_json(output_dir / f"{video_path.stem}.json")
            return result

        pipeline.process_video.side_effect = side_effect
        return pipeline

    def test_batch_processing_with_mixed_inputs(self, tmp_path, mock_pipeline):
        """Verify batch processor processes multiple inputs and isolates failures.

        Setup:
        - video1.mp4 (valid)
        - video2.mp4 (valid)
        - corrupt.mp4 (failing)
        - ignored.txt (unsupported extension - should be skipped)

        Expectations:
        - 3 video files discovered
        - 2 successfully processed
        - 1 recorded as failure
        - Output JSONs created for video1 and video2
        - Batch summary JSON contains accurate metrics
        """
        input_dir = tmp_path / "videos"
        input_dir.mkdir()
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create dummy test files
        (input_dir / "video1.mp4").write_bytes(b"dummy mp4")
        (input_dir / "video2.mov").write_bytes(b"dummy mov")
        (input_dir / "corrupt.mp4").write_bytes(b"bad bytes")
        (input_dir / "ignored.txt").write_text("not a video")

        batch_processor = BatchProcessor(pipeline=mock_pipeline)
        result = batch_processor.process_directory(input_dir=input_dir, output_dir=output_dir)

        # 1. Verify counts
        assert isinstance(result, BatchResult)
        assert result.total_files_discovered == 3
        assert result.successful_count == 2
        assert result.failed_count == 1
        assert result.total_processed == 3

        # 2. Verify individual outputs
        assert (output_dir / "video1.json").exists()
        assert (output_dir / "video2.json").exists()
        assert not (output_dir / "corrupt.json").exists()

        # 3. Verify failure was isolated and recorded
        failed_records = [r for r in result.results if r.status == "FAILED"]
        assert len(failed_records) == 1
        assert failed_records[0].filename == "corrupt.mp4"
        assert "Corrupted video container" in failed_records[0].error_message

        # 4. Verify batch summary JSON
        summary_path = output_dir / "batch_summary.json"
        assert summary_path.exists()
        summary_data = json.loads(summary_path.read_text())
        assert summary_data["summary"]["total_files_discovered"] == 3
        assert summary_data["summary"]["successful"] == 2
        assert summary_data["summary"]["failed"] == 1

        # 5. Verify console summary formatting
        console_text = result.format_console_summary()
        assert "BATCH PROCESSING SUMMARY" in console_text
        assert "video1.mp4" in console_text
        assert "SUCCESS" in console_text
        assert "corrupt.mp4" in console_text
        assert "FAILED" in console_text

    def test_batch_empty_directory(self, tmp_path):
        """Empty input directory should yield zero processed files cleanly."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        output_dir = tmp_path / "outputs"

        processor = BatchProcessor()
        result = processor.process_directory(empty_dir, output_dir)

        assert result.total_files_discovered == 0
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert (output_dir / "batch_summary.json").exists()

    def test_batch_missing_directory_raises_error(self, tmp_path):
        """Missing input directory should raise FileNotFoundError."""
        missing = tmp_path / "non_existent_folder"
        processor = BatchProcessor()
        with pytest.raises(FileNotFoundError):
            processor.discover_videos(missing)
