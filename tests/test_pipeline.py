"""End-to-end integration tests for the VideoAIPipeline.

Validates:
- Complete pipeline execution on input video
- Proper integration of VideoProcessor, SceneAnalyzer, and PromptGenerator
- Structured JSON output verification (filename, duration, segments, timestamps, moods, prompts)
- Mocking of external video decoding / AI layers to test pipeline logic deterministically
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sociodrift.pipeline import VideoAIPipeline, VideoAnalysisResult
from sociodrift.video_processor import VideoMetadata, VideoSegment


class TestVideoAIPipeline:
    """Integration tests for the single-video pipeline."""

    @pytest.fixture
    def mock_video_metadata(self):
        """Standard mock video metadata for a 9-second video."""
        return VideoMetadata(
            filepath="/tmp/mock_sample.mp4",
            filename="mock_sample.mp4",
            duration_seconds=9.0,
            fps=30.0,
            total_frames=270,
            width=1280,
            height=720,
        )

    def test_pipeline_end_to_end_mocked(self, tmp_path, mock_video_metadata):
        """Verify the full pipeline workflow from video ingestion to JSON output.

        Mocks OpenCV frame extraction to run in full deterministic isolation.
        """
        pipeline = VideoAIPipeline(segment_duration=3.0, demo_mode=True)

        with patch.object(pipeline.video_processor, "load_video", return_value=mock_video_metadata):
            dummy_video_path = tmp_path / "mock_sample.mp4"
            dummy_video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

            output_dir = tmp_path / "outputs"
            result = pipeline.process_video(dummy_video_path, output_dir=output_dir)

            # 1. Validate top-level metadata
            assert isinstance(result, VideoAnalysisResult)
            assert result.status == "SUCCESS"
            assert result.filename == "mock_sample.mp4"
            assert result.video_duration == 9.0
            assert result.total_segments == 3
            assert result.resolution == "1280x720"

            # 2. Validate scenes structure
            assert len(result.scenes) == 3
            scene_1, scene_2, scene_3 = result.scenes

            assert scene_1.start_time == 0.0
            assert scene_1.end_time == 3.0
            assert scene_2.start_time == 3.0
            assert scene_2.end_time == 6.0
            assert scene_3.start_time == 6.0
            assert scene_3.end_time == 9.0

            # 3. Validate narrative flow roles
            assert scene_1.story_role == "opening"
            assert scene_3.story_role == "ending"

            # 4. Validate scene descriptions and moods
            for scene in result.scenes:
                assert scene.mood != ""
                assert len(scene.semantic_labels) > 0
                assert scene.description != ""
                assert "metrics" in scene.to_dict()
                assert "music_cue" in scene.to_dict()
                assert "prompt" in scene.to_dict()

            # 5. Validate generated music prompt directives
            for scene in result.scenes:
                prompt = scene.prompt
                assert "Genre:" in prompt
                assert "Emotion:" in prompt
                assert "Energy:" in prompt
                assert "Tempo Target:" in prompt
                assert "Instrumentation:" in prompt
                assert "Primary Instrument:" in prompt
                assert "Transition In:" in prompt
                assert "Transition Out:" in prompt
                assert "Continuity:" in prompt
                assert "Visual Metrics:" in prompt

            # 6. Validate JSON serialization on disk
            json_file = output_dir / "mock_sample.json"
            assert json_file.exists(), f"Output JSON not found at {json_file}"

            loaded_data = json.loads(json_file.read_text())
            assert loaded_data["status"] == "SUCCESS"
            assert loaded_data["filename"] == "mock_sample.mp4"
            assert loaded_data["total_segments"] == 3
            assert len(loaded_data["scenes"]) == 3
            assert loaded_data["scenes"][0]["scene_id"] == "scene-1"

    def test_pipeline_raises_on_unreadable_video(self, tmp_path):
        """Pipeline must raise an exception when given a non-existent or unreadable file."""
        pipeline = VideoAIPipeline()
        non_existent = tmp_path / "ghost.mp4"
        with pytest.raises(FileNotFoundError):
            pipeline.process_video(non_existent)
