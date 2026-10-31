"""End-to-end video AI pipeline module.

Coordinates video ingestion, temporal segmentation, computer vision scene analysis,
and AI music prompt generation, saving structured JSON artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sociodrift.prompt_generator import MusicCue, PromptGenerator
from sociodrift.scene_analyzer import SceneAnalysis, SceneAnalyzer
from sociodrift.video_processor import VideoMetadata, VideoProcessor, VideoSegment

logger = logging.getLogger(__name__)


@dataclass
class ProcessedScene:
    """Consolidated representation of a processed scene segment."""

    scene_id: str
    index: int
    start_time: float
    end_time: float
    duration: float
    mood: str
    story_role: str
    semantic_labels: list[str]
    description: str
    metrics: dict[str, Any]
    music_cue: dict[str, Any]
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "index": self.index,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "mood": self.mood,
            "story_role": self.story_role,
            "semantic_labels": self.semantic_labels,
            "description": self.description,
            "metrics": self.metrics,
            "music_cue": self.music_cue,
            "prompt": self.prompt,
        }


@dataclass
class VideoAnalysisResult:
    """Structured output schema for a complete video processing run."""

    filename: str
    filepath: str
    video_duration: float
    total_frames: int
    fps: float
    resolution: str
    segment_duration_target: float
    total_segments: int
    status: str
    scenes: list[ProcessedScene]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "filename": self.filename,
            "filepath": self.filepath,
            "video_duration": round(self.video_duration, 2),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "resolution": self.resolution,
            "segment_duration_target": self.segment_duration_target,
            "total_segments": self.total_segments,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "error": self.error,
        }

    def save_json(self, output_path: str | Path) -> Path:
        """Serialize analysis results to a formatted JSON file."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved pipeline results to: %s", path)
        return path


class VideoAIPipeline:
    """Orchestrates video processing, vision analysis, and prompt generation."""

    def __init__(
        self,
        segment_duration: float = 3.0,
        demo_mode: bool = False,
    ) -> None:
        self.segment_duration = segment_duration
        self.demo_mode = demo_mode
        self.video_processor = VideoProcessor()
        self.scene_analyzer = SceneAnalyzer(demo_mode=demo_mode)
        self.prompt_generator = PromptGenerator()

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> VideoAnalysisResult:
        """Run the end-to-end pipeline on a single input video file.

        Args:
            video_path: Path to the target video.
            output_dir: Optional directory to save output JSON.

        Returns:
            VideoAnalysisResult instance.

        Raises:
            FileNotFoundError: If input file is missing.
            ValueError: If input video is corrupt, empty, or unreadable.
        """
        path = Path(video_path).resolve()
        logger.info("Starting Video AI pipeline for: %s", path.name)

        # 1. Video Loading and Metadata Extraction
        metadata = self.video_processor.load_video(path)
        logger.info(
            "Loaded %s (%.2fs, %dx%d, %.1f FPS)",
            metadata.filename,
            metadata.duration_seconds,
            metadata.width,
            metadata.height,
            metadata.fps,
        )

        # 2. Video Segmentation
        segments = self.video_processor.divide_into_segments(
            metadata, segment_duration=self.segment_duration
        )
        logger.info("Generated %d temporal segments", len(segments))

        # 3. Scene Frame Extraction & Analysis
        raw_analyses: list[SceneAnalysis] = []
        for seg in segments:
            frames = (
                []
                if self.demo_mode
                else self.video_processor.extract_segment_frames(path, seg)
            )
            analysis = self.scene_analyzer.analyze_frames(
                frames=frames,
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
            )
            raw_analyses.append(analysis)

        # 4. Narrative Role Assignment
        roles = self.scene_analyzer.assign_story_roles(raw_analyses)
        analyzed_scenes: list[SceneAnalysis] = []
        for analysis, role in zip(raw_analyses, roles):
            # Reconstruct with assigned story role
            analyzed_scenes.append(
                SceneAnalysis(
                    segment_id=analysis.segment_id,
                    start_time=analysis.start_time,
                    end_time=analysis.end_time,
                    duration=analysis.duration,
                    metrics=analysis.metrics,
                    labels=analysis.labels,
                    semantic_labels=analysis.semantic_labels,
                    emotion=analysis.emotion,
                    mood=analysis.mood,
                    story_role=role,
                )
            )

        # 5. Music Prompt Generation
        cues_and_prompts = self.prompt_generator.generate_prompts(analyzed_scenes)

        # 6. Assemble Processed Scenes
        processed_scenes: list[ProcessedScene] = []
        for idx, (analysis, (cue, prompt_text)) in enumerate(
            zip(analyzed_scenes, cues_and_prompts)
        ):
            description = (
                f"{analysis.mood.capitalize()} scene ({', '.join(analysis.semantic_labels)}) "
                f"with {analysis.metrics.estimated_camera_movement}."
            )
            processed_scenes.append(
                ProcessedScene(
                    scene_id=analysis.segment_id,
                    index=idx,
                    start_time=analysis.start_time,
                    end_time=analysis.end_time,
                    duration=analysis.duration,
                    mood=analysis.mood,
                    story_role=analysis.story_role,
                    semantic_labels=analysis.semantic_labels,
                    description=description,
                    metrics=analysis.metrics.to_dict(),
                    music_cue=cue.to_dict(),
                    prompt=prompt_text,
                )
            )

        result = VideoAnalysisResult(
            filename=metadata.filename,
            filepath=str(path),
            video_duration=metadata.duration_seconds,
            total_frames=metadata.total_frames,
            fps=metadata.fps,
            resolution=f"{metadata.width}x{metadata.height}",
            segment_duration_target=self.segment_duration,
            total_segments=len(processed_scenes),
            status="SUCCESS",
            scenes=processed_scenes,
        )

        if output_dir:
            out_path = Path(output_dir) / f"{path.stem}.json"
            result.save_json(out_path)

        return result
