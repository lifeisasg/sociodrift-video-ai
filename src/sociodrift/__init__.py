"""SocioDrift Video AI Pipeline.

A standalone, reproducible video AI module for automated video segmentation,
scene analysis, and cinematic music prompt generation.
"""

from sociodrift.batch_processor import BatchProcessor, BatchResult
from sociodrift.pipeline import VideoAIPipeline, VideoAnalysisResult
from sociodrift.prompt_generator import MusicDirector, PromptGenerator
from sociodrift.scene_analyzer import SceneAnalyzer, SceneMetrics
from sociodrift.video_processor import VideoMetadata, VideoProcessor, VideoSegment

__version__ = "0.1.0"

__all__ = [
    "BatchProcessor",
    "BatchResult",
    "MusicDirector",
    "PromptGenerator",
    "SceneAnalyzer",
    "SceneMetrics",
    "VideoAIPipeline",
    "VideoAnalysisResult",
    "VideoMetadata",
    "VideoProcessor",
    "VideoSegment",
]
