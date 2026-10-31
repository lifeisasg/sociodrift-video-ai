"""Scene analysis module using computer vision and semantic scene understanding.

Extracts optical flow, motion intensity, color palettes, visual complexity,
semantic labels, emotion profiles, and story flow roles directly from video frames.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticLabel:
    """Semantic scene descriptor with confidence score."""

    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 3)}


@dataclass(frozen=True)
class EmotionProfile:
    """Dominant mood and full emotion distribution across 12 dimensions."""

    primary: str
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


@dataclass(frozen=True)
class SceneMetrics:
    """Visual metrics extracted from frames within a scene segment."""

    duration: float
    average_brightness: float
    average_saturation: float
    dominant_colors: list[str]
    optical_flow: float
    camera_motion_intensity: float
    visual_complexity: float
    estimated_camera_movement: str
    scene_stability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneAnalysis:
    """Complete analysis result for a single video segment."""

    segment_id: str
    start_time: float
    end_time: float
    duration: float
    metrics: SceneMetrics
    labels: list[SemanticLabel]
    semantic_labels: list[str]
    emotion: EmotionProfile
    mood: str
    story_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "mood": self.mood,
            "emotion": self.emotion.to_dict(),
            "story_role": self.story_role,
            "semantic_labels": self.semantic_labels,
            "labels": [lbl.to_dict() for lbl in self.labels],
            "metrics": self.metrics.to_dict(),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a floating point value to [low, high]."""
    return max(low, min(high, float(value)))


def _classify_camera_movement(flow: float, stability: float) -> str:
    """Classify camera dynamic behavior from optical flow magnitude and stability."""
    if flow < 0.7 and stability > 0.75:
        return "locked-off"
    if flow < 1.8:
        return "gentle pan or handheld drift"
    if flow < 4.0:
        return "active camera movement"
    return "rapid movement or action camera"


class SceneAnalyzer:
    """Performs frame analysis, semantic labeling, emotion profiling, and story flow."""

    def __init__(self, demo_mode: bool = False) -> None:
        self.demo_mode = demo_mode
        self._cv2 = None
        self._np = None

    def _ensure_libraries(self):
        if self._cv2 is None or self._np is None:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
        return self._cv2, self._np

    def analyze_frames(
        self,
        frames: list["np.ndarray"],
        segment_id: str,
        start_time: float,
        end_time: float,
        story_role: str = "scene",
    ) -> SceneAnalysis:
        """Analyze a list of frames from a segment.

        Args:
            frames: List of BGR frames extracted by VideoProcessor.
            segment_id: Identifier for the segment.
            start_time: Start timestamp in seconds.
            end_time: End timestamp in seconds.
            story_role: Optional narrative role tag.

        Returns:
            SceneAnalysis with complete visual metrics, emotion profile, and labels.
        """
        duration = max(0.1, round(end_time - start_time, 2))

        if self.demo_mode or not frames:
            # Deterministic fallback when running without OpenCV or on empty frames
            metrics = self._generate_default_metrics(duration)
            labels = [SemanticLabel("cinematic", 0.85), SemanticLabel("storytelling", 0.75)]
            emotion = EmotionProfile(
                primary="calm",
                scores={"calm": 0.82, "peaceful": 0.74, "emotional": 0.52},
            )
            return SceneAnalysis(
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                metrics=metrics,
                labels=labels,
                semantic_labels=[lbl.label for lbl in labels],
                emotion=emotion,
                mood=emotion.primary,
                story_role=story_role,
            )

        cv2, np = self._ensure_libraries()

        brightness_values: list[float] = []
        saturation_values: list[float] = []
        complexity_values: list[float] = []
        flow_values: list[float] = []
        color_samples: list[np.ndarray] = []
        previous_gray: np.ndarray | None = None

        for frame in frames:
            resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 80, 160)

            brightness_values.append(float(np.mean(gray)))
            saturation_values.append(float(np.mean(hsv[:, :, 1])))
            complexity_values.append(float(np.mean(edges > 0)))
            color_samples.extend(self._extract_dominant_colors(resized))

            if previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                magnitude, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
                flow_values.append(float(np.mean(magnitude)))

            previous_gray = gray

        flow_mean = float(np.mean(flow_values)) if flow_values else 0.5
        flow_std = float(np.std(flow_values)) if flow_values else 0.1
        bright_std = float(np.std(brightness_values)) if brightness_values else 5.0
        stability = _clamp(1.0 - (flow_std / 4.0) - (bright_std / 90.0))

        avg_brightness = float(np.mean(brightness_values)) if brightness_values else 100.0
        avg_saturation = float(np.mean(saturation_values)) if saturation_values else 100.0
        avg_complexity = float(np.mean(complexity_values)) if complexity_values else 0.1

        metrics = SceneMetrics(
            duration=duration,
            average_brightness=round(avg_brightness, 2),
            average_saturation=round(avg_saturation, 2),
            dominant_colors=self._build_color_palette(color_samples),
            optical_flow=round(flow_mean, 3),
            camera_motion_intensity=round(flow_mean * 18.0, 2),
            visual_complexity=round(avg_complexity, 4),
            estimated_camera_movement=_classify_camera_movement(flow_mean, stability),
            scene_stability=round(stability, 3),
        )

        labels = self._extract_semantic_labels(metrics)
        emotion = self._compute_emotion_profile(metrics, labels)

        return SceneAnalysis(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            metrics=metrics,
            labels=labels,
            semantic_labels=[lbl.label for lbl in labels],
            emotion=emotion,
            mood=emotion.primary,
            story_role=story_role,
        )

    def _extract_dominant_colors(self, frame: "np.ndarray") -> list["np.ndarray"]:
        """Extract dominant color centroids using K-means."""
        cv2, np = self._ensure_libraries()
        small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        pixels = small.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        try:
            _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=len(centers))
            ordered = centers[np.argsort(counts)[::-1]]
            return [color for color in ordered[:2]]
        except Exception as exc:
            logger.debug("K-means clustering fallback: %s", exc)
            return [np.array([128, 128, 128], dtype=np.float32)]

    def _build_color_palette(self, colors: list["np.ndarray"]) -> list[str]:
        """Convert sampled colors into hex color strings."""
        if not colors:
            return ["#2a2a2a", "#808080"]

        cv2, np = self._ensure_libraries()
        pixels = np.array(colors, dtype=np.float32)
        cluster_count = min(4, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        try:
            _, labels, centers = cv2.kmeans(
                pixels, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            counts = np.bincount(labels.flatten(), minlength=len(centers))
            hex_colors: list[str] = []
            for bgr in centers[np.argsort(counts)[::-1]]:
                b, g, r = [int(_clamp(ch, 0, 255)) for ch in bgr]
                hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            return hex_colors
        except Exception:
            return ["#1a1a1a", "#4a4a4a"]

    def _extract_semantic_labels(self, metrics: SceneMetrics) -> list[SemanticLabel]:
        """Infer semantic context labels using visual feature rules."""
        b = metrics.average_brightness
        s = metrics.average_saturation
        f = metrics.optical_flow
        c = metrics.visual_complexity
        scores: dict[str, float] = {}

        def record(label: str, score: float) -> None:
            scores[label] = max(scores.get(label, 0.0), _clamp(score))

        record("night", (95 - b) / 95.0)
        record("outdoor", (b - 95) / 90.0 + s / 420.0)
        record("indoor", (135 - b) / 130.0 + (0.18 - c))
        record("landscape", (b - 100) / 100.0 + (0.12 - c))
        record("city", c * 4.5 + s / 520.0)
        record("architecture", c * 4.0 + (0.55 if metrics.scene_stability > 0.65 else 0.0))
        record("sports", f / 4.2 + c * 2.0)
        record("action", f / 3.5)
        record("travel", (b - 85) / 115.0 + f / 6.0 + s / 500.0)
        record("nature", (b - 95) / 120.0 + (0.2 - c) + s / 430.0)
        record("sunset", s / 260.0 + (150 - abs(b - 125)) / 280.0)
        record("people", c * 2.2 + f / 7.0)

        chosen = [
            SemanticLabel(label=label, confidence=round(val, 3))
            for label, val in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if val >= 0.28
        ]
        return chosen[:5] or [SemanticLabel("visual-story", 0.5)]

    def _compute_emotion_profile(
        self, metrics: SceneMetrics, labels: list[SemanticLabel]
    ) -> EmotionProfile:
        """Compute distribution across 12 mood dimensions based on visual features."""
        label_weights = {l.label: l.confidence for l in labels}
        motion = _clamp(metrics.optical_flow / 4.5)
        brightness = _clamp(metrics.average_brightness / 180.0)
        darkness = _clamp((110.0 - metrics.average_brightness) / 110.0)
        saturation = _clamp(metrics.average_saturation / 160.0)
        complexity = _clamp(metrics.visual_complexity / 0.16)
        instability = 1.0 - metrics.scene_stability

        # Calm and peaceful require ambient daylight; darkness represents tension, dread, or night
        light_presence = _clamp((metrics.average_brightness - 42.0) / 85.0)

        # Slow motion in dark imagery creates creeping dread / horror suspense
        slow_dread = (1.0 - motion) * 0.35
        darkness_boost = darkness * 0.58
        night_weight = label_weights.get("night", 0.0) * 0.25

        scores = {
            "calm": ((1 - motion) * 0.44 + metrics.scene_stability * 0.36 + (1 - complexity) * 0.20) * light_presence,
            "peaceful": ((1 - motion) * 0.34 + saturation * 0.26 + label_weights.get("nature", 0) * 0.24 + metrics.scene_stability * 0.16) * light_presence,
            "happy": (brightness * 0.34 + saturation * 0.32 + label_weights.get("travel", 0) * 0.16) * light_presence,
            "travel": label_weights.get("travel", 0) * 0.42 + brightness * 0.20 + saturation * 0.16 + motion * 0.22,
            "dramatic": darkness * 0.32 + complexity * 0.24 + instability * 0.18 + motion * 0.18 + label_weights.get("city", 0) * 0.17,
            "epic": brightness * 0.16 + saturation * 0.16 + motion * 0.26 + complexity * 0.20 + label_weights.get("landscape", 0) * 0.22,
            "romantic": (brightness * 0.20 + saturation * 0.24 + (1 - motion) * 0.24 + label_weights.get("sunset", 0) * 0.32) * max(0.2, light_presence),
            "emotional": (1 - motion) * 0.26 + darkness * 0.16 + metrics.scene_stability * 0.24 + complexity * 0.10 + label_weights.get("people", 0) * 0.24,
            "mysterious": darkness_boost * 0.85 + (1 - saturation) * 0.22 + complexity * 0.18 + metrics.scene_stability * 0.14 + night_weight,
            "suspense": darkness_boost + slow_dread + night_weight + complexity * 0.15,
            "action": motion * 0.46 + instability * 0.18 + complexity * 0.16 + label_weights.get("action", 0) * 0.20,
            "energetic": motion * 0.34 + saturation * 0.25 + brightness * 0.15 + label_weights.get("sports", 0) * 0.26,
        }

        normalized = {k: round(_clamp(v), 3) for k, v in scores.items()}
        primary = max(normalized.items(), key=lambda item: item[1])[0]
        return EmotionProfile(
            primary=primary,
            scores=dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True)),
        )

    def assign_story_roles(self, analyses: list[SceneAnalysis]) -> list[str]:
        """Assign narrative arc roles across the sequence of segments."""
        count = len(analyses)
        if count <= 1:
            return ["complete_story"]

        energies = [
            float(a.emotion.scores.get("action", 0)) + float(a.emotion.scores.get("energetic", 0))
            for a in analyses
        ]
        climax_idx = int(energies.index(max(energies))) if energies else max(0, count - 2)

        roles: list[str] = []
        for idx in range(count):
            progress = idx / max(1, count - 1)
            if idx == 0:
                roles.append("opening")
            elif idx == count - 1:
                roles.append("ending")
            elif idx == climax_idx:
                roles.append("climax")
            elif progress < 0.45:
                roles.append("build-up")
            elif progress < 0.78:
                roles.append("transition")
            else:
                roles.append("resolution")
        return roles

    @staticmethod
    def _generate_default_metrics(duration: float) -> SceneMetrics:
        """Fallback synthetic metrics for offline demo mode."""
        return SceneMetrics(
            duration=round(duration, 2),
            average_brightness=112.5,
            average_saturation=98.0,
            dominant_colors=["#2a3b4c", "#d1a256"],
            optical_flow=0.92,
            camera_motion_intensity=16.5,
            visual_complexity=0.082,
            estimated_camera_movement="gentle pan or handheld drift",
            scene_stability=0.88,
        )
