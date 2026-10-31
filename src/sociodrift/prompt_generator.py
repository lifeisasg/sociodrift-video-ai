"""Music prompt generation and musical score direction module.

Converts visual features, scene context, narrative roles, and emotion profiles
into structured conditioning prompts for music generation models.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sociodrift.scene_analyzer import SceneAnalysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MusicCue:
    """Musical direction and arrangement parameters for a scene segment."""

    music_style: str
    emotion: str
    energy: int
    tempo: int
    primary_instrument: str
    instrumentation: str
    transition_in: str
    transition_out: str
    continuity: str
    intensity_curve: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EMOTION_STYLE_MAP: dict[str, tuple[str, str, str]] = {
    "calm": ("ambient cinematic", "felt piano", "felt piano, warm pads, subtle low strings"),
    "peaceful": ("organic ambient", "acoustic piano", "acoustic piano, soft strings, breathy pads"),
    "happy": ("uplifting cinematic pop", "guitar", "clean guitar, light percussion, bright strings"),
    "travel": ("travel documentary", "guitar", "muted guitar, marimba pulses, warm strings"),
    "dramatic": ("dramatic cinematic", "strings", "low strings, piano pulses, restrained brass"),
    "epic": ("epic orchestral", "strings", "large strings, taiko-style percussion, brass swells"),
    "romantic": ("romantic cinematic", "piano", "piano, legato strings, soft harp accents"),
    "emotional": ("emotional cinematic", "piano", "expressive piano, warm strings, soft ambience"),
    "mysterious": ("mysterious ambient", "cinematic textures", "low drones, glassy pads, sparse piano"),
    "suspense": ("suspense score", "cinematic pulses", "low drones, ticking pulses, muted percussion"),
    "action": ("action cinematic", "percussion", "driving drums, pulsing bass, sharp string ostinato"),
    "energetic": ("energetic cinematic pop", "synth", "pulsing synth, bright percussion, rhythmic strings"),
}

ROLE_ENERGY_OFFSETS: dict[str, int] = {
    "opening": -10,
    "build-up": 5,
    "transition": 0,
    "climax": 20,
    "resolution": -12,
    "ending": -18,
    "complete_story": 0,
}

ROLE_CURVES: dict[str, str] = {
    "opening": "fade in gently, introduce a memorable motif without crowding the video",
    "build-up": "gradually add rhythm and harmonic motion",
    "transition": "keep the motif alive while changing texture smoothly",
    "climax": "reach the strongest musical statement with fuller arrangement",
    "resolution": "release tension and reduce percussion",
    "ending": "settle into a clean emotional cadence",
    "complete_story": "self-contained arc with a clear beginning, gentle rise, and soft ending",
}


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


class MusicDirector:
    """Plans musical direction and cue attributes based on scene analysis."""

    def plan_cues(self, analyses: list[SceneAnalysis]) -> list[MusicCue]:
        """Generate musical cues for each scene analysis in temporal sequence."""
        cues: list[MusicCue] = []
        previous_emotion = ""
        previous_style = ""
        count = len(analyses)

        for index, scene in enumerate(analyses):
            emotion = scene.mood
            role = scene.story_role
            base_style, primary_inst, instrumentation = EMOTION_STYLE_MAP.get(
                emotion, EMOTION_STYLE_MAP["emotional"]
            )

            metrics = scene.metrics
            motion_energy = min(42.0, float(metrics.camera_motion_intensity) * 0.9)
            emotion_energy = float(scene.emotion.scores.get(emotion, 0.5)) * 45.0
            progression_energy = (index / max(1, count - 1)) * 12.0
            offset = ROLE_ENERGY_OFFSETS.get(role, 0)

            energy = _clamp_int(
                22.0 + motion_energy + emotion_energy + progression_energy + offset, 12, 96
            )
            tempo = _clamp_int(
                72.0 + (energy * 0.72) + (float(metrics.optical_flow) * 9.0), 62, 156
            )

            transition_in = self._compute_transition(previous_emotion, emotion, entering=True)
            next_emotion = analyses[index + 1].mood if index + 1 < count else ""
            transition_out = self._compute_transition(emotion, next_emotion, entering=False)

            continuity = "establish the main motif"
            if previous_style:
                continuity = f"preserve the previous {previous_style} motif while evolving into {base_style}"
            if role == "climax":
                continuity = "bring earlier motifs forward with the strongest harmony and rhythm"
            elif role in {"resolution", "ending"}:
                continuity = "resolve earlier motifs with fewer layers and warmer harmony"

            cues.append(
                MusicCue(
                    music_style=base_style,
                    emotion=emotion,
                    energy=energy,
                    tempo=tempo,
                    primary_instrument=primary_inst,
                    instrumentation=instrumentation,
                    transition_in=transition_in,
                    transition_out=transition_out,
                    continuity=continuity,
                    intensity_curve=ROLE_CURVES.get(role, ROLE_CURVES["transition"]),
                )
            )
            previous_emotion = emotion
            previous_style = base_style

        return cues

    def _compute_transition(self, previous: str, current: str, entering: bool) -> str:
        """Determine smooth audio transition instructions between scenes."""
        if not previous or not current:
            return "natural fade in" if entering else "prepare a clean tail for the next scene"

        high_motion = {"action", "energetic", "epic", "suspense"}
        soft = {"calm", "peaceful", "romantic", "emotional"}

        if previous == current:
            return "continue the shared motif with a subtle texture change"
        if previous in soft and current in high_motion:
            return "use a rising swell and percussion lift"
        if previous in high_motion and current in soft:
            return "release energy with a soft orchestral fall and warm reverb tail"
        return "crossfade with a short ambient bridge"


class PromptGenerator:
    """Builds structured text prompts ready for AI music generators."""

    def __init__(self) -> None:
        self.director = MusicDirector()

    def generate_prompts(
        self, analyses: list[SceneAnalysis]
    ) -> list[tuple[MusicCue, str]]:
        """Generate (MusicCue, prompt_text) pairs for each scene analysis."""
        cues = self.director.plan_cues(analyses)
        results: list[tuple[MusicCue, str]] = []

        for index, (scene, cue) in enumerate(zip(analyses, cues), start=1):
            labels_str = ", ".join(scene.semantic_labels)
            metrics = scene.metrics

            prev_ctx = "none; this is the opening cue"
            if index > 1:
                prev_cue = cues[index - 2]
                prev_ctx = f"previous scene was {prev_cue.emotion} {prev_cue.music_style} at {prev_cue.energy}% energy"

            next_ctx = "none; resolve the soundtrack naturally"
            if index < len(cues):
                next_c = cues[index]
                next_ctx = f"next scene becomes {next_c.emotion} {next_c.music_style} at {next_c.energy}% energy"

            prompt_lines = [
                f"Scene {index} custom cinematic background score",
                f"Genre: {cue.music_style}",
                f"Emotion: {cue.emotion}",
                f"Story Role: {scene.story_role}",
                f"Energy: {cue.energy}%",
                f"Tempo Target: {cue.tempo} BPM",
                f"Instrumentation: {cue.instrumentation}",
                f"Primary Instrument: {cue.primary_instrument}",
                f"Musical Progression: {cue.intensity_curve}",
                f"Transition In: {cue.transition_in}",
                f"Transition Out: {cue.transition_out}",
                f"Continuity: {cue.continuity}",
                f"Previous Context: {prev_ctx}",
                f"Next Context: {next_ctx}",
                f"Visual Labels: {labels_str}",
                f"Visual Metrics: brightness {metrics.average_brightness}, saturation {metrics.average_saturation}, motion {metrics.camera_motion_intensity}, stability {metrics.scene_stability}",
                "Mix Notes: instrumental only, no vocals, no abrupt restart, clean background mix, leave room for video dialogue and ambient sound",
            ]
            prompt_text = "\n".join(prompt_lines)
            results.append((cue, prompt_text))

        return results
