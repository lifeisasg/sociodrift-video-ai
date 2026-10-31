# SocioDrift Video AI – Automated Video Segmentation & AI Music Prompting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](#15-automated-testing-instructions)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.84-orange.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A clean, reproducible, production-grade video AI pipeline packaged as an independent slice of the **SocioDrift** platform. It ingests video files, computes visual and temporal features, performs computer vision scene understanding and emotion profiling, generates structured cinematic music-generation conditioning prompts, and serializes production-ready JSON artifacts.

---

## 1. Project Overview

**SocioDrift Video AI** is the core computer-vision and generative-audio intelligence engine extracted from the larger SocioDrift multi-modal creation suite. Designed to run completely standalone and in isolated CI/CD environments, it provides deterministic temporal segmentation, optical flow tracking, color palette extraction, narrative role classification, and prompt orchestration for generative audio models (such as Meta's MusicGen, Stable Audio, Suno, and Udio).

---

## 2. Background: SocioDrift Video AI Prototype

The full SocioDrift platform is an AI music co-pilot for short-form video creators featuring:
- React + Vite + Tailwind CSS interactive editing studio.
- FastAPI backend serving reference BPM audio analysis and MLX-accelerated MusicGen generation.
- Real-time video scene detection using OpenCV frame differencing and optical flow.
- Automated score direction and multi-segment audio crossfading.

This repository packages the **video AI intelligence slice** into a standalone, modular, and dependency-pinned Python package that can be evaluated and executed in isolation without requiring web servers, frontend dependencies, or GPU audio synthesis weights.

---

## 3. Problem Statement

Video creators often struggle to find or compose soundtrack music that matches the emotional pacing, dynamic movement, and narrative progression of their video clips. Manual music editing is time-consuming and requires music theory expertise.

Existing text-to-music AI models (e.g., MusicGen, Stable Audio) accept descriptive text conditioning, but creators lack an automated bridge that converts raw video pixels and motion into rich, context-aware musical prompts.

**SocioDrift Video AI solves this** by translating visual features (brightness, saturation, motion intensity, optical flow, visual complexity) into musical parameters (genre, tempo BPM, energy level, instrumentation, story role, scene transitions, and continuity instructions) exported as structured JSON.

---

## 4. Features

- **Robust Video Ingestion & Validation**: Fast metadata extraction (resolution, framerate, frame count, duration) with strict error handling for non-existent, empty, or corrupt video containers.
- **Configurable Temporal Segmentation**: Mathematically clean interval partitioning that guarantees continuous, non-overlapping windows and captures partial trailing intervals accurately.
- **Dense Optical Flow & Motion Analysis**: Uses Gunner Farneback's algorithm to calculate pixel velocity vectors, classify camera movement (locked-off, gentle pan, active, action), and compute scene stability.
- **Dominant Color Extraction**: K-Means clustering in 3D color space to extract dominant visual color palettes (hex format).
- **Semantic Scene Understanding**: Heuristic visual rule engine classifying scenes across 12 semantic tags (`outdoor`, `indoor`, `nature`, `sports`, `action`, `travel`, `night`, `landscape`, `city`, `sunset`, `people`, `architecture`).
- **12-Dimensional Emotion Profiling**: Multi-axis mood classifier computing score distributions for `calm`, `peaceful`, `happy`, `travel`, `dramatic`, `epic`, `romantic`, `emotional`, `mysterious`, `suspense`, `action`, and `energetic`.
- **Story Flow Detection**: Assigns narrative arc roles (`opening`, `build-up`, `transition`, `climax`, `resolution`, `ending`) across chronological segments.
- **Music Cue Planning & Conditioning Prompts**: Determines musical style, target tempo (BPM), energy percentage, primary instrument, instrumentation layers, and transition guidance.
- **Fault-Tolerant Batch Processing**: Discovers video files across directories, executes independent pipelines, saves per-video JSON outputs, isolates corrupted files without aborting the batch, and exports a unified `batch_summary.json`.
- **Deterministic Offline Demo Mode**: Runs cleanly even without video codecs or external hardware using the `--demo` flag.

---

## 5. Architecture Overview

```text
                                 [INPUT VIDEO]
                                       |
                                       v
                             +-------------------+
                             |  VideoProcessor   |
                             +-------------------+
                                       |
                         +-------------+-------------+
                         |                           |
                         v                           v
                 [Video Metadata]            [Temporal Segments]
               (duration, fps, res)          (scene-1, scene-2...)
                                                     |
                                                     v
                                            +------------------+
                                            |  SceneAnalyzer   |
                                            +------------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                           |                           |
                         v                           v                           v
                 [Frame Metrics]             [Semantic Labels]           [Emotion Profile]
             (brightness, saturation,      (outdoor, sunset, city)       (primary mood, 12D
             optical flow, stability)                                   confidence scores)
                         \                           |                           /
                          +--------------------------+--------------------------+
                                                     |
                                                     v
                                            +------------------+
                                            |  PromptGenerator |
                                            |  (MusicDirector) |
                                            +------------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v                                                       v
                   [Music Cues]                                          [Conditioning Text]
             (genre, tempo, energy,                                    (structured multi-line
             instruments, transitions)                                    prompt for AI)
                         \                                                       /
                          +--------------------------+--------------------------+
                                                     |
                                                     v
                                            +------------------+
                                            |     Pipeline     |
                                            +------------------+
                                                     |
                                                     v
                                         [STRUCTURED JSON OUTPUT]
                                        (outputs/<video_stem>.json)
```

---

## 6. Processing Pipeline Explanation

1. **Ingestion & Validation**: `VideoProcessor.load_video()` opens the video stream with OpenCV (`cv2.VideoCapture`), validates the container format, frame count, FPS, and returns a `VideoMetadata` record.
2. **Segmentation**: `VideoProcessor.calculate_segments()` takes total duration and desired segment length (default: 3.0s), generating non-overlapping time windows `(start, end)`.
3. **Sampling**: `VideoProcessor.extract_segment_frames()` samples equidistant BGR frames across each segment window.
4. **Computer Vision Extraction**:
   - `cv2.absdiff` & grayscale mean for average luminance.
   - HSV color space transformation for saturation channel mean.
   - `cv2.Canny` edge detector for visual spatial complexity.
   - `cv2.calcOpticalFlowFarneback` calculates displacement vectors between adjacent frames.
   - `cv2.kmeans` partitions pixel colors to identify primary palettes.
5. **Context & Emotion Classification**: Computes weighted scores across 12 moods and semantic tags based on brightness, saturation, motion intensity, and edge complexity.
6. **Narrative Sequencing**: `SceneAnalyzer.assign_story_roles()` evaluates energy trajectory to identify climax, opening, build-up, and resolution phases.
7. **Score Direction**: `MusicDirector.plan_cues()` calculates target tempo (62–156 BPM), energy curve (12–96%), primary instrument, instrumentation, and inter-scene transition logic (e.g. crossfade, swell, fade in).
8. **Prompt Generation**: `PromptGenerator` formats a conditioning prompt containing musical parameters, continuity instructions, previous/next scene context, and visual metrics.
9. **JSON Serialization**: `VideoAnalysisResult.save_json()` formats and saves the entire manifest to disk.

---

## 7. Technology Stack

- **Language**: Python 3.9+ (tested on Python 3.9.6, 3.11, 3.12)
- **Computer Vision**: `opencv-python-headless==4.10.0.84`
- **Numerical Computing**: `numpy==1.26.4`
- **Testing Framework**: `pytest==8.3.3`
- **Environment Management**: `python-dotenv==1.0.1`
- **Packaging**: `pyproject.toml` (PEP 517 / 621) + `setuptools`

---

## 8. Repository Structure

```text
sociodrift-video-ai/
│
├── src/
│   └── sociodrift/
│       ├── __init__.py           # Package exports & versioning
│       ├── video_processor.py    # Video loading, metadata, segmentation, frame sampling
│       ├── scene_analyzer.py     # Optical flow, metrics, semantic labels, emotion profiling
│       ├── prompt_generator.py   # MusicDirector cue planning & prompt generation
│       ├── pipeline.py           # Single-video orchestration & JSON serialization
│       └── batch_processor.py    # Multi-video discovery, fault isolation, summary reporting
│
├── tests/
│   ├── test_video_processor.py   # Happy path segmentation & edge case tests
│   ├── test_pipeline.py          # End-to-end single video integration tests
│   └── test_batch_processor.py   # Batch processing, failure isolation, summary tests
│
├── samples/
│   ├── sample_video.mp4          # Synthetic 9.0s 3-scene test video
│   ├── sample_video_2.mp4        # Synthetic 6.0s 2-scene test video
│   ├── corrupt_video.mp4         # Synthetic corrupted container test fixture
│   └── generate_sample_videos.py # Deterministic sample video generator
│
├── outputs/                      # Generated JSON outputs and batch summaries
│   ├── sample_video.json
│   ├── sample_video_2.json
│   └── batch_summary.json
│
├── main.py                       # CLI entry point (single & batch execution)
├── requirements.txt              # Direct pinned dependencies
├── pyproject.toml                # Project packaging & pytest configuration
├── .env.example                  # Environment variable template
├── .gitignore                    # Git exclusions
├── TECHNICAL_WALKTHROUGH.md      # In-depth technical walkthrough & interview prep
└── README.md                     # Comprehensive documentation (this file)
```

---

## 9. Prerequisites

- **Python**: Version 3.9 or higher (`python3 --version`).
- **Operating System**: macOS (Apple Silicon or Intel), Linux (Ubuntu, Debian, CentOS), or Windows.
- **Git**: For version control.
- *Note*: No FFmpeg binary or GPU hardware is required for this standalone video AI module; `opencv-python-headless` runs entirely on standard CPU.

---

## 10. Installation Instructions

1. Clone or navigate into the repository:
   ```bash
   cd sociodrift-video-ai
   ```

2. Create a clean Python virtual environment:
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows**:
     ```powershell
     .venv\Scripts\activate
     ```

4. Upgrade `pip` and install pinned dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Installs core Video AI dependencies and pytest in ~5 seconds. To run the optional Web Studio & Meta MusicGen, also run `pip install -r requirements-studio.txt`)*

5. (Optional) Generate the sample video fixtures:
   ```bash
   python samples/generate_sample_videos.py
   ```

---

## 11. Environment Variable Configuration

Copy the example configuration:
```bash
cp .env.example .env
```

Available environment variables:
| Variable | Default | Description |
|---|---|---|
| `SOCIODRIFT_SEGMENT_DURATION` | `3.0` | Default duration in seconds per scene segment |
| `SOCIODRIFT_OUTPUT_DIR` | `outputs` | Default output folder for JSON manifests |
| `SOCIODRIFT_DEMO_MODE` | `0` | Set to `1` for deterministic offline demo mode |
| `SOCIODRIFT_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 12. Single-Video Execution Instructions

Run the end-to-end pipeline on any supported video file (`.mp4`, `.mov`, `.m4v`, `.avi`):

```bash
python main.py --input samples/sample_video.mp4
```

### Additional Single-Video Options:
- **Custom Segment Length (e.g. 4.5 seconds)**:
  ```bash
  python main.py --input samples/sample_video.mp4 --segment-duration 4.5
  ```
- **Custom Output Directory**:
  ```bash
  python main.py --input samples/sample_video.mp4 --output-dir my_custom_outputs/
  ```
- **Verbose Debug Logging**:
  ```bash
  python main.py --input samples/sample_video.mp4 --verbose
  ```
- **Deterministic Offline Demo Mode**:
  ```bash
  python main.py --input samples/sample_video.mp4 --demo
  ```

---

## 13. Batch Execution Instructions

Process an entire directory containing multiple video files in a single command:

```bash
python main.py --input-dir samples/
```

### Batch Behavior:
1. Automatically discovers all supported video files in the specified directory.
2. Processes each video sequentially through the full video AI pipeline.
3. Saves an individual JSON result for every successful video: `outputs/<filename_stem>.json`.
4. **Isolates Failures**: If one video file is corrupted or unreadable, the pipeline logs the failure, records it in the batch summary, and continues processing the remaining files.
5. Saves a consolidated summary to `outputs/batch_summary.json`.
6. Prints a clean status table to the terminal.

---

## 14. Output JSON Explanation

Each processed video outputs a detailed JSON file structured as follows:

```json
{
  "status": "SUCCESS",
  "filename": "sample_video.mp4",
  "filepath": "/path/to/samples/sample_video.mp4",
  "video_duration": 9.0,
  "total_frames": 216,
  "fps": 24.0,
  "resolution": "640x360",
  "segment_duration_target": 3.0,
  "total_segments": 3,
  "scenes": [
    {
      "scene_id": "scene-1",
      "index": 0,
      "start": 0.0,
      "end": 3.0,
      "duration": 3.0,
      "mood": "calm",
      "story_role": "opening",
      "semantic_labels": [
        "outdoor",
        "nature",
        "travel",
        "sunset",
        "landscape"
      ],
      "description": "Calm scene (outdoor, nature, travel, sunset, landscape) with locked-off.",
      "metrics": {
        "duration": 3.0,
        "average_brightness": 160.81,
        "average_saturation": 137.64,
        "dominant_colors": [
          "#57a9ce",
          "#6cbde2",
          "#6dbee3",
          "#58a9ce"
        ],
        "optical_flow": 0.103,
        "camera_motion_intensity": 1.85,
        "visual_complexity": 0.0025,
        "estimated_camera_movement": "locked-off",
        "scene_stability": 0.992
      },
      "music_cue": {
        "music_style": "ambient cinematic",
        "emotion": "calm",
        "energy": 57,
        "tempo": 114,
        "primary_instrument": "felt piano",
        "instrumentation": "felt piano, warm pads, subtle low strings",
        "transition_in": "natural fade in",
        "transition_out": "crossfade with a short ambient bridge",
        "continuity": "establish the main motif",
        "intensity_curve": "fade in gently, introduce a memorable motif without crowding the video"
      },
      "prompt": "Scene 1 custom cinematic background score\nGenre: ambient cinematic\nEmotion: calm\nStory Role: opening\nEnergy: 57%\nTempo Target: 114 BPM\nInstrumentation: felt piano, warm pads, subtle low strings\nPrimary Instrument: felt piano\nMusical Progression: fade in gently, introduce a memorable motif without crowding the video\nTransition In: natural fade in\nTransition Out: crossfade with a short ambient bridge\nContinuity: establish the main motif\nPrevious Context: none; this is the opening cue\nNext Context: next scene becomes romantic romantic cinematic at 96% energy\nVisual Labels: outdoor, nature, travel, sunset, landscape\nVisual Metrics: brightness 160.81, saturation 137.64, motion 1.85, stability 0.992\nMix Notes: instrumental only, no vocals, no abrupt restart, clean background mix, leave room for video dialogue and ambient sound"
    }
  ],
  "error": null
}
```

---

## 15. Automated Testing Instructions

The repository includes a comprehensive test suite written for **Pytest**.

### Run All Tests:
```bash
pytest
```

### Run With Verbose Output:
```bash
pytest -v
```

### Test Coverage Highlights:
- **Happy Path (`test_video_processor.py`)**: Tests partitioning a 10.0s video with 3.0s segments into 4 intervals `[(0, 3), (3, 6), (6, 9), (9, 10)]`, confirming correct timestamp boundaries, non-overlapping intervals, and exact final partial duration (1.0s).
- **Edge Cases (`test_video_processor.py`)**: Tests zero duration, negative duration, zero segment length, negative segment length, missing video files, unsupported extensions (`.pdf`), and 0-byte empty files.
- **Pipeline Integration (`test_pipeline.py`)**: End-to-end integration test validating metadata, story roles, prompt directives, and disk JSON serialization.
- **Batch Processing (`test_batch_processor.py`)**: Verifies multi-file processing, individual output creation, empty directory handling, and **failure isolation** where corrupted files do not abort other inputs.

---

## 16. Sample Execution Output

### Single-Video Terminal Output:
```text
$ python main.py --input samples/sample_video.mp4
22:46:07 [INFO] sociodrift.pipeline: Starting Video AI pipeline for: sample_video.mp4
22:46:07 [INFO] sociodrift.pipeline: Loaded sample_video.mp4 (9.00s, 640x360, 24.0 FPS)
22:46:07 [INFO] sociodrift.pipeline: Generated 3 temporal segments
22:46:08 [INFO] sociodrift.pipeline: Saved pipeline results to: outputs/sample_video.json

============================================================
           SOCIODRIFT VIDEO AI - EXECUTION SUMMARY           
============================================================
File            : sample_video.mp4
Status          : SUCCESS
Duration        : 9.00 seconds
Resolution      : 640x360
Framerate       : 24.0 FPS (Total frames: 216)
Target Segments : 3 (Interval: 3.0s)
------------------------------------------------------------
Detected Scenes & Musical Prompts:

  [SCENE-1] (0.0s -> 3.0s)
    Mood / Role   : CALM | Story Role: opening
    Music Style   : ambient cinematic @ 114 BPM (Energy: 57%)
    Instrument    : felt piano (felt piano, warm pads, subtle low strings)
    Labels        : outdoor, nature, travel, sunset, landscape
    Prompt Preview: Scene 1 custom cinematic background score | Genre: ambient cinematic | Emotion: calm...

  [SCENE-2] (3.0s -> 6.0s)
    Mood / Role   : ROMANTIC | Story Role: climax
    Music Style   : romantic cinematic @ 148 BPM (Energy: 96%)
    Instrument    : piano (piano, legato strings, soft harp accents)
    Labels        : outdoor, travel, nature, sunset, landscape
    Prompt Preview: Scene 2 custom cinematic background score | Genre: romantic cinematic | Emotion: romantic...

  [SCENE-3] (6.0s -> 9.0s)
    Mood / Role   : CALM | Story Role: ending
    Music Style   : ambient cinematic @ 115 BPM (Energy: 56%)
    Instrument    : felt piano (felt piano, warm pads, subtle low strings)
    Labels        : indoor, night, architecture, sunset
    Prompt Preview: Scene 3 custom cinematic background score | Genre: ambient cinematic | Emotion: calm...
============================================================

Output saved to: outputs/sample_video.json
```

### Batch Terminal Output:
```text
$ python main.py --input-dir samples/
22:46:36 [INFO] sociodrift.batch_processor: Discovered 3 video files in samples
22:46:36 [INFO] sociodrift.batch_processor: --- Batch Processing: corrupt_video.mp4 ---
22:46:36 [ERROR] sociodrift.batch_processor: Failed processing corrupt_video.mp4: Failed to open video file: corrupt_video.mp4
22:46:36 [INFO] sociodrift.batch_processor: --- Batch Processing: sample_video.mp4 ---
22:46:36 [INFO] sociodrift.pipeline: Saved pipeline results to: outputs/sample_video.json
22:46:36 [INFO] sociodrift.batch_processor: --- Batch Processing: sample_video_2.mp4 ---
22:46:36 [INFO] sociodrift.pipeline: Saved pipeline results to: outputs/sample_video_2.json
22:46:36 [INFO] sociodrift.batch_processor: Saved batch summary to: outputs/batch_summary.json

========================================
       BATCH PROCESSING SUMMARY       
========================================
Total videos discovered : 3
Successfully processed  : 2
Failed                  : 1
Skipped                 : 0
----------------------------------------
File Status Details:
  corrupt_video.mp4         [FAILED] - Reason: Failed to open video file (corrupted or unreadable codec): corrupt_video.mp4
  sample_video.mp4          [SUCCESS] (3 segments)
  sample_video_2.mp4        [SUCCESS] (2 segments)
========================================
Batch summary JSON saved to: outputs/batch_summary.json
```

---

## 17. Error-Handling Behaviour

| Condition | Raised Error | Handled Outcome |
|---|---|---|
| File does not exist | `FileNotFoundError` | CLI prints clear error message, exits with code 1 |
| Unsupported extension (`.pdf`, `.txt`) | `ValueError` | Rejected before opening container; skipped in batch discovery |
| 0-byte empty file | `ValueError` | Explicit check catches empty files before decoding |
| Corrupt container / missing moov atom | `ValueError` | Caught by `VideoProcessor.load_video()`; isolated in batch mode |
| Invalid segment duration (`<= 0`) | `ValueError` | Caught immediately during segment calculation |
| Invalid video duration (`<= 0`) | `ValueError` | Caught with descriptive exception message |
| Frame sampling failure | Fallback / Error | `SceneAnalyzer` logs warning and applies safe defaults |

---

## 18. Known Limitations

1. **Fixed Interval vs Adaptive Cut Segmentation**: The CLI divides videos into configurable fixed-length temporal segments (e.g. 3.0s). The full SocioDrift platform also supports adaptive scene cut detection based on SSIM and histogram spikes.
2. **CPU-Bound Optical Flow**: Gunner Farneback dense optical flow runs on CPU via OpenCV. On 4K 60FPS videos, downscaling frames (internally downscaled to 320x180) ensures high speed (~0.05s per frame), but GPU acceleration via CUDA/Metal can be integrated for ultra-high-throughput enterprise pipelines.

---

## 19. Interactive Web Studio & Neural Music Synthesis

In addition to the core CLI, this repository includes an interactive full-stack Web Studio and real Meta MusicGen neural audio synthesis:

### Launch Interactive Web Studio:
```bash
python web_app.py
```
Open **`http://localhost:8000`** in any browser.

**Features:**
- **Official 3D SocioDrift Logo**: Integrated in the header with a dark theme aesthetic.
- **Single Video Studio**: Drag & drop video, inspect scene metrics, synthesize AI music, preview synchronized video, and download final video (100% lossless video stream copying).
- **Multi-Video Batch Queue**: Select multiple clips, monitor sequential processing in real time with progress bars, and download audio/video as each item completes.
- **Refresh State Persistence**: Analysis and batch results persist in `localStorage` across browser reloads until reset.

### Standalone Neural Audio Synthesis (CLI):
```bash
python generate_music.py --json outputs/sample_video.json --merge-video
```
Synthesizes `.wav` waveforms with Meta MusicGen, applies PyDub 500ms equal-power crossfading, -16 dBFS loudness normalization, 1000ms ending fade-out, and multiplexes into `outputs/<stem>_with_ai_music.mp4` with bit-for-bit video quality preservation.
