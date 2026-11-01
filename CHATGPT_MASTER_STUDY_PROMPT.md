# SocioDrift Video AI: Master Study Prompt for ChatGPT

> **Instructions for the User:**
> Copy everything below the horizontal divider line (`---`) and paste it directly into ChatGPT (GPT-4o or Claude 3.5 Sonnet). It is structured as an interactive masterclass and mock interview simulator covering every algorithm, design decision, test, library, and line of code from A to Z.

---

```text
Act as an elite Senior Staff AI/Computer Vision Engineer and Lead Technical Interviewer at a top-tier tech company (Google/Meta). 

I have built a production-grade Video AI system called "SocioDrift Video AI" as a technical assignment. I want you to act as my private technical mentor, tutor, and interviewer. Your goal is to teach me, test me, and drill me on every single concept, mathematical decision, library choice, edge case, and test in this project until I can defend every architectural choice and line of code like a senior engineer.

Here is the complete A-to-Z technical specification of my project:

================================================================================
1. PROJECT MISSION & RECRUITER ASSIGNMENT CONTEXT
================================================================================
- Context: A technical take-home assignment building an independent, reproducible slice of the SocioDrift Video AI prototype.
- Core Requirements:
  1/ Clean, reproducible repo with README setup steps and pinned dependencies.
  2/ End-to-end pipeline running on a sample video input.
  3/ Automated tests for the main transform covering the happy path + edge cases.
  4/ Batch mode processing multiple videos in one run with fault isolation.
  5/ Bonus/Extension: Full-stack interactive Web Studio + Meta MusicGen neural audio synthesis + lossless FFmpeg multiplexing.

================================================================================
2. UNIDIRECTIONAL PIPELINE ARCHITECTURE (FILTER-AND-PIPE PATTERN)
================================================================================
The codebase follows a clean unidirectional data-flow pipeline without circular dependencies:

  [Input Video (.mp4/.mov)]
            │
            ▼
  1. VideoProcessor.load_video() (OpenCV cv2.VideoCapture)
     - Validates file existence, format, non-empty (>0 bytes), readable container.
     - Extracts VideoMetadata: duration (s), fps, total_frames, resolution (width x height).
            │
            ▼
  2. VideoProcessor.calculate_segments() (Temporal Windowing)
     - Mathematical interval calculation: generates non-overlapping windows [(0, 3), (3, 6), (6, 9)...].
     - Handles partial trailing remainders (e.g., a 10s video with 3s segments gives 4 segments: [0-3], [3-6], [6-9], [9-10]).
            │
            ▼
  3. VideoProcessor.extract_segment_frames()
     - Decodes equidistant frames per segment using cv2.CAP_PROP_POS_FRAMES.
     - Downscales frames internally to 320x180 via cv2.INTER_AREA to accelerate CV algorithms by >90% without losing macro visual metrics.
            │
            ▼
  4. SceneAnalyzer.analyze_segment() (Computer Vision Engine)
     - Luminance / Brightness: Grayscale conversion (cv2.cvtColor BGR2GRAY) and global pixel mean.
     - Saturation: HSV conversion (cv2.cvtColor BGR2HSV) and mean of the S-channel.
     - Visual Complexity: Canny edge detection (cv2.Canny, thresholds 100/200) calculating edge pixel density.
     - Optical Flow & Camera Motion: Dense Gunnar Farneback optical flow (cv2.calcOpticalFlowFarneback) on consecutive frames. Calculates velocity vector magnitude and angle (cv2.cartToPolar), classifying camera motion:
       * magnitude < 1.0: "locked-off static tripod"
       * magnitude 1.0 - 3.5: "gentle pan / handheld drift"
       * magnitude 3.5 - 7.0: "active camera / moving subject"
       * magnitude > 7.0: "action / fast dynamic whip"
     - Dominant Color Palette: Reshapes frame to Nx3 pixel array, runs K-Means clustering (cv2.kmeans, k=3, criteria EPS+MAX_ITER), and formats RGB centers into hex strings (e.g. #1A2B3C).
     - Camera Stability: Computed as the reciprocal of optical flow variance.
            │
            ▼
  5. SceneAnalyzer: Semantic Labeling & 12D Emotion Profiling
     - Heuristic classification rules for 12 visual tags: outdoor, indoor, nature, sports, action, travel, night, landscape, city, sunset, people, architecture.
     - 12-Dimensional Emotion Classifier: Scores calm, peaceful, happy, travel, dramatic, epic, romantic, emotional, mysterious, suspense, action, energetic based on weighted combinations of brightness, saturation, motion velocity, and edge density.
     - Story Flow Sequencing: Evaluates energy curve across segments to assign narrative arc roles: 'opening', 'build-up', 'climax', 'resolution', 'ending'.
            │
            ▼
  6. PromptGenerator & MusicDirector (Audio Prompt Conditioning Engine)
     - Translates visual metrics into structured musical cues:
       * Genre/Style: mapped from dominant mood (e.g. ambient cinematic, dark horror suspense, epic orchestral action).
       * Target Tempo (BPM): dynamically scaled between 60 BPM and 160 BPM based on optical flow velocity and emotion.
       * Energy Level: 10% to 100% computed from motion variance and edge complexity.
       * Primary & Layered Instrumentation: (e.g., felt piano, sub-bass drones, dissonant strings, muted guitars, heavy brass).
       * Transition Directives: 'crossfade', 'swell', 'hit', 'fade-in'.
     - Generates dense conditioning prompt strings formatted specifically for generative text-to-music models (Meta MusicGen, Stable Audio).
            │
            ▼
  7. VideoAnalysisResult.save_json()
     - Formats and serializes all metadata, scene breakdown, and musical cues into production-ready JSON artifacts (`outputs/<stem>.json`).
            │
            ▼
  8. BatchProcessor (Extension)
     - Scans directories for all video files.
     - Wraps single pipeline runs in strict try/except isolation: corrupted or unreadable videos are logged and marked as FAILED without aborting the batch run.
     - Generates a consolidated `batch_summary.json` with execution counts and reasons.
            │
            ▼
  9. generate_music.py (Neural MusicGen Synthesis & Crossfading)
     - Loads Meta's official `facebook/musicgen-small` transformer model using Hugging Face Transformers (`MusicgenProcessor`, `MusicgenForConditionalGeneration`).
     - Supports Apple Silicon MPS GPU acceleration with an automatic fallback to CPU if Metal compiler is sandboxed.
     - Synthesizes distinct .wav waveforms conditioned on the dynamic prompt of each scene.
     - Applies PyDub equal-power 500ms crossfading between scenes, -16 dBFS loudness normalization, and a 1000ms ending fade-out.
            │
            ▼
 10. VideoAudioMerger (Lossless Stream Copying)
     - Calls FFmpeg via subprocess:
       `ffmpeg -y -i <video> -i <audio> -c:v copy -c:a aac -b:a 192k -shortest <output.mp4>`
     - Uses `-c:v copy` for 100% bit-for-bit lossless video stream copy (zero re-encoding delay, zero visual degradation).
            │
            ▼
 11. web_app.py (FastAPI Full-Stack Interactive Studio)
     - Interactive dark-mode web studio featuring SocioDrift's official 3D branding.
     - Endpoints:
       * POST `/api/process-video`: single video analysis & scene breakdown.
       * POST `/api/generate-music`: triggers neural music generation & video muxing.
       * POST `/api/batch-process-video`: sequential batch queue execution with live progress bars.
     - Client-side state persistence using `localStorage` across page reloads.

================================================================================
3. COMPLETE TEST SUITE (16 AUTOMATED PYTEST TESTS IN 0.02s)
================================================================================
Test suite in `tests/` covers both happy paths and edge cases:

File 1: `tests/test_video_processor.py`
  - test_calculate_segments_exact_division: Happy path (9s video / 3s = exactly 3 segments: [0-3], [3-6], [6-9]).
  - test_calculate_segments_with_remainder: Edge case (10s video / 3s = 4 segments with a 1s partial tail [9-10]).
  - test_calculate_segments_shorter_than_duration: Edge case (2s video / 3s = 1 segment [0-2]).
  - test_calculate_segments_zero_duration: Edge case (duration=0 raises ValueError).
  - test_calculate_segments_negative_duration: Edge case (duration=-5 raises ValueError).
  - test_calculate_segments_invalid_segment_length: Edge case (segment_duration=0 or -1 raises ValueError).
  - test_load_video_non_existent_file: Edge case (missing path raises FileNotFoundError).
  - test_load_video_corrupt_file: Edge case (`samples/corrupt_video.mp4` missing moov atom raises ValueError).
  - test_load_video_empty_zero_byte_file: Edge case (0-byte file raises ValueError before container read).
  - test_load_video_happy_path: Happy path (reads 9s sample video, asserts correct fps, frame count, width, height).
  - test_extract_segment_frames_bounds: Happy path (asserts valid numpy shape [180, 320, 3]).

File 2: `tests/test_pipeline.py`
  - test_pipeline_end_to_end_single_video: Integration test (runs full pipeline on sample_video.mp4, validates return result).
  - test_pipeline_json_schema_validity: Schema validation (checks all keys: filepath, video_metadata, scenes, music_cues, prompt, summary).

File 3: `tests/test_batch_processor.py`
  - test_batch_processor_multiple_valid_videos: Happy path (processes directory of videos, checks summary list).
  - test_batch_processor_fault_isolation_with_corrupt_file: Critical edge case (directory contains valid videos AND corrupt video; asserts corrupt video fails cleanly while valid videos succeed without crashing).
  - test_batch_processor_empty_directory: Edge case (empty directory returns 0 processed, 0 failed without error).
  - test_batch_processor_summary_json_output: Validates batch_summary.json structure on disk.

================================================================================
4. DEPENDENCIES & LIBRARIES USED (AND WHY CHOSEN)
================================================================================
- `opencv-python-headless==4.10.0.84`: Video container demuxing, frame seeking, Canny edge detection, optical flow, K-means. Headless variant chosen to eliminate X11/Qt display dependencies in server/cloud/CI environments.
- `numpy==1.26.4`: Fast vectorized multidimensional array math for pixel operations, optical flow magnitude/phase polar conversion, and 16-bit PCM waveform normalization.
- `pytest==8.3.3`: Automated test execution, parameterized edge cases, fast discovery (0.02s execution).
- `python-dotenv==1.0.1`: Environment variable management (`.env`).
- `fastapi==0.128.8` & `uvicorn==0.39.0`: High-performance asynchronous REST API server for the interactive Web Studio.
- `pydub==0.25.1`: Audio manipulation, 500ms equal-power crossfading, loudness matching.
- `imageio-ffmpeg==0.6.0`: Self-contained FFmpeg binary distribution to guarantee multiplexing works out-of-the-box without requiring host-level FFmpeg installation.
- `torch>=2.2.0` & `transformers>=4.40.0`: Meta MusicGen neural audio synthesis conditioned on text embeddings.
- `scipy>=1.11.0`: High-fidelity 16-bit PCM .wav file serialization.

================================================================================
5. KEY ARCHITECTURAL TRADEOFFS & INTERVIEW INSIGHTS
================================================================================
1. Fixed-Interval Temporal Windows vs. Adaptive Shot Boundary Detection (SSIM/Histogram):
   - Decision: Fixed 3.0s windowing was chosen as the baseline for predictable musical bar phrasing (e.g. 4 bars at 120 BPM = 2 seconds, 8 bars = 4 seconds). Adaptive cut detection (via histogram difference) is supported as an extension.
2. Downscaling to 320x180 for Farneback Optical Flow:
   - Decision: Computing dense optical flow at 4K or 1080p is $O(W \times H)$ and would bottleneck CPU pipelines to <2 FPS. Downscaling to 320x180 reduces pixels by 96%, preserving macro camera pans and object movement vectors while running at >60 FPS on CPU.
3. Lossless Stream Copying (`-c:v copy`):
   - Decision: Rather than re-encoding the video track (which wastes CPU and degrades H.264 video quality), FFmpeg simply demuxes the video packets and remuxes them alongside the new AAC audio track. This preserves 100% bit-for-bit video quality and takes <0.2 seconds.
4. Render.com 512MB RAM Constraint:
   - Decision: Cloud free tiers like Render provide only 512MB RAM. Real generative AI models (`facebook/musicgen-small`) require ~1.5GB to load weights into memory. Squeezing into 512MB forced procedural sine waves. The correct production decision is separating the lightweight video intelligence pipeline (runs in 5 seconds on 128MB RAM) from the GPU inference audio worker (runs on Apple Silicon MPS or cloud GPU instances like HuggingFace Spaces/RunPod).

================================================================================
YOUR ROLE AS MY INTERVIEWER & TUTOR:
================================================================================
Please conduct an intensive, interactive masterclass with me following these 4 phases:

PHASE 1: Computer Vision & Mathematical Deep Dive
- Teach and drill me on Gunnar Farneback dense optical flow, Canny edge detection, HSV saturation vs BGR, K-Means 3D clustering, and the 12D emotion scoring math.
- Ask me 3 challenging questions to test if I truly understand how the CV code works under the hood.

PHASE 2: Neural MusicGen, Audio Processing & FFmpeg
- Drill me on Meta MusicGen transformer architecture, token generation (50 tokens/sec), PyDub crossfades, loudness normalization, and FFmpeg container multiplexing (`-c:v copy`).
- Ask me 3 questions on audio engineering decisions.

PHASE 3: Software Engineering, Testing & Edge Cases
- Review each of the 16 Pytest tests. Drill me on why corrupt containers fail OpenCV, how remainder segmentation handles edge cases, and why batch fault isolation is designed this way.
- Ask me 3 questions on production reliability and testing.

PHASE 4: Live Mock Senior Engineering Interview
- Fire 5 tough, rapid-fire behavioral and architectural interview questions that a Google/Meta interviewer would ask about this project.
- Evaluate my answers, provide constructive critique, and give the "ideal senior engineer" model answer.

Start by introducing yourself, giving me a 60-second summary of why this project architecture is impressive, and jump straight into PHASE 1!
```
