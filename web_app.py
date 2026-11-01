"""SocioDrift Interactive Web Studio.

Provides a modern, full-featured web interface to:
1. Single Video Mode: Drag & drop, scene analysis, Meta MusicGen audio synthesis, lossless video muxing.
2. Batch Video Mode: Multi-file drag & drop, sequential one-by-one queue processing, real-time cards, download buttons.
3. Refresh Persistence: State saved in localStorage so page refreshes never lose your work until reset.

Run via:
    python web_app.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sociodrift.pipeline import VideoAIPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sociodrift-web")

app = FastAPI(title="SocioDrift Video AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"
AUDIO_DIR = OUTPUTS_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs_root")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/static/logo.png")
async def get_logo():
    logo_path = STATIC_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")

pipeline = VideoAIPipeline(segment_duration=3.0, demo_mode=False)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze a single video and produce structured scene breakdowns and prompts."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    target_path = UPLOAD_DIR / file.filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = pipeline.process_video(target_path, output_dir=OUTPUTS_DIR)
        result_dict = result.to_dict()
        result_dict["video_url"] = f"/static/uploads/{file.filename}"

        # Compute dominant emotion from scenes
        mood_counts = {}
        for s in result.scenes:
            m = (s.mood or "NEUTRAL").upper()
            mood_counts[m] = mood_counts.get(m, 0) + 1
        dom_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "NEUTRAL"

        result_dict["metadata"] = {
            "duration_seconds": result.video_duration,
            "fps": result.fps,
            "total_frames": result.total_frames,
            "resolution": result.resolution,
        }
        result_dict["summary"] = {
            "dominant_emotion": dom_mood,
            "total_scenes": len(result.scenes),
        }
        return JSONResponse(result_dict)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate-music")
async def generate_music(payload: dict):
    """Run Meta MusicGen audio generation and lossless video muxing for a video analysis."""
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    stem = Path(filename).stem
    json_file = OUTPUTS_DIR / f"{stem}.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Analysis JSON not found")

    cmd = [
        sys.executable,
        str(BASE_DIR / "generate_music.py"),
        "--json",
        str(json_file),
        "--output-dir",
        str(AUDIO_DIR),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("MusicGen output: %s", proc.stdout)
    except subprocess.CalledProcessError as exc:
        logger.error("Music generation failed: %s", exc.stderr)
        raise HTTPException(status_code=500, detail=exc.stderr or "Generation failed")

    soundtrack_name = f"{stem}_soundtrack.wav" if (AUDIO_DIR / f"{stem}_soundtrack.wav").exists() else "final_soundtrack.wav"
    final_video_name = f"{stem}_with_ai_music.mp4"
    final_video_path = OUTPUTS_DIR / final_video_name
    final_video_url = f"/static/outputs/{final_video_name}" if final_video_path.exists() else None

    scenes_data = payload.get("scenes", [])
    scenes_response = []
    for i in range(len(scenes_data)):
        scene_id = f"scene-{i+1}"
        stem_scene = AUDIO_DIR / f"{stem}_{scene_id}.wav"
        url = f"/static/audio/{stem}_{scene_id}.wav" if stem_scene.exists() else f"/static/audio/{scene_id}.wav"
        scenes_response.append({"scene_id": scene_id, "audio_url": url})

    return {
        "status": "SUCCESS",
        "soundtrack_url": f"/static/audio/{soundtrack_name}",
        "final_video_url": final_video_url,
        "scenes": scenes_response,
    }


@app.post("/api/batch-process-video")
async def batch_process_video(file: UploadFile = File(...)):
    """Complete end-to-end processing for a batch queue item: analyzes, generates music & muxes video."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    target_path = UPLOAD_DIR / file.filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Step 1: Run Video AI Pipeline
        result = pipeline.process_video(target_path, output_dir=OUTPUTS_DIR)
        result_dict = result.to_dict()
        stem = Path(file.filename).stem
        json_file = OUTPUTS_DIR / f"{stem}.json"

        # Calculate dominant mood
        mood_counts = {}
        for s in result.scenes:
            m = (s.mood or "NEUTRAL").upper()
            mood_counts[m] = mood_counts.get(m, 0) + 1
        dom_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "NEUTRAL"

        result_dict["video_url"] = f"/static/uploads/{file.filename}"
        result_dict["metadata"] = {
            "duration_seconds": result.video_duration,
            "fps": result.fps,
            "total_frames": result.total_frames,
            "resolution": result.resolution,
        }
        result_dict["summary"] = {
            "dominant_emotion": dom_mood,
            "total_scenes": len(result.scenes),
        }

        # Step 2: Run Meta MusicGen AI music generation & lossless video muxing
        cmd = [
            sys.executable,
            str(BASE_DIR / "generate_music.py"),
            "--json",
            str(json_file),
            "--output-dir",
            str(AUDIO_DIR),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Batch MusicGen output for %s: %s", file.filename, proc.stdout)

        soundtrack_name = f"{stem}_soundtrack.wav" if (AUDIO_DIR / f"{stem}_soundtrack.wav").exists() else "final_soundtrack.wav"
        final_video_name = f"{stem}_with_ai_music.mp4"
        final_video_path = OUTPUTS_DIR / final_video_name

        scenes_data = result_dict.get("scenes", [])
        scenes_list = []
        for i in range(len(scenes_data)):
            scene_id = f"scene-{i+1}"
            stem_scene = AUDIO_DIR / f"{stem}_{scene_id}.wav"
            url = f"/static/audio/{stem}_{scene_id}.wav" if stem_scene.exists() else f"/static/audio/{scene_id}.wav"
            scenes_list.append({"scene_id": scene_id, "audio_url": url})

        return JSONResponse({
            "status": "SUCCESS",
            "filename": file.filename,
            "stem": stem,
            "video_url": f"/static/uploads/{file.filename}",
            "soundtrack_url": f"/static/audio/{soundtrack_name}",
            "final_video_url": f"/static/outputs/{final_video_name}" if final_video_path.exists() else None,
            "duration": round(result.video_duration, 2),
            "scenes_count": len(scenes_data),
            "dominant_emotion": dom_mood,
            "scenes": scenes_list,
            "analysis": result_dict,
        })
    except Exception as exc:
        logger.error("Batch processing failed for %s: %s", file.filename, exc)
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "filename": file.filename, "detail": str(exc)},
        )


@app.get("/", response_class=HTMLResponse)
async def get_studio():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SocioDrift – Video AI & Music Co-Pilot Studio</title>
  <link rel="icon" type="image/png" href="/static/logo.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090a0f;
      --card-bg: rgba(18, 20, 29, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-grad: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }
    header {
      border-bottom: 1px solid var(--card-border);
      padding: 0.95rem 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(12px);
      background: rgba(9, 10, 15, 0.85);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .logo-group {
      display: flex;
      align-items: center;
      gap: 1.1rem;
    }
    .logo-link {
      display: flex;
      align-items: center;
      text-decoration: none;
    }
    .brand-logo-img {
      height: 48px;
      width: auto;
      object-fit: contain;
      filter: drop-shadow(0 0 16px rgba(168, 85, 247, 0.45)) drop-shadow(0 1px 2px rgba(255, 255, 255, 0.4));
      transition: transform 0.25s ease, filter 0.25s ease;
      cursor: pointer;
    }
    .brand-logo-img:hover {
      transform: scale(1.04);
      filter: drop-shadow(0 0 24px rgba(168, 85, 247, 0.75)) drop-shadow(0 1px 4px rgba(255, 255, 255, 0.6));
    }
    .brand-tagline-container {
      display: flex;
      flex-direction: column;
      gap: 3px;
      border-left: 1px solid var(--card-border);
      padding-left: 1rem;
    }
    .studio-pill {
      font-size: 0.68rem;
      font-weight: 800;
      letter-spacing: 1.2px;
      color: #c7d2fe;
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      padding: 1px 7px;
      border-radius: 4px;
      width: fit-content;
    }
    .brand-subtitle { font-size: 0.74rem; color: var(--text-muted); font-weight: 500; }
    
    .nav-tabs {
      display: flex;
      background: rgba(255, 255, 255, 0.04);
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      gap: 4px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.5rem 1.1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.2s ease;
    }
    .tab-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.06); }
    .tab-btn.active {
      background: var(--accent);
      color: white;
      box-shadow: 0 2px 10px var(--accent-glow);
    }
    .badge-count {
      background: rgba(255, 255, 255, 0.2);
      font-size: 0.7rem;
      padding: 1px 6px;
      border-radius: 20px;
    }

    .header-actions { display: flex; align-items: center; gap: 0.75rem; }
    .btn-reset {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 0.45rem 0.9rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s;
    }
    .btn-reset:hover { background: rgba(239, 68, 68, 0.2); }

    main { max-width: 1400px; margin: 0 auto; padding: 2rem; width: 100%; flex: 1; }
    
    /* Upload Zone */
    .upload-container {
      border: 2px dashed rgba(99, 102, 241, 0.35);
      border-radius: 20px;
      padding: 3rem 2rem;
      text-align: center;
      background: rgba(18, 20, 29, 0.4);
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      margin-bottom: 2rem;
    }
    .upload-container:hover, .upload-container.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.08);
      transform: translateY(-2px);
      box-shadow: 0 12px 30px -10px var(--accent-glow);
    }
    .upload-icon {
      font-size: 3rem;
      margin-bottom: 0.75rem;
      display: inline-block;
      filter: drop-shadow(0 4px 10px rgba(99, 102, 241, 0.5));
    }
    .upload-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.4rem; }
    .upload-hint { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 1.25rem; }
    
    .btn {
      background: var(--accent-grad);
      color: white;
      border: none;
      padding: 0.75rem 1.6rem;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.92rem;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      text-decoration: none;
    }
    .btn:hover { opacity: 0.92; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      box-shadow: none;
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
    .btn-success {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .btn-success:hover { box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45); }

    /* Single Studio Grid */
    .studio-grid {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 2rem;
      align-items: start;
    }
    @media (max-width: 1024px) {
      .studio-grid { grid-template-columns: 1fr; }
    }
    
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.5rem;
      backdrop-filter: blur(10px);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
      padding-bottom: 0.85rem;
      border-bottom: 1px solid var(--card-border);
    }
    .panel-title { font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }

    .video-wrapper {
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    video { width: 100%; height: 100%; object-fit: contain; }

    .meta-table { width: 100%; font-size: 0.85rem; margin-bottom: 1.25rem; }
    .meta-table td { padding: 0.4rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
    .meta-table td:first-child { color: var(--text-muted); width: 40%; }
    .meta-table td:last-child { font-weight: 600; text-align: right; }

    /* Audio Player Custom */
    .audio-deck {
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 14px;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
    }
    .audio-deck audio { width: 100%; height: 40px; margin-top: 0.6rem; border-radius: 8px; }

    /* Scenes List */
    .scenes-grid {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: 820px;
      overflow-y: auto;
      padding-right: 0.5rem;
    }
    .scene-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.15rem;
      transition: all 0.2s;
    }
    .scene-card:hover {
      border-color: rgba(99, 102, 241, 0.4);
      background: rgba(255, 255, 255, 0.05);
    }
    .scene-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }
    .scene-badge {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .mood-SUSPENSE { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .mood-CALM { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .mood-ENERGETIC { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .mood-NEUTRAL { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
    .mood-MELANCHOLIC { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }

    .metrics-chips {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
    }
    .chip {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.25rem 0.55rem;
      border-radius: 6px;
      font-size: 0.74rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .chip strong { color: var(--text); }

    .prompt-box {
      background: rgba(0, 0, 0, 0.35);
      border-left: 3px solid var(--accent);
      padding: 0.65rem 0.85rem;
      border-radius: 0 8px 8px 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      line-height: 1.45;
      color: #e2e8f0;
      white-space: pre-wrap;
    }
    .colors-row { display: flex; gap: 0.35rem; align-items: center; margin: 0.6rem 0; }
    .color-swatch {
      width: 18px;
      height: 18px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Batch Mode Styles */
    .batch-queue-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }
    .batch-queue-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.75rem 1.25rem;
      font-size: 0.88rem;
    }
    .batch-results-list {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .batch-result-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 1.5rem;
      align-items: center;
      transition: all 0.25s;
    }
    @media (max-width: 900px) {
      .batch-result-card { grid-template-columns: 1fr; }
    }
    .batch-result-card.success { border-color: rgba(16, 185, 129, 0.35); }
    .batch-result-card.error { border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.05); }

    .progress-bar-container {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 100px;
      overflow: hidden;
      margin: 1rem 0;
    }
    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent-grad);
      transition: width 0.4s ease;
    }

    /* Spinner */
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.15);
      border-top-color: var(--accent);
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .pulse { animation: pulseAnim 2s infinite ease-in-out; }
    @keyframes pulseAnim {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }
  </style>
</head>
<body>

  <header>
    <div class="logo-group">
      <a href="/" class="logo-link" title="SocioDrift Studio Home">
        <img src="/static/logo.png" alt="SocioDrift" class="brand-logo-img">
      </a>
      <div class="brand-tagline-container">
        <span class="studio-pill">STUDIO</span>
        <span class="brand-subtitle">AI Video & Neural Music Co-Pilot</span>
      </div>
    </div>

    <!-- Mode Selector Tabs -->
    <div class="nav-tabs">
      <button class="tab-btn active" id="tabSingleBtn" onclick="switchMode('single')">
        <span>🎬 Single Video Studio</span>
      </button>
      <button class="tab-btn" id="tabBatchBtn" onclick="switchMode('batch')">
        <span>⚡ Multi-Video Batch Queue</span>
        <span class="badge-count" id="batchBadge">0</span>
      </button>
    </div>

    <div class="header-actions">
      <button class="btn-reset" onclick="resetActiveWorkspace()" title="Clear cached session data and reset studio">
        <span>🔄 Reset Studio</span>
      </button>
    </div>
  </header>

  <main>
    <!-- ==================== TAB 1: SINGLE VIDEO STUDIO ==================== -->
    <section id="singleModeSection">
      <!-- Upload Zone -->
      <div class="upload-container" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="video/mp4,video/quicktime,video/x-msvideo,video/webm" style="display:none;" onchange="handleSingleFileSelect(event)">
        <div class="upload-icon">🎬</div>
        <div class="upload-title">Drop your video here or click to browse</div>
        <div class="upload-hint">Supports MP4, MOV, AVI, WebM. Evaluates motion, colors, lighting & suspense.</div>
        <button class="btn" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">Select Video File</button>
      </div>

      <!-- Loading State -->
      <div id="singleLoading" class="panel" style="display:none; text-align:center; padding:3rem; margin-bottom:2rem;">
        <div class="spinner" style="width:40px; height:40px; margin-bottom:1rem;"></div>
        <div id="singleLoadingText" style="font-size:1.15rem; font-weight:700; margin-bottom:0.5rem;">Analyzing Video Dynamics...</div>
        <div style="color:var(--text-muted); font-size:0.9rem;">Extracting frames, computing optical flow, lighting balance & AI musical prompts...</div>
      </div>

      <!-- Studio Interactive Workspace -->
      <div id="studioGrid" class="studio-grid" style="display:none;">
        
        <!-- Left Column: Video Preview & Primary Controls -->
        <div>
          <div class="panel">
            <div class="panel-header">
              <span class="panel-title">📹 Video Preview</span>
              <span id="videoStatusPill" class="scene-badge mood-CALM">ORIGINAL</span>
            </div>

            <div class="video-wrapper">
              <video id="videoPlayer" controls playsinline></video>
            </div>

            <table class="meta-table">
              <tr><td>Filename:</td><td id="metaFilename">-</td></tr>
              <tr><td>Duration:</td><td id="metaDuration">-</td></tr>
              <tr><td>Resolution:</td><td id="metaRes">-</td></tr>
              <tr><td>FPS / Codec:</td><td id="metaFps">-</td></tr>
              <tr><td>Dominant Mood:</td><td id="metaMood">-</td></tr>
              <tr><td>Analyzed Scenes:</td><td id="metaScenesCount">-</td></tr>
            </table>

            <div style="display:flex; flex-direction:column; gap:0.75rem;">
              <button id="generateMusicBtn" class="btn" onclick="generateMusic()">
                <span>🎵 Synthesize AI Music & Multiplex Video</span>
              </button>
              
              <div id="finalVideoDownloadRow" style="display:none;">
                <a id="downloadFinalVideoBtn" class="btn btn-success" style="width:100%;" download>
                  <span>⬇ Download Final Video (100% Quality)</span>
                </a>
              </div>

              <div style="display:flex; gap:0.5rem;">
                <button class="btn btn-secondary" style="flex:1;" onclick="downloadJson()">
                  <span>📄 Export JSON</span>
                </button>
                <button class="btn btn-secondary" style="flex:1;" onclick="document.getElementById('fileInput').click()">
                  <span>🔄 New Video</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: AI Audio & Scene Breakdown -->
        <div>
          <!-- Master Soundtrack Deck (Revealed when generated) -->
          <div id="soundtrackDeck" class="audio-deck" style="display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <div style="font-size:1.05rem; font-weight:700; color:#c7d2fe;">🎶 Meta MusicGen Master Soundtrack</div>
                <div style="font-size:0.78rem; color:var(--text-muted); margin-top:2px;">500ms PyDub Equal-Power Crossfade • -16 dBFS Loudness Normalized • 1000ms Fade-out</div>
              </div>
              <a id="downloadSoundtrackBtn" class="btn btn-secondary" style="font-size:0.8rem; padding:0.4rem 0.85rem;" download>
                <span>⬇ Download Audio (.wav)</span>
              </a>
            </div>
            <audio id="soundtrackPlayer" controls></audio>
          </div>

          <!-- Scene Cue Breakdown List -->
          <div class="panel">
            <div class="panel-header">
              <span class="panel-title">🎼 Scene Cues & AI Music Prompts</span>
              <span id="sceneCountBadge" style="font-size:0.8rem; color:var(--text-muted);">0 Scenes</span>
            </div>

            <div class="scenes-grid" id="sceneList">
              <!-- Dynamically populated scene cards -->
            </div>
          </div>
        </div>

      </div>
    </section>

    <!-- ==================== TAB 2: MULTI-VIDEO BATCH QUEUE ==================== -->
    <section id="batchModeSection" style="display:none;">
      <!-- Batch Upload Zone -->
      <div class="upload-container" id="batchDropZone" onclick="document.getElementById('batchFileInput').click()">
        <input type="file" id="batchFileInput" multiple accept="video/mp4,video/quicktime,video/x-msvideo,video/webm" style="display:none;" onchange="handleBatchFilesSelect(event)">
        <div class="upload-icon">⚡</div>
        <div class="upload-title">Drop multiple videos here for Batch AI Processing</div>
        <div class="upload-hint">Upload multiple video files. Each will be analyzed, synthesized with Meta MusicGen, and muxed one by one.</div>
        <button class="btn" onclick="event.stopPropagation(); document.getElementById('batchFileInput').click()">Select Multiple Videos</button>
      </div>

      <!-- Batch Queue Status Panel -->
      <div id="batchQueuePanel" class="panel" style="margin-bottom:2rem; display:none;">
        <div class="panel-header">
          <span class="panel-title">📋 Video Processing Queue</span>
          <div style="display:flex; gap:0.5rem;">
            <button id="startBatchBtn" class="btn btn-success" onclick="startBatchProcessing()">
              <span>🚀 Start Batch Processing</span>
            </button>
            <button class="btn btn-secondary" onclick="clearBatchQueue()">
              <span>Clear Queue</span>
            </button>
          </div>
        </div>

        <div class="progress-bar-container" id="batchProgressBarBox" style="display:none;">
          <div class="progress-bar-fill" id="batchProgressBar"></div>
        </div>
        <div id="batchProgressText" style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem; display:none;"></div>

        <div class="batch-queue-list" id="batchQueueList"></div>
      </div>

      <!-- Completed Batch Results Feed -->
      <div id="batchResultsContainer" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.25rem;">
          <h2 style="font-size:1.25rem; font-weight:700;">🎬 Completed Video AI Soundtracks (<span id="batchCompletedCount">0</span>)</h2>
          <button class="btn btn-secondary" onclick="clearBatchResults()">Clear Results History</button>
        </div>
        <div class="batch-results-list" id="batchResultsList"></div>
      </div>
    </section>
  </main>

  <script>
    // LocalStorage Keys
    const STORAGE_KEY_SINGLE = 'sociodrift_studio_single_v1';
    const STORAGE_KEY_BATCH = 'sociodrift_studio_batch_v1';
    const STORAGE_KEY_TAB = 'sociodrift_studio_active_tab';

    // State
    let singleState = null;
    let batchQueue = [];
    let batchResults = [];
    let isBatchRunning = false;

    // Initialization
    window.addEventListener('DOMContentLoaded', () => {
      setupDragAndDrop();
      restoreSavedState();
    });

    // Tab Switching
    function switchMode(mode) {
      localStorage.setItem(STORAGE_KEY_TAB, mode);
      const tabSingleBtn = document.getElementById('tabSingleBtn');
      const tabBatchBtn = document.getElementById('tabBatchBtn');
      const singleSection = document.getElementById('singleModeSection');
      const batchSection = document.getElementById('batchModeSection');

      if (mode === 'batch') {
        tabSingleBtn.classList.remove('active');
        tabBatchBtn.classList.add('active');
        singleSection.style.display = 'none';
        batchSection.style.display = 'block';
      } else {
        tabBatchBtn.classList.remove('active');
        tabSingleBtn.classList.add('active');
        batchSection.style.display = 'none';
        singleSection.style.display = 'block';
      }
    }

    // Restore from localStorage
    function restoreSavedState() {
      const savedTab = localStorage.getItem(STORAGE_KEY_TAB);
      if (savedTab === 'batch') switchMode('batch');

      // 1. Restore Single State
      try {
        const rawSingle = localStorage.getItem(STORAGE_KEY_SINGLE);
        if (rawSingle) {
          singleState = JSON.parse(rawSingle);
          if (singleState && singleState.data) {
            renderSingleAnalysis(singleState.data, false);
            if (singleState.music) {
              renderGeneratedMusic(singleState.music);
            }
          }
        }
      } catch (e) {
        console.warn('Could not restore single state:', e);
      }

      // 2. Restore Batch State
      try {
        const rawBatch = localStorage.getItem(STORAGE_KEY_BATCH);
        if (rawBatch) {
          batchResults = JSON.parse(rawBatch) || [];
          if (batchResults.length > 0) {
            renderAllBatchResults();
          }
        }
      } catch (e) {
        console.warn('Could not restore batch state:', e);
      }
      updateBatchBadge();
    }

    function resetActiveWorkspace() {
      if (confirm('Reset workspace and clear cached video sessions?')) {
        localStorage.removeItem(STORAGE_KEY_SINGLE);
        localStorage.removeItem(STORAGE_KEY_BATCH);
        localStorage.removeItem(STORAGE_KEY_TAB);
        location.reload();
      }
    }

    // Drag and Drop
    function setupDragAndDrop() {
      const dropZone = document.getElementById('dropZone');
      ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
      });
      ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
      });
      dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
          if (files.length > 1) {
            // User dropped multiple files into single zone -> seamlessly route to batch mode!
            switchMode('batch');
            queueBatchFiles(Array.from(files));
          } else {
            uploadSingleFile(files[0]);
          }
        }
      });

      const batchDropZone = document.getElementById('batchDropZone');
      ['dragenter', 'dragover'].forEach(name => {
        batchDropZone.addEventListener(name, (e) => { e.preventDefault(); batchDropZone.classList.add('dragover'); });
      });
      ['dragleave', 'drop'].forEach(name => {
        batchDropZone.addEventListener(name, (e) => { e.preventDefault(); batchDropZone.classList.remove('dragover'); });
      });
      batchDropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
          queueBatchFiles(Array.from(files));
        }
      });
    }

    // ==================== SINGLE VIDEO LOGIC ====================
    function handleSingleFileSelect(e) {
      const file = e.target.files[0];
      if (file) uploadSingleFile(file);
    }

    async function uploadSingleFile(file) {
      const loading = document.getElementById('singleLoading');
      const dropZone = document.getElementById('dropZone');
      const studioGrid = document.getElementById('studioGrid');

      dropZone.style.display = 'none';
      studioGrid.style.display = 'none';
      loading.style.display = 'block';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        
        singleState = { data: data, music: null };
        localStorage.setItem(STORAGE_KEY_SINGLE, JSON.stringify(singleState));

        loading.style.display = 'none';
        renderSingleAnalysis(data, true);
      } catch (err) {
        loading.style.display = 'none';
        dropZone.style.display = 'block';
        alert('Analysis error: ' + err.message);
      }
    }

    function renderSingleAnalysis(data, autoPlay = false) {
      document.getElementById('dropZone').style.display = 'none';
      const grid = document.getElementById('studioGrid');
      grid.style.display = 'grid';

      // Video preview
      const video = document.getElementById('videoPlayer');
      video.src = data.video_url;
      video.onloadedmetadata = () => {
        if (video.duration && !isNaN(video.duration) && video.duration > 0) {
          document.getElementById('metaDuration').textContent = `${video.duration.toFixed(1)}s`;
        }
      };
      if (autoPlay) video.play().catch(() => {});

      // Metadata extraction
      const durationVal = data.video_duration ?? data.duration ?? data.metadata?.duration_seconds ?? 0;
      const resVal = data.resolution ?? (data.metadata?.width ? `${data.metadata.width}x${data.metadata.height}` : '-');
      const fpsVal = data.fps ?? data.metadata?.fps ?? 24;
      const framesVal = data.total_frames ?? data.metadata?.total_frames ?? '-';

      document.getElementById('metaFilename').textContent = data.filename || 'Uploaded Video';
      document.getElementById('metaDuration').textContent = `${Number(durationVal).toFixed(1)}s`;
      document.getElementById('metaRes').textContent = resVal;
      document.getElementById('metaFps').textContent = `${Number(fpsVal).toFixed(0)} FPS (${framesVal} frames)`;

      // Dominant Mood calculation
      let domMood = data.summary?.dominant_emotion;
      if (!domMood && data.scenes && data.scenes.length > 0) {
        const counts = {};
        data.scenes.forEach(s => {
          const m = (s.mood || s.primary_emotion || 'NEUTRAL').toUpperCase();
          counts[m] = (counts[m] || 0) + 1;
        });
        domMood = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b, 'NEUTRAL');
      }
      domMood = (domMood || 'NEUTRAL').toUpperCase();

      const moodEl = document.getElementById('metaMood');
      moodEl.textContent = domMood;
      moodEl.className = `scene-badge mood-${domMood}`;

      const scenesCount = data.scenes?.length || data.total_segments || 0;
      document.getElementById('metaScenesCount').textContent = `${scenesCount} segments`;
      document.getElementById('sceneCountBadge').textContent = `${scenesCount} Scenes`;

      // Render Scene Cards
      const sceneList = document.getElementById('sceneList');
      sceneList.innerHTML = '';

      (data.scenes || []).forEach((scene, idx) => {
        const card = document.createElement('div');
        card.className = 'scene-card';
        const sceneId = scene.scene_id || `scene-${idx + 1}`;
        card.id = `card_${sceneId}`;

        // Robust time extraction
        const startTime = Number(scene.start !== undefined ? scene.start : (scene.start_time !== undefined ? scene.start_time : idx * 3.0));
        const endTime = Number(scene.end !== undefined ? scene.end : (scene.end_time !== undefined ? scene.end_time : startTime + 3.0));
        const mood = (scene.mood || scene.primary_emotion || 'NEUTRAL').toUpperCase();

        // Robust metrics extraction
        const motion = Number(scene.metrics?.optical_flow ?? scene.metrics?.camera_motion_intensity ?? scene.features?.motion_score ?? 0);
        const brightness = Number(scene.metrics?.average_brightness ?? scene.features?.brightness ?? 0);
        const energy = Number(scene.music_cue?.energy ?? scene.music_cues?.energy_percent ?? 50);
        const tempo = Number(scene.music_cue?.tempo ?? scene.music_cues?.tempo_bpm ?? 110);

        // Dominant colors extraction
        const colors = scene.metrics?.dominant_colors || scene.features?.dominant_colors || [];
        const colorsHtml = colors.map(c => {
          const bg = typeof c === 'string' ? c : `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
          return `<div class="color-swatch" style="background: ${bg};" title="${bg}"></div>`;
        }).join('');

        const promptText = scene.prompt || scene.description || 'AI Music cue planned for this scene.';

        card.innerHTML = `
          <div class="scene-top">
            <div style="display:flex; align-items:center; gap:0.6rem;">
              <strong style="color:#c7d2fe; font-size:0.95rem;">${sceneId.toUpperCase()}</strong>
              <span style="font-size:0.75rem; color:#94a3b8; font-family:'JetBrains Mono';">${startTime.toFixed(1)}s – ${endTime.toFixed(1)}s</span>
            </div>
            <span class="scene-badge mood-${mood}">${mood}</span>
          </div>

          <div class="metrics-chips">
            <div class="chip">Motion: <strong>${motion.toFixed(2)}</strong></div>
            <div class="chip">Light: <strong>${brightness.toFixed(0)}</strong></div>
            <div class="chip">Energy: <strong>${energy.toFixed(0)}%</strong></div>
            <div class="chip">Tempo: <strong>${tempo.toFixed(0)} BPM</strong></div>
          </div>

          <div class="colors-row">
            <span style="font-size:0.75rem; color:#94a3b8;">Dominant Colors:</span>
            ${colorsHtml || '<span style="font-size:0.75rem; color:#64748b;">(auto-detected)</span>'}
          </div>

          <div class="prompt-box"><strong>Generated AI Prompt:</strong>\n${promptText}</div>

          <div id="audioRow_${sceneId}" style="margin-top:0.75rem; display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
              <span style="font-size:0.75rem; font-weight:700; color:#a5b4fc;">Scene Audio:</span>
              <a id="dl_${sceneId}" class="pill" style="font-size:0.68rem; color:#93c5fd; text-decoration:none;" download="${sceneId}.wav">⬇ Download WAV</a>
            </div>
            <audio id="audio_${sceneId}" controls style="width:100%; height:32px;"></audio>
          </div>
        `;
        sceneList.appendChild(card);
      });
    }

    async function generateMusic() {
      if (!singleState || !singleState.data) return;

      const btn = document.getElementById('generateMusicBtn');
      btn.disabled = true;
      btn.innerHTML = `<div class="spinner" style="width:18px; height:18px;"></div> <span>Synthesizing with Meta MusicGen (Apple MPS GPU)...</span>`;

      try {
        const res = await fetch('/api/generate-music', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: singleState.data.filename,
            scenes: singleState.data.scenes
          })
        });

        if (!res.ok) throw new Error(await res.text());
        const musicData = await res.json();

        // Save state to localStorage
        singleState.music = musicData;
        localStorage.setItem(STORAGE_KEY_SINGLE, JSON.stringify(singleState));

        renderGeneratedMusic(musicData);
      } catch (err) {
        alert('Generation error: ' + err.message);
        btn.disabled = false;
        btn.innerHTML = `<span>🎵 Retry Music Generation</span>`;
      }
    }

    function renderGeneratedMusic(musicData) {
      const btn = document.getElementById('generateMusicBtn');
      btn.innerHTML = `<span>✅ Music Synced & Multiplexed with Video!</span>`;
      btn.disabled = false;
      btn.style.background = '#10b981';

      // Reveal master soundtrack
      const deck = document.getElementById('soundtrackDeck');
      deck.style.display = 'block';
      const stPlayer = document.getElementById('soundtrackPlayer');
      stPlayer.src = musicData.soundtrack_url + '?t=' + Date.now();
      
      const dlSoundtrack = document.getElementById('downloadSoundtrackBtn');
      if (dlSoundtrack) {
        dlSoundtrack.href = musicData.soundtrack_url;
        dlSoundtrack.download = (singleState.data.filename || "video").replace(/\.[^/.]+$/, "") + "_soundtrack.wav";
      }

      // Reveal scene audios
      (musicData.scenes || []).forEach(s => {
        const row = document.getElementById(`audioRow_${s.scene_id}`);
        if (row) {
          row.style.display = 'block';
          const audioElem = document.getElementById(`audio_${s.scene_id}`);
          audioElem.src = s.audio_url + '?t=' + Date.now();
          const dlScene = document.getElementById(`dl_${s.scene_id}`);
          if (dlScene) {
            dlScene.href = s.audio_url;
          }
        }
      });

      // Switch video player to final lossless video with AI music
      if (musicData.final_video_url) {
        const video = document.getElementById('videoPlayer');
        video.src = musicData.final_video_url + '?t=' + Date.now();
        video.load();

        const pill = document.getElementById('videoStatusPill');
        if (pill) {
          pill.textContent = 'AI SOUNDTRACK SYNCED (100% QUALITY)';
          pill.className = 'scene-badge mood-CALM';
        }

        const dlRow = document.getElementById('finalVideoDownloadRow');
        const dlBtn = document.getElementById('downloadFinalVideoBtn');
        if (dlRow && dlBtn) {
          dlRow.style.display = 'block';
          dlBtn.href = musicData.final_video_url;
          dlBtn.download = (singleState.data.filename || "video").replace(/\.[^/.]+$/, "") + "_with_ai_music.mp4";
        }
      }
    }

    function downloadJson() {
      if (!singleState || !singleState.data) return;
      const blob = new Blob([JSON.stringify(singleState.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (singleState.data.filename ? singleState.data.filename.replace(/\.[^/.]+$/, "") : "analysis") + ".json";
      a.click();
      URL.revokeObjectURL(url);
    }

    // ==================== BATCH MULTI-VIDEO LOGIC ====================
    function handleBatchFilesSelect(e) {
      const files = Array.from(e.target.files);
      if (files.length > 0) queueBatchFiles(files);
    }

    function queueBatchFiles(files) {
      files.forEach(f => {
        // Prevent duplicates in current queue
        if (!batchQueue.some(item => item.file.name === f.name && item.file.size === f.size)) {
          batchQueue.push({ file: f, status: 'pending' });
        }
      });
      renderBatchQueue();
      updateBatchBadge();
    }

    function renderBatchQueue() {
      const panel = document.getElementById('batchQueuePanel');
      const list = document.getElementById('batchQueueList');
      if (batchQueue.length === 0) {
        panel.style.display = 'none';
        return;
      }
      panel.style.display = 'block';
      list.innerHTML = '';

      batchQueue.forEach((item, idx) => {
        const row = document.createElement('div');
        row.className = 'batch-queue-item';
        
        let statusBadge = '<span class="scene-badge mood-NEUTRAL">Pending</span>';
        if (item.status === 'processing') {
          statusBadge = '<span class="scene-badge mood-ENERGETIC pulse">Processing...</span>';
        } else if (item.status === 'completed') {
          statusBadge = '<span class="scene-badge mood-CALM">Completed</span>';
        } else if (item.status === 'failed') {
          statusBadge = '<span class="scene-badge mood-SUSPENSE">Failed</span>';
        }

        row.innerHTML = `
          <div style="display:flex; align-items:center; gap:0.75rem;">
            <strong style="color:#c7d2fe;">#${idx + 1}</strong>
            <div>
              <div style="font-weight:600;">${item.file.name}</div>
              <div style="font-size:0.75rem; color:#94a3b8;">${(item.file.size / (1024 * 1024)).toFixed(2)} MB</div>
            </div>
          </div>
          <div>${statusBadge}</div>
        `;
        list.appendChild(row);
      });
    }

    function updateBatchBadge() {
      const badge = document.getElementById('batchBadge');
      badge.textContent = batchQueue.length + batchResults.length;
    }

    function clearBatchQueue() {
      if (isBatchRunning) {
        alert('Batch processing is actively running. Please wait for completion.');
        return;
      }
      batchQueue = [];
      renderBatchQueue();
      updateBatchBadge();
    }

    function clearBatchResults() {
      if (confirm('Clear all completed batch results?')) {
        batchResults = [];
        localStorage.removeItem(STORAGE_KEY_BATCH);
        renderAllBatchResults();
        updateBatchBadge();
      }
    }

    async function startBatchProcessing() {
      if (isBatchRunning) return;
      if (batchQueue.length === 0) {
        alert('Please add at least one video to the batch queue first.');
        return;
      }

      isBatchRunning = true;
      const startBtn = document.getElementById('startBatchBtn');
      startBtn.disabled = true;
      startBtn.innerHTML = `<div class="spinner" style="width:16px; height:16px;"></div> <span>Processing Batch...</span>`;

      const progressBox = document.getElementById('batchProgressBarBox');
      const progressBar = document.getElementById('batchProgressBar');
      const progressText = document.getElementById('batchProgressText');
      progressBox.style.display = 'block';
      progressText.style.display = 'block';

      const total = batchQueue.length;

      for (let i = 0; i < batchQueue.length; i++) {
        const item = batchQueue[i];
        item.status = 'processing';
        renderBatchQueue();

        const pct = Math.round((i / total) * 100);
        progressBar.style.width = `${pct}%`;
        progressText.innerHTML = `<strong>Processing ${i + 1} of ${total}:</strong> ${item.file.name} (Extracting scenes & synthesizing neural audio)...`;

        const formData = new FormData();
        formData.append('file', item.file);

        try {
          const res = await fetch('/api/batch-process-video', { method: 'POST', body: formData });
          if (!res.ok) {
            const errTxt = await res.text();
            throw new Error(errTxt);
          }
          const result = await res.json();
          item.status = 'completed';

          // Prepend to completed results so newest is at the top
          batchResults.unshift(result);
          localStorage.setItem(STORAGE_KEY_BATCH, JSON.stringify(batchResults));
          renderAllBatchResults();

        } catch (err) {
          console.error('Batch error on file:', item.file.name, err);
          item.status = 'failed';
          const failedResult = {
            status: 'ERROR',
            filename: item.file.name,
            detail: err.message
          };
          batchResults.unshift(failedResult);
          localStorage.setItem(STORAGE_KEY_BATCH, JSON.stringify(batchResults));
          renderAllBatchResults();
        }

        renderBatchQueue();
      }

      progressBar.style.width = '100%';
      progressText.innerHTML = `<strong>Batch Completed!</strong> Processed ${total} video clips.`;

      isBatchRunning = false;
      startBtn.disabled = false;
      startBtn.innerHTML = `<span>🚀 Process Another Batch</span>`;
    }

    function renderAllBatchResults() {
      const container = document.getElementById('batchResultsContainer');
      const list = document.getElementById('batchResultsList');
      const count = document.getElementById('batchCompletedCount');

      if (!batchResults || batchResults.length === 0) {
        container.style.display = 'none';
        return;
      }

      container.style.display = 'block';
      count.textContent = batchResults.length;
      list.innerHTML = '';

      batchResults.forEach((res, idx) => {
        const card = document.createElement('div');
        if (res.status === 'ERROR') {
          card.className = 'batch-result-card error';
          card.innerHTML = `
            <div>
              <div style="font-size:1.1rem; font-weight:700; color:#f87171; margin-bottom:0.4rem;">⚠️ ${res.filename}</div>
              <span class="scene-badge mood-SUSPENSE">PROCESSING FAILED</span>
            </div>
            <div style="color:#fca5a5; font-size:0.88rem;">
              <strong>Error Details:</strong> ${res.detail || 'Could not process video'}
            </div>
          `;
        } else {
          card.className = 'batch-result-card success';
          const videoSrc = res.final_video_url || res.video_url;
          const dlName = (res.filename || "video").replace(/\.[^/.]+$/, "");

          card.innerHTML = `
            <!-- Left: Video Player -->
            <div>
              <div class="video-wrapper" style="aspect-ratio:16/9; margin-bottom:0.75rem;">
                <video src="${videoSrc}" controls playsinline style="width:100%; height:100%; object-fit:contain;"></video>
              </div>
              <div style="font-size:0.75rem; color:#94a3b8; text-align:center;">
                ${res.final_video_url ? '✅ Merged with AI Soundtrack (100% Lossless)' : 'Original Video'}
              </div>
            </div>

            <!-- Right: Details, Audio Player & Downloads -->
            <div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <div style="font-size:1.15rem; font-weight:700; color:#f8fafc;">${res.filename}</div>
                <span class="scene-badge mood-${res.dominant_emotion || 'CALM'}">${res.dominant_emotion || 'NEUTRAL'}</span>
              </div>

              <div class="metrics-chips" style="margin-bottom:0.75rem;">
                <div class="chip">Duration: <strong>${Number(res.duration ?? res.analysis?.video_duration ?? res.analysis?.metadata?.duration_seconds ?? 0).toFixed(1)}s</strong></div>
                <div class="chip">Scenes: <strong>${res.scenes_count || (res.scenes && res.scenes.length) || (res.analysis?.scenes && res.analysis.scenes.length) || 0}</strong></div>
                <div class="chip">Format: <strong>16-bit PCM / 32kHz</strong></div>
              </div>

              <!-- Audio Player -->
              <div style="margin-bottom:1rem; background:rgba(0,0,0,0.3); padding:0.75rem; border-radius:10px;">
                <div style="font-size:0.75rem; font-weight:600; color:#a5b4fc; margin-bottom:0.35rem;">
                  🎵 Synthesized AI Soundtrack (PyDub Equal-Power Crossfade):
                </div>
                <audio src="${res.soundtrack_url}" controls style="width:100%; height:32px;"></audio>
              </div>

              <!-- Action Download Buttons -->
              <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                <a class="btn btn-secondary" style="font-size:0.8rem; padding:0.5rem 0.85rem;" href="${res.soundtrack_url}" download="${dlName}_soundtrack.wav">
                  <span>⬇ Download Audio (.wav)</span>
                </a>
                ${res.final_video_url ? `
                  <a class="btn btn-success" style="font-size:0.8rem; padding:0.5rem 0.85rem;" href="${res.final_video_url}" download="${dlName}_with_ai_music.mp4">
                    <span>⬇ Download Final Video</span>
                  </a>
                ` : ''}
                <button class="btn btn-secondary" style="font-size:0.8rem; padding:0.5rem 0.85rem;" onclick="downloadBatchItemJson(${idx})">
                  <span>📄 Export JSON</span>
                </button>
              </div>
            </div>
          `;
        }
        list.appendChild(card);
      });
    }

    function downloadBatchItemJson(idx) {
      const item = batchResults[idx];
      if (!item || !item.analysis) return;
      const blob = new Blob([JSON.stringify(item.analysis, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${item.stem || "video"}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("      SOCIODRIFT VIDEO AI INTERACTIVE WEB STUDIO      ")
    print("=" * 60)
    print("  * Studio URL : http://localhost:8000")
    print("  * Features   : Single Studio + Multi-Video Batch Processing")
    print("  * Persist    : State preserved across page refreshes")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
