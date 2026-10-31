"""SocioDrift Interactive Web Studio.

Provides a modern, full-featured web interface to:
1. Drag and drop any video directly from the browser.
2. Run the end-to-end Video AI pipeline (scenes, metrics, mood, prompts).
3. Generate real cinematic audio with Meta MusicGen with one click.
4. Preview the video with synchronized AI music playback.

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
from fastapi.responses import HTMLResponse, JSONResponse
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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs_root")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

pipeline = VideoAIPipeline(segment_duration=3.0, demo_mode=False)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Save uploaded file
    target_path = UPLOAD_DIR / file.filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = pipeline.process_video(target_path, output_dir=BASE_DIR / "outputs")
        result_dict = result.to_dict()
        result_dict["video_url"] = f"/static/uploads/{file.filename}"
        return JSONResponse(result_dict)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate-music")
async def generate_music(payload: dict):
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    json_file = BASE_DIR / "outputs" / f"{Path(filename).stem}.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Analysis JSON not found")

    # Run generate_music.py using current python environment
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

    final_video_name = f"{Path(filename).stem}_with_ai_music.mp4"
    final_video_path = OUTPUTS_DIR / final_video_name
    final_video_url = f"/static/outputs/{final_video_name}" if final_video_path.exists() else None

    return {
        "status": "SUCCESS",
        "soundtrack_url": "/static/audio/final_soundtrack.wav",
        "final_video_url": final_video_url,
        "scenes": [
            {"scene_id": f"scene-{i+1}", "audio_url": f"/static/audio/scene-{i+1}.wav"}
            for i in range(len(payload.get("scenes", [])))
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def get_studio():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SocioDrift – Video AI & Music Co-Pilot Studio</title>
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
      padding: 1.25rem 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(12px);
      background: rgba(9, 10, 15, 0.85);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .logo-group { display: flex; align-items: center; gap: 0.85rem; }
    .logo-badge {
      background: var(--accent-grad);
      color: white;
      font-weight: 800;
      font-size: 1.1rem;
      padding: 0.35rem 0.75rem;
      border-radius: 8px;
      letter-spacing: -0.5px;
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .brand-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.3px; }
    .brand-subtitle { font-size: 0.78rem; color: var(--text-muted); font-weight: 500; }
    .header-pills { display: flex; gap: 0.6rem; }
    .pill {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      padding: 0.3rem 0.8rem;
      border-radius: 100px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .pill.active { border-color: rgba(99, 102, 241, 0.5); color: #c7d2fe; background: rgba(99, 102, 241, 0.1); }
    main { max-width: 1400px; margin: 0 auto; padding: 2.5rem 2rem; width: 100%; flex: 1; }
    
    /* Upload Zone */
    .upload-container {
      border: 2px dashed rgba(99, 102, 241, 0.35);
      border-radius: 20px;
      padding: 3.5rem 2rem;
      text-align: center;
      background: rgba(18, 20, 29, 0.4);
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      margin-bottom: 2.5rem;
    }
    .upload-container:hover, .upload-container.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.08);
      transform: translateY(-2px);
      box-shadow: 0 12px 30px -10px var(--accent-glow);
    }
    .upload-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      display: inline-block;
      filter: drop-shadow(0 4px 10px rgba(99, 102, 241, 0.5));
    }
    .upload-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem; }
    .upload-hint { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
    .btn {
      background: var(--accent-grad);
      color: white;
      border: none;
      padding: 0.75rem 1.75rem;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .btn:hover { opacity: 0.92; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      box-shadow: none;
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
    
    /* Studio Grid */
    .studio-grid {
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 2rem;
      display: none;
    }
    .studio-grid.active { display: grid; }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.5rem;
      backdrop-filter: blur(12px);
    }
    .panel-title {
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .video-preview-wrapper {
      border-radius: 14px;
      overflow: hidden;
      background: #000;
      aspect-ratio: 16/9;
      margin-bottom: 1.25rem;
      border: 1px solid var(--card-border);
    }
    video { width: 100%; height: 100%; object-fit: contain; display: block; }
    
    /* Audio Deck */
    .audio-deck {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-top: 1.25rem;
    }
    .deck-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; }
    .deck-title { font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; gap: 0.4rem; }
    audio { width: 100%; height: 40px; border-radius: 8px; filter: invert(0.9) hue-rotate(180deg); }
    
    /* Scene List */
    .scene-list { display: flex; flex-direction: column; gap: 1rem; }
    .scene-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem;
      transition: all 0.2s;
    }
    .scene-card:hover { border-color: rgba(99, 102, 241, 0.4); background: rgba(99, 102, 241, 0.03); }
    .scene-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.85rem;
    }
    .scene-id { font-weight: 800; font-size: 0.95rem; text-transform: uppercase; color: #c7d2fe; }
    .badge {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-calm { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-romantic { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
    .badge-action { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-role { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
    
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    .metric-box {
      background: rgba(0, 0, 0, 0.25);
      padding: 0.6rem;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .metric-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; }
    .metric-value { font-size: 0.88rem; font-weight: 700; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace; }
    
    .colors-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; }
    .color-swatch {
      width: 20px;
      height: 20px;
      border-radius: 5px;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .prompt-box {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      color: #cbd5e1;
      line-height: 1.45;
      max-height: 100px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    /* Loading overlay */
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top: 3px solid var(--accent);
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="logo-group">
      <div class="logo-badge">SD</div>
      <div>
        <div class="brand-title">SocioDrift Video AI</div>
        <div class="brand-subtitle">Automated Video Segmentation & Generative Music Studio</div>
      </div>
    </div>
    <div class="header-pills">
      <span class="pill active">● OpenCV Computer Vision</span>
      <span class="pill active">● Meta MusicGen</span>
      <span class="pill active">● Apple MPS Acceleration</span>
    </div>
  </header>

  <main>
    <!-- Upload Area -->
    <div class="upload-container" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="video/mp4,video/quicktime,video/x-msvideo" style="display:none" onchange="handleFileSelect(event)">
      <div class="upload-icon">🎬</div>
      <div class="upload-title">Drop your video here, or click to browse</div>
      <div class="upload-hint">Supports MP4, MOV, AVI (e.g. sample_video.mp4) • Analyzes scenes, motion & mood instantly</div>
      <button class="btn" type="button">Select Video File</button>
    </div>

    <!-- Active Studio View -->
    <div class="studio-grid" id="studioGrid">
      
      <!-- Left Column: Video & Master Soundtrack -->
      <div>
        <div class="panel">
          <div class="panel-title">
            <span>Video Preview</span>
            <span id="videoMetaBadge" class="badge badge-role">0.0s</span>
          </div>
          <div class="video-preview-wrapper">
            <video id="videoPlayer" controls></video>
          </div>

          <div style="margin-top: 1rem; display: flex; gap: 0.75rem;">
            <button class="btn" id="generateMusicBtn" onclick="generateMusic()" style="flex:1;">
              <span>🎵 Generate AI Music (Meta MusicGen)</span>
            </button>
          </div>

          <!-- Stitched Soundtrack Player -->
          <div class="audio-deck" id="soundtrackDeck" style="display: none;">
            <div class="deck-header">
              <div class="deck-title">🎶 Complete Synced Soundtrack</div>
              <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="badge badge-calm">9.0s • 32kHz</span>
                <a id="downloadSoundtrackBtn" class="pill" style="color: #6ee7b7; border-color: rgba(16, 185, 129, 0.4); text-decoration: none; cursor: pointer;">⬇ Download WAV</a>
              </div>
            </div>
            <audio id="soundtrackPlayer" controls></audio>
          </div>
        </div>
      </div>

      <!-- Right Column: Scene Breakdown -->
      <div>
        <div class="panel">
          <div class="panel-title">
            <span>Detected Scene Segments</span>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span id="sceneCountBadge" class="badge badge-role">0 Scenes</span>
              <a id="downloadJsonBtn" class="pill" style="color:#c7d2fe; text-decoration:none; cursor:pointer;" onclick="downloadJson()">⬇ Export JSON</a>
            </div>
          </div>

          <div class="scene-list" id="sceneList">
            <!-- Scene cards populated here -->
          </div>
        </div>
      </div>

    </div>
  </main>

  <script>
    let currentData = null;
    const dropZone = document.getElementById('dropZone');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
    });
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, () => dropZone.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length) uploadFile(files[0]);
    });

    function handleFileSelect(event) {
      const files = event.target.files;
      if (files.length) uploadFile(files[0]);
    }

    async function uploadFile(file) {
      dropZone.innerHTML = `
        <div class="spinner"></div>
        <div class="upload-title" style="margin-top:1rem;">Analyzing Video with Computer Vision...</div>
        <div class="upload-hint">Extracting optical flow, luminance, edge complexity & mood...</div>
      `;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        if (!res.ok) throw new Error(await res.text());
        currentData = await res.json();
        renderStudio(currentData);
      } catch (err) {
        alert('Analysis failed: ' + err.message);
        location.reload();
      }
    }

    function renderStudio(data) {
      dropZone.style.display = 'none';
      const grid = document.getElementById('studioGrid');
      grid.classList.add('active');

      const video = document.getElementById('videoPlayer');
      video.src = data.video_url;

      document.getElementById('videoMetaBadge').textContent = `${data.video_duration}s • ${data.fps} FPS • ${data.resolution}`;
      document.getElementById('sceneCountBadge').textContent = `${data.total_segments} Scenes`;

      const sceneList = document.getElementById('sceneList');
      sceneList.innerHTML = '';

      data.scenes.forEach(scene => {
        const moodBadgeClass = scene.mood === 'calm' ? 'badge-calm' : (scene.mood === 'romantic' ? 'badge-romantic' : 'badge-action');
        const colorsHtml = (scene.metrics.dominant_colors || []).map(c => 
          `<div class="color-swatch" style="background:${c}" title="${c}"></div>`
        ).join('');

        const card = document.createElement('div');
        card.className = 'scene-card';
        card.innerHTML = `
          <div class="scene-card-header">
            <div>
              <span class="scene-id">${scene.scene_id}</span>
              <span style="font-size:0.8rem; color:#94a3b8; margin-left:0.5rem;">(${scene.start}s - ${scene.end}s)</span>
            </div>
            <div>
              <span class="badge ${moodBadgeClass}">${scene.mood}</span>
              <span class="badge badge-role">${scene.story_role}</span>
            </div>
          </div>

          <div class="metrics-row">
            <div class="metric-box">
              <div class="metric-label">Style / BPM</div>
              <div class="metric-value">${scene.music_cue.tempo} BPM</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Motion / Flow</div>
              <div class="metric-value">${scene.metrics.camera_motion_intensity}</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Brightness</div>
              <div class="metric-value">${Math.round(scene.metrics.average_brightness)}</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Instrument</div>
              <div class="metric-value" style="font-size:0.75rem;">${scene.music_cue.primary_instrument}</div>
            </div>
          </div>

          <div class="colors-row">
            <span style="font-size:0.75rem; color:#94a3b8;">Dominant Colors:</span>
            ${colorsHtml}
          </div>

          <div class="prompt-box"><strong>Generated AI Prompt:</strong>\n${scene.prompt}</div>

          <div id="audioRow_${scene.scene_id}" style="margin-top:0.75rem; display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
              <span style="font-size:0.75rem; font-weight:700; color:#a5b4fc;">Scene Audio:</span>
              <a id="dl_${scene.scene_id}" class="pill" style="font-size:0.68rem; color:#93c5fd; text-decoration:none;" download="${scene.scene_id}.wav">⬇ Download WAV</a>
            </div>
            <audio id="audio_${scene.scene_id}" controls style="width:100%; height:32px;"></audio>
          </div>
        `;
        sceneList.appendChild(card);
      });
    }

    function downloadJson() {
      if (!currentData) return;
      const blob = new Blob([JSON.stringify(currentData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (currentData.filename ? currentData.filename.replace(/\\.[^/.]+$/, "") : "analysis") + ".json";
      a.click();
      URL.revokeObjectURL(url);
    }

    async function generateMusic() {
      const btn = document.getElementById('generateMusicBtn');
      btn.disabled = true;
      btn.innerHTML = `<div class="spinner" style="width:18px; height:18px;"></div> <span>Synthesizing with Meta MusicGen (Apple MPS GPU)...</span>`;

      try {
        const res = await fetch('/api/generate-music', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: currentData.filename,
            scenes: currentData.scenes
          })
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        // Reveal master soundtrack
        const deck = document.getElementById('soundtrackDeck');
        deck.style.display = 'block';
        const stPlayer = document.getElementById('soundtrackPlayer');
        const stUrl = data.soundtrack_url + '?t=' + Date.now();
        stPlayer.src = stUrl;
        
        const dlSoundtrack = document.getElementById('downloadSoundtrackBtn');
        if (dlSoundtrack) {
          dlSoundtrack.href = data.soundtrack_url;
          dlSoundtrack.download = currentData.filename.replace(/\.[^/.]+$/, "") + "_soundtrack.wav";
        }

        // Reveal scene audios
        data.scenes.forEach(s => {
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

        // Switch video to final synchronized video with AI music
        if (data.final_video_url) {
          const video = document.getElementById('videoPlayer');
          video.src = data.final_video_url + '?t=' + Date.now();
          video.load();
          video.play().catch(() => {});

          let dlBtn = document.getElementById('downloadVideoBtn');
          if (!dlBtn) {
            dlBtn = document.createElement('a');
            dlBtn.id = 'downloadVideoBtn';
            dlBtn.className = 'btn';
            dlBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            dlBtn.style.marginTop = '0.85rem';
            dlBtn.style.display = 'flex';
            dlBtn.style.justifyContent = 'center';
            dlBtn.style.boxShadow = '0 6px 20px rgba(16, 185, 129, 0.4)';
            dlBtn.style.textDecoration = 'none';
            dlBtn.textContent = '⬇ Download Final Video (100% Quality Preserved)';
            video.parentElement.after(dlBtn);
          }
          dlBtn.href = data.final_video_url;
          dlBtn.download = currentData.filename.replace(/\.[^/.]+$/, "") + "_with_ai_music.mp4";
        }

        btn.innerHTML = `<span>✅ Music Synced & Merged with Video!</span>`;
        btn.disabled = false;
        btn.style.background = '#10b981';

      } catch (err) {
        alert('Generation error: ' + err.message);
        btn.disabled = false;
        btn.innerHTML = `<span>🎵 Retry Music Generation</span>`;
      }
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
    print("  * Ready for drag-and-drop video upload & AI music generation!")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
