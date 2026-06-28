# YouTube Audio Intel Technical Design

## Goal

Build a local-first workflow that takes a YouTube URL and produces:

- Original audio/media metadata.
- A normalized `16 kHz` mono WAV generated through FFmpeg.
- Timestamped transcript from a local GPU ASR model.
- Detailed Chinese analysis from a local LLM.
- CLI and HTTP API surfaces so another machine can call the GPU machine remotely.

## Local Environment

Current host capability:

- OS shell: Windows PowerShell.
- GPU: NVIDIA GeForce RTX 4070 Ti class, 12 GB VRAM.
- CUDA driver: visible through `nvidia-smi`.
- Python: 3.10.
- ASR runtime: `faster-whisper` + `ctranslate2`, CUDA compute types include `float16`.
- LLM runtime: Ollama is installed; start with `ollama serve`.
- FFmpeg: system WinGet install hit a permissions issue, so the workflow falls back to `imageio-ffmpeg`.
- CUDA runtime: `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are installed through pip and registered at runtime on Windows.

## Architecture

```text
YouTube URL
  -> yt-dlp Python API
  -> raw media + metadata + available subtitles
  -> FFmpeg
  -> audio/audio-16k-mono.wav
  -> faster-whisper on CUDA
  -> transcript.md/json/srt
  -> Ollama /api/chat
  -> analysis/final-report.md
```

## Components

### CLI

Entry point: `yai`

Commands:

- `yai check`: validate dependencies, FFmpeg path, and CTranslate2 CUDA support.
- `yai run <url>`: execute the complete workflow.
- `yai serve`: expose the workflow as an HTTP API.

### Workflow Engine

File: `src/youtube_audio_intel/workflow.py`

Responsibilities:

- Resolve FFmpeg path from `YAI_FFMPEG`, system PATH, or `imageio-ffmpeg`.
- Register pip-installed NVIDIA DLL directories before creating the Whisper model.
- Download best available audio via `yt-dlp`.
- Extract `16 kHz` mono WAV with FFmpeg.
- Transcribe with `faster-whisper`.
- Write `json`, `md`, and `srt` transcript outputs.
- Chunk transcript text and call Ollama for detailed analysis.

### API Server

File: `src/youtube_audio_intel/api.py`

Endpoints:

- `GET /health`
- `GET /queue`
- `GET /resources`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/files/{file_path}`

Job detail responses include:

- `files`: full artifact index by relative path
- `downloads`: ready-to-use absolute URLs for common outputs such as report, transcript, subtitles, metadata, and result JSON

Queue behavior:

- `YAI_MAX_WORKERS` / `--max-workers` controls concurrent jobs. Default: `1`.
- `YAI_QUEUE_MAXSIZE` / `--queue-size` controls queued jobs. Default: `100`.
- Submissions beyond queue capacity return HTTP `429`.
- Recommended GPU service setting: one worker unless the model size and VRAM budget are tested under load.
- After each job, a worker enters a cooldown period. Defaults: minimum 60 seconds, maximum 300 seconds, target GPU temperature <= 65 C and utilization <= 20%.
- `GET /resources` reports GPU metrics from `nvidia-smi` plus CPU/memory metrics from `psutil`.

Security:

- Optional API key through `YAI_API_KEY`.
- If set, callers must pass `X-API-Key`.
- Bind to `127.0.0.1` for local-only use, or `0.0.0.0` for LAN/VPN access.

### Remote Client

File: `scripts/remote_client.py`

Remote machines can submit a job, poll status, and download the final report/transcripts without installing GPU dependencies.

## Recommended Models

ASR:

- Default: `large-v3` with `device=cuda`, `compute_type=float16`.
- Faster option: `medium` or `small`.
- Lower VRAM option: `compute_type=int8_float16`.

LLM:

- Default: `qwen3:8b` because it is already installed on this host.
- Stronger if available: `qwen3:32b`, `qwen2.5:32b`, or another local long-context model.
- Faster: `gemma3n:e4b` or another small local model.

## Operations

Start local run:

```powershell
yai run "https://www.youtube.com/watch?v=VIDEO_ID"
```

Start LAN API:

```powershell
$env:YAI_API_KEY = "change-this-secret"
yai serve --host 0.0.0.0 --port 8765
```

GPU-friendly queue mode:

```powershell
yai serve --host 0.0.0.0 --port 8765 --max-workers 1 --queue-size 100 --cooldown-seconds 60 --cooldown-max-seconds 300 --cooldown-gpu-temp 65
```

Remote invocation:

```powershell
$env:YAI_SERVER_URL = "http://GPU_MACHINE_IP:8765"
$env:YAI_API_KEY = "change-this-secret"
python scripts\remote_client.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Future Extensions

- Add WhisperX for speaker diarization and word-level alignment.
- Add persistent job database instead of in-memory job state.
- Add queue concurrency controls per GPU memory budget.
- Add file upload endpoint for non-YouTube local audio/video.
- Add webhook callback when jobs complete.
