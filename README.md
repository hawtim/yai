# YAI

YAI, short for YouTube Audio Intel, is a local-first service for turning a YouTube URL into structured audio intelligence:

- download best-available audio with `yt-dlp`
- normalize it to `16 kHz` mono WAV with `ffmpeg`
- transcribe it with GPU-backed `faster-whisper`
- analyze the transcript with a local Ollama model
- expose the workflow through both a CLI and an HTTP API

The project is designed for a Windows workstation with an NVIDIA GPU, while still being easy to call from another machine on the same LAN or VPN.

## Features

- Local CLI for one-off runs: `yai run <url>`
- Remote API for queue-based service usage: `POST /jobs`
- GPU-aware cooldown between jobs to avoid sustained high load
- Large backlog queue with bounded execution concurrency
- GPU telemetry from `nvidia-smi`
- CPU and memory telemetry from `psutil`
- Transcript outputs in `json`, `md`, and `srt`
- Analysis outputs in Markdown

## Requirements

- Windows with Python `3.10+`
- NVIDIA GPU for the recommended path
- Ollama installed locally
- Network access for YouTube and initial model downloads

Recommended local tools:

- `Deno` for better `yt-dlp` JavaScript extraction compatibility
- `ffmpeg` in PATH if you want a system-wide install

The project can still run without a system `ffmpeg` because it falls back to the bundled executable from `imageio-ffmpeg`.

## Installation

```powershell
cd D:\Dev
python -m pip install -e .\youtube-audio-intel
```

Optional but recommended:

```powershell
winget install -e --id DenoLand.Deno
```

Pull an Ollama model that fits your machine:

```powershell
ollama pull qwen3:8b
```

## Quick Start

Run a full local workflow:

```powershell
cd D:\Dev\youtube-audio-intel
yai check
yai run "https://www.youtube.com/watch?v=VIDEO_ID"
```

Outputs are written under:

```text
D:\Dev\youtube-audio-intel\runs\<run-id>-<video-title>\
```

Typical output files:

- `metadata.json`
- `audio/audio-16k-mono.wav`
- `transcript/transcript.json`
- `transcript/transcript.md`
- `transcript/transcript.srt`
- `analysis/final-report.md`
- `result.json`

## Running As A Service

Start the API server on the GPU host:

```powershell
cd D:\Dev\youtube-audio-intel
$env:YAI_API_KEY = "change-this-secret"
yai serve --host 0.0.0.0 --port 8765 --max-workers 1 --queue-size 100 --cooldown-seconds 60 --cooldown-max-seconds 300 --cooldown-gpu-temp 65 --cooldown-gpu-utilization 20
```

This mode:

- accepts a larger backlog
- runs one workflow at a time
- enforces a post-job cooldown
- waits for the GPU to cool before starting the next queued job

Useful endpoints:

- `GET /health`
- `GET /queue`
- `GET /resources`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/files/{file_path}`

`GET /jobs/{job_id}` includes a `downloads` object with ready-to-use URLs for common artifacts, so clients usually do not need to build file URLs manually.

Check current queue and host load:

```powershell
Invoke-RestMethod http://GPU_MACHINE_IP:8765/queue
Invoke-RestMethod http://GPU_MACHINE_IP:8765/resources
```

## Remote Usage

From another machine, use the included client:

```powershell
$env:YAI_SERVER_URL = "http://GPU_MACHINE_IP:8765"
$env:YAI_API_KEY = "change-this-secret"
python D:\Dev\youtube-audio-intel\scripts\remote_client.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Project Layout

```text
src/youtube_audio_intel/
  api.py          FastAPI service and queue control
  cli.py          Typer CLI entrypoints
  resources.py    GPU and CPU telemetry helpers
  workflow.py     download -> transcode -> transcribe -> analyze
scripts/
  remote_client.py
  start-yai-server.ps1
docs/
  TECHNICAL_DESIGN.md
```

## Notes

- Use `--cookies-from-browser chrome` if YouTube requires a logged-in browser session.
- Use `--skip-analysis` if Ollama is unavailable and you only want transcripts.
- Use a smaller Whisper model such as `medium` for faster first-pass runs.
- GPU metrics come from `nvidia-smi`.
- CPU temperature may be unavailable on Windows; CPU load and memory remain available.
