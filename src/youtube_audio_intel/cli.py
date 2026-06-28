from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import ctranslate2
import typer
from rich.console import Console
from rich.table import Table

from .workflow import (
    DEFAULT_RUNS_DIR,
    WorkflowOptions,
    configure_nvidia_dll_paths,
    find_ffmpeg,
    run_workflow,
)


app = typer.Typer(help="YouTube audio intelligence CLI.")
console = Console()


@app.command()
def check() -> None:
    """Check local dependencies and GPU transcription support."""
    table = Table(title="youtube-audio-intel environment")
    table.add_column("Item")
    table.add_column("Status")

    table.add_row("Python package", "ok")
    table.add_row("yt-dlp CLI", shutil.which("yt-dlp") or "use Python module fallback")
    table.add_row("Deno JS runtime", shutil.which("deno") or "not found in current PATH")
    try:
        table.add_row("FFmpeg", find_ffmpeg())
    except Exception as exc:
        table.add_row("FFmpeg", f"missing: {exc}")

    try:
        cuda_types = ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda")))
    except Exception as exc:
        cuda_types = f"unavailable: {exc}"
    table.add_row("CTranslate2 CUDA", cuda_types)
    dll_dirs = configure_nvidia_dll_paths()
    table.add_row("NVIDIA DLL dirs", f"{len(dll_dirs)} registered")
    table.add_row("Default runs dir", str(DEFAULT_RUNS_DIR))
    console.print(table)


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube video URL."),
    output_root: Path = typer.Option(DEFAULT_RUNS_DIR, "--output-root", "-o"),
    whisper_model: str = typer.Option("large-v3", "--whisper-model"),
    device: str = typer.Option("cuda", "--device"),
    compute_type: str = typer.Option("float16", "--compute-type"),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", "--ollama-url"),
    ollama_model: str = typer.Option("qwen3:8b", "--ollama-model"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis"),
    cookies_from_browser: Optional[str] = typer.Option(
        None, "--cookies-from-browser", help="Example: chrome, edge, firefox"
    ),
    no_keep_raw: bool = typer.Option(False, "--no-keep-raw"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Run download, FFmpeg extraction, Whisper transcription, and Ollama analysis."""
    opts = WorkflowOptions(
        url=url,
        output_root=output_root,
        model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        skip_analysis=skip_analysis,
        cookies_from_browser=cookies_from_browser,
        keep_raw=not no_keep_raw,
        use_cache=not no_cache,
    )
    result = run_workflow(opts)
    console.print(f"[green]Done[/green] run_id={result.run_id}")
    console.print(f"Run dir: {result.run_dir}")
    console.print(f"Cache hit: {result.cache_hit}")
    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(json.dumps(result.files, ensure_ascii=False, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    max_workers: int = typer.Option(1, "--max-workers", help="Concurrent workflow jobs."),
    queue_size: int = typer.Option(100, "--queue-size", help="Maximum queued jobs before HTTP 429."),
    cooldown_seconds: int = typer.Option(
        60, "--cooldown-seconds", help="Minimum rest time after each completed job."
    ),
    cooldown_max_seconds: int = typer.Option(
        300, "--cooldown-max-seconds", help="Maximum rest time while waiting for GPU to cool."
    ),
    cooldown_gpu_temp: float = typer.Option(
        65.0, "--cooldown-gpu-temp", help="Target GPU temperature in Celsius before next job."
    ),
    cooldown_gpu_utilization: float = typer.Option(
        20.0, "--cooldown-gpu-utilization", help="Target GPU utilization percent before next job."
    ),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the remote HTTP API server."""
    import uvicorn

    os.environ["YAI_MAX_WORKERS"] = str(max_workers)
    os.environ["YAI_QUEUE_MAXSIZE"] = str(queue_size)
    os.environ["YAI_COOLDOWN_SECONDS"] = str(cooldown_seconds)
    os.environ["YAI_COOLDOWN_MAX_SECONDS"] = str(cooldown_max_seconds)
    os.environ["YAI_COOLDOWN_GPU_MAX_TEMP_C"] = str(cooldown_gpu_temp)
    os.environ["YAI_COOLDOWN_GPU_MAX_UTILIZATION"] = str(cooldown_gpu_utilization)
    uvicorn.run(
        "youtube_audio_intel.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
