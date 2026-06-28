from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .resources import get_system_stats, primary_gpu_is_cool
from .workflow import DEFAULT_RUNS_DIR, WorkflowOptions, run_workflow


app = FastAPI(title="YouTube Audio Intel API", version="0.1.0")
MAX_WORKERS = max(1, int(os.environ.get("YAI_MAX_WORKERS", "1")))
MAX_QUEUE_SIZE = max(1, int(os.environ.get("YAI_QUEUE_MAXSIZE", "100")))
COOLDOWN_SECONDS = max(0, int(os.environ.get("YAI_COOLDOWN_SECONDS", "60")))
COOLDOWN_MAX_SECONDS = max(
    COOLDOWN_SECONDS, int(os.environ.get("YAI_COOLDOWN_MAX_SECONDS", "300"))
)
COOLDOWN_GPU_MAX_TEMP_C = float(os.environ.get("YAI_COOLDOWN_GPU_MAX_TEMP_C", "65"))
COOLDOWN_GPU_MAX_UTILIZATION = float(
    os.environ.get("YAI_COOLDOWN_GPU_MAX_UTILIZATION", "20")
)
job_queue: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
worker_states: dict[str, dict] = {}
worker_states_lock = threading.Lock()
workers_started = False


class JobRequest(BaseModel):
    url: str
    output_root: str | None = None
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    skip_analysis: bool = False
    cookies_from_browser: str | None = None
    keep_raw: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    result: dict | None = None
    error: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None


class QueueStatus(BaseModel):
    max_workers: int
    max_queue_size: int
    cooldown_seconds: int
    cooldown_max_seconds: int
    cooldown_gpu_max_temp_c: float
    cooldown_gpu_max_utilization: float
    queued: int
    running: int
    cooling_workers: int
    succeeded: int
    failed: int
    total_jobs: int
    workers: dict[str, dict] = Field(default_factory=dict)


def require_api_key(x_api_key: str | None) -> None:
    expected = os.environ.get("YAI_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def queue_position(job_id: str) -> int | None:
    with job_queue.mutex:
        queued_ids = list(job_queue.queue)
    try:
        return queued_ids.index(job_id) + 1
    except ValueError:
        return None


def serialize_job(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        payload = dict(jobs[job_id])
    if payload.get("status") == "queued":
        payload["queue_position"] = queue_position(job_id)
    else:
        payload["queue_position"] = None
    return payload


def execute_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = now_iso()
        request = job["request"]
    try:
        opts = WorkflowOptions(
            url=request.url,
            output_root=Path(request.output_root) if request.output_root else DEFAULT_RUNS_DIR,
            model=request.whisper_model,
            device=request.device,
            compute_type=request.compute_type,
            language=request.language,
            ollama_url=request.ollama_url,
            ollama_model=request.ollama_model,
            skip_analysis=request.skip_analysis,
            cookies_from_browser=request.cookies_from_browser,
            keep_raw=request.keep_raw,
        )
        result = run_workflow(opts)
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "succeeded",
                    "finished_at": now_iso(),
                    "result": {
                        "run_id": result.run_id,
                        "run_dir": str(result.run_dir),
                        "metadata": result.metadata,
                        "warnings": result.warnings,
                    },
                    "files": result.files,
                }
            )
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "finished_at": now_iso(),
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )


def set_worker_state(name: str, state: str, job_id: str | None = None) -> None:
    with worker_states_lock:
        worker_states[name] = {
            "state": state,
            "job_id": job_id,
            "updated_at": now_iso(),
        }


def cooldown_worker(name: str) -> None:
    if COOLDOWN_MAX_SECONDS <= 0:
        return
    set_worker_state(name, "cooling")
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        min_elapsed = elapsed >= COOLDOWN_SECONDS
        max_elapsed = elapsed >= COOLDOWN_MAX_SECONDS
        cool_enough = primary_gpu_is_cool(
            COOLDOWN_GPU_MAX_TEMP_C, COOLDOWN_GPU_MAX_UTILIZATION
        )
        if (min_elapsed and cool_enough) or max_elapsed:
            return
        next_check_seconds = 5.0
        if not min_elapsed:
            next_check_seconds = min(next_check_seconds, COOLDOWN_SECONDS - elapsed)
        if not max_elapsed:
            next_check_seconds = min(next_check_seconds, COOLDOWN_MAX_SECONDS - elapsed)
        time.sleep(max(0.25, next_check_seconds))


def worker_loop() -> None:
    name = threading.current_thread().name
    set_worker_state(name, "idle")
    while True:
        job_id = job_queue.get()
        try:
            set_worker_state(name, "running", job_id)
            execute_job(job_id)
        finally:
            job_queue.task_done()
            cooldown_worker(name)
            set_worker_state(name, "idle")


def start_workers() -> None:
    global workers_started
    if workers_started:
        return
    workers_started = True
    for index in range(MAX_WORKERS):
        thread = threading.Thread(
            target=worker_loop,
            name=f"yai-worker-{index + 1}",
            daemon=True,
        )
        thread.start()


@app.on_event("startup")
def on_startup() -> None:
    start_workers()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/queue", response_model=QueueStatus)
def get_queue(x_api_key: str | None = Header(None)) -> QueueStatus:
    require_api_key(x_api_key)
    with jobs_lock:
        statuses = [job["status"] for job in jobs.values()]
    with worker_states_lock:
        workers = dict(worker_states)
    return QueueStatus(
        max_workers=MAX_WORKERS,
        max_queue_size=MAX_QUEUE_SIZE,
        cooldown_seconds=COOLDOWN_SECONDS,
        cooldown_max_seconds=COOLDOWN_MAX_SECONDS,
        cooldown_gpu_max_temp_c=COOLDOWN_GPU_MAX_TEMP_C,
        cooldown_gpu_max_utilization=COOLDOWN_GPU_MAX_UTILIZATION,
        queued=statuses.count("queued"),
        running=statuses.count("running"),
        cooling_workers=sum(1 for item in workers.values() if item.get("state") == "cooling"),
        succeeded=statuses.count("succeeded"),
        failed=statuses.count("failed"),
        total_jobs=len(statuses),
        workers=workers,
    )


@app.get("/resources")
def get_resources(x_api_key: str | None = Header(None)) -> dict:
    require_api_key(x_api_key)
    return get_system_stats()


@app.post("/jobs", response_model=JobStatus)
def submit_job(request: JobRequest, x_api_key: str | None = Header(None)) -> JobStatus:
    require_api_key(x_api_key)
    start_workers()
    if job_queue.full():
        raise HTTPException(
            status_code=429,
            detail=(
                "Job queue is full. Retry later or increase YAI_QUEUE_MAXSIZE "
                "if this machine can tolerate more backlog."
            ),
        )
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "result": None,
            "error": None,
            "files": {},
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request,
        }
    try:
        job_queue.put_nowait(job_id)
    except queue.Full:
        with jobs_lock:
            jobs.pop(job_id, None)
        raise HTTPException(status_code=429, detail="Job queue is full")
    return JobStatus(**serialize_job(job_id))


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, x_api_key: str | None = Header(None)) -> JobStatus:
    require_api_key(x_api_key)
    return JobStatus(**serialize_job(job_id))


@app.get("/jobs/{job_id}/files/{file_path:path}")
def get_file(job_id: str, file_path: str, x_api_key: str | None = Header(None)) -> FileResponse:
    require_api_key(x_api_key)
    payload = serialize_job(job_id)
    files = payload.get("files") or {}
    if file_path not in files:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(files[file_path])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File disappeared")
    return FileResponse(path)
