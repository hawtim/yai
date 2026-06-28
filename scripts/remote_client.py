from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request_json(method: str, url: str, api_key: str | None, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def download_file(url: str, target: Path, api_key: str | None) -> None:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as response:
        target.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote client for YouTube Audio Intel API.")
    parser.add_argument("url", help="YouTube URL to process.")
    parser.add_argument("--server", default=os.environ.get("YAI_SERVER_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--api-key", default=os.environ.get("YAI_API_KEY"))
    parser.add_argument("--output-dir", default="yai-remote-output")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()

    server = args.server.rstrip("/")
    payload = {
        "url": args.url,
        "ollama_model": args.ollama_model,
        "whisper_model": args.whisper_model,
        "skip_analysis": args.skip_analysis,
    }
    job = request_json("POST", f"{server}/jobs", args.api_key, payload)
    job_id = job["job_id"]
    print(f"submitted job_id={job_id}")

    while True:
        job = request_json("GET", f"{server}/jobs/{job_id}", args.api_key)
        print(f"status={job['status']}")
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(args.poll_seconds)

    output_dir = Path(args.output_dir) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    if job["status"] == "failed":
        print(job.get("error", "job failed"), file=sys.stderr)
        return 1

    preferred = [
        "analysis/final-report.md",
        "transcript/transcript.md",
        "transcript/transcript.json",
        "transcript/transcript.srt",
        "metadata.json",
        "result.json",
    ]
    files = job.get("files") or {}
    for name in preferred:
        if name in files:
            encoded = urllib.parse.quote(name, safe="/")
            download_file(f"{server}/jobs/{job_id}/files/{encoded}", output_dir / name, args.api_key)
            print(f"downloaded {name}")
    print(f"output_dir={output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
