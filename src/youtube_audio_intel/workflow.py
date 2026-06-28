from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass
class WorkflowOptions:
    url: str
    output_root: Path = DEFAULT_RUNS_DIR
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    skip_analysis: bool = False
    cookies_from_browser: str | None = None
    keep_raw: bool = True
    chunk_chars: int = 9000


@dataclass
class WorkflowResult:
    run_id: str
    run_dir: Path
    metadata: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def slugify(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE).strip()
    value = re.sub(r"[\s.]+", "-", value)
    return value[:max_len].strip("-") or "youtube-audio"


def find_ffmpeg() -> str:
    env_path = os.environ.get("YAI_FFMPEG")
    if env_path and Path(env_path).exists():
        return env_path

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - defensive message path
        raise RuntimeError(
            "FFmpeg was not found in PATH and imageio-ffmpeg fallback failed. "
            "Install FFmpeg or reinstall imageio-ffmpeg."
        ) from exc


def configure_nvidia_dll_paths() -> list[str]:
    """Register pip-installed NVIDIA CUDA DLL folders on Windows."""
    if os.name != "nt":
        return []

    import site

    added: list[str] = []
    candidates: list[Path] = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_dir = Path(site_dir) / "nvidia"
        candidates.extend(nvidia_dir.glob("*/bin"))

    existing_path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        if not candidate.exists():
            continue
        value = str(candidate)
        if value not in existing_path_parts:
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
            existing_path_parts.insert(0, value)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(value))
            except OSError:
                pass
        added.append(value)
    return added


def run_command(args: list[str]) -> None:
    proc = subprocess.run(args, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(args)
            + "\nSTDOUT:\n"
            + proc.stdout
            + "\nSTDERR:\n"
            + proc.stderr
        )


def prepare_run_dir(output_root: Path, title: str | None = None) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    prefix = slugify(title or "pending")
    run_id = f"{timestamp}-{short_id}"
    run_dir = output_root / f"{run_id}-{prefix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def download_audio(opts: WorkflowOptions, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "writeinfojson": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["zh-Hans", "zh-Hant", "zh", "en"],
    }
    if opts.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (opts.cookies_from_browser,)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(opts.url, download=True)
        filename = Path(ydl.prepare_filename(info))

    if not filename.exists():
        candidates = sorted(raw_dir.glob(f"{info.get('id', '*')}.*"))
        media_candidates = [
            p for p in candidates if p.suffix.lower() not in {".json", ".vtt", ".srt"}
        ]
        if not media_candidates:
            raise FileNotFoundError(f"Downloaded media file was not found under {raw_dir}")
        filename = media_candidates[0]

    metadata = {
        "id": info.get("id"),
        "title": info.get("title"),
        "webpage_url": info.get("webpage_url") or opts.url,
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "description": info.get("description"),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return filename, metadata


def extract_wav(input_file: Path, run_dir: Path) -> Path:
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "audio-16k-mono.wav"
    ffmpeg = find_ffmpeg()
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    return wav_path


def format_timestamp(seconds: float, sep: str = ",") -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def write_srt(segments: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(
            f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}"
        )
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_audio(opts: WorkflowOptions, wav_path: Path, run_dir: Path) -> dict[str, Any]:
    transcript_dir = run_dir / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    configure_nvidia_dll_paths()
    model = WhisperModel(opts.model, device=opts.device, compute_type=opts.compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=opts.language,
        vad_filter=True,
        beam_size=5,
    )
    segments = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments_iter
    ]
    payload = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "elapsed_seconds": round(time.time() - start, 2),
        "segments": segments,
    }
    (transcript_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plain = "\n".join(
        f"[{format_timestamp(seg['start'], '.')} -> {format_timestamp(seg['end'], '.')}] {seg['text']}"
        for seg in segments
    )
    (transcript_dir / "transcript.md").write_text(plain, encoding="utf-8")
    write_srt(segments, transcript_dir / "transcript.srt")
    return payload


def chunk_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [line for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current and current_len + len(para) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text[:max_chars]]


def ollama_chat(
    base_url: str, model: str, messages: list[dict[str, str]], timeout: int = 600
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    response = requests.post(
        url,
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def analyze_transcript(
    opts: WorkflowOptions,
    transcript: dict[str, Any],
    metadata: dict[str, Any],
    run_dir: Path,
) -> str:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    transcript_text = "\n".join(
        f"[{format_timestamp(seg['start'], '.')}] {seg['text']}"
        for seg in transcript["segments"]
    )
    chunks = chunk_text(transcript_text, opts.chunk_chars)

    system = (
        "你是一个严谨的视频音频内容分析助手。只根据给定转写内容分析；"
        "保留时间戳证据；不要编造没有出现在转写里的信息；"
        "不要输出思考过程、<think> 标签或隐藏推理。"
    )
    chunk_notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""
视频标题：{metadata.get('title')}
频道/上传者：{metadata.get('channel') or metadata.get('uploader')}
分块：{index}/{len(chunks)}

请从这段带时间戳的转写中提取详细信息，输出中文 Markdown：
- 本段核心观点
- 重要事实、数字、名称、工具、链接或术语
- 可引用的关键原话/表达，保留时间戳
- 行动项或决策建议，如果没有就写“无”
- 不确定或听不清的地方

转写：
{chunk}
""".strip()
        note = ollama_chat(
            opts.ollama_url,
            opts.ollama_model,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        chunk_notes.append(f"## Chunk {index}\n\n{note.strip()}")
        (analysis_dir / f"chunk-{index:03d}.md").write_text(note, encoding="utf-8")

    final_prompt = f"""
视频元数据：
{json.dumps(metadata, ensure_ascii=False, indent=2)}

以下是逐段分析结果。请生成一份中文详细报告，结构如下：
1. 一句话结论
2. 执行摘要
3. 章节大纲，尽量带时间戳
4. 关键观点与证据
5. 实体清单：人物、公司、产品、工具、地点、数据
6. 可执行行动项
7. 风险、争议点、不确定信息
8. 值得回看的时间戳

逐段分析：
{chr(10).join(chunk_notes)}
""".strip()
    report = ollama_chat(
        opts.ollama_url,
        opts.ollama_model,
        [{"role": "system", "content": system}, {"role": "user", "content": final_prompt}],
        timeout=900,
    )
    report_path = analysis_dir / "final-report.md"
    report_path.write_text(report, encoding="utf-8")
    (analysis_dir / "chunk-notes.md").write_text(
        "\n\n".join(chunk_notes), encoding="utf-8"
    )
    return report


def collect_files(run_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in run_dir.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(run_dir)).replace("\\", "/")] = str(path)
    return files


def run_workflow(opts: WorkflowOptions) -> WorkflowResult:
    opts.output_root.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = prepare_run_dir(opts.output_root)
    warnings: list[str] = []
    metadata: dict[str, Any] = {"url": opts.url}

    try:
        raw_file, metadata = download_audio(opts, run_dir)
        new_name = run_dir.name.replace("pending", slugify(metadata.get("title") or "video"))
        desired = run_dir.with_name(new_name)
        if desired != run_dir and not desired.exists():
            old_run_dir = run_dir
            run_dir.rename(desired)
            run_dir = desired
            raw_file = run_dir / raw_file.relative_to(old_run_dir)

        wav_path = extract_wav(raw_file, run_dir)
        transcript = transcribe_audio(opts, wav_path, run_dir)

        if not opts.skip_analysis:
            try:
                analyze_transcript(opts, transcript, metadata, run_dir)
            except Exception as exc:
                warnings.append(f"Ollama analysis skipped or failed: {exc}")
                (run_dir / "analysis-error.txt").write_text(str(exc), encoding="utf-8")

        if not opts.keep_raw:
            shutil.rmtree(run_dir / "raw", ignore_errors=True)

        result_path = run_dir / "result.json"
        result = WorkflowResult(
            run_id=run_id,
            run_dir=run_dir,
            metadata=metadata,
            warnings=warnings,
        )
        result.files = collect_files(run_dir)
        result.files["result.json"] = str(result_path)
        result_path.write_text(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "run_dir": str(result.run_dir),
                    "metadata": result.metadata,
                    "files": result.files,
                    "warnings": result.warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
    except Exception:
        (run_dir / "failed.txt").write_text(
            f"Workflow failed at {datetime.now().isoformat()}", encoding="utf-8"
        )
        raise
