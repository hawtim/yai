from __future__ import annotations

import shutil
import subprocess
from typing import Any

import psutil


def get_gpu_stats() -> list[dict[str, Any]]:
    """Return NVIDIA GPU stats from nvidia-smi when available."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    query = (
        "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,"
        "power.draw,power.limit"
    )
    proc = subprocess.run(
        [
            nvidia_smi,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    if proc.returncode != 0:
        return []

    stats: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        stats.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "temperature_c": _to_float(parts[2]),
                "utilization_percent": _to_float(parts[3]),
                "memory_used_mb": _to_float(parts[4]),
                "memory_total_mb": _to_float(parts[5]),
                "power_draw_w": _to_float(parts[6]),
                "power_limit_w": _to_float(parts[7]),
            }
        )
    return stats


def get_cpu_stats() -> dict[str, Any]:
    """Return portable CPU and memory stats; CPU temperature is best-effort."""
    memory = psutil.virtual_memory()
    payload: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_mb": round(memory.total / 1024 / 1024, 1),
        "memory_available_mb": round(memory.available / 1024 / 1024, 1),
        "memory_percent": memory.percent,
        "temperature_c": None,
    }
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        temps = {}
    for entries in temps.values():
        if entries:
            payload["temperature_c"] = entries[0].current
            break
    return payload


def get_system_stats() -> dict[str, Any]:
    return {"gpu": get_gpu_stats(), "cpu": get_cpu_stats()}


def primary_gpu_is_cool(max_temp_c: float, max_utilization_percent: float) -> bool:
    stats = get_gpu_stats()
    if not stats:
        return True
    gpu = stats[0]
    temp = gpu.get("temperature_c")
    utilization = gpu.get("utilization_percent")
    temp_ok = temp is None or temp <= max_temp_c
    utilization_ok = utilization is None or utilization <= max_utilization_percent
    return temp_ok and utilization_ok


def _to_float(value: str) -> float | None:
    if value in {"", "[Not Supported]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
