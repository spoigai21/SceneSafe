"""Results files: every measured run writes one JSON file with enough context to trace it."""

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def hardware() -> dict:
    info = {"machine": platform.machine(), "system": platform.system(), "cpu_count": os.cpu_count()}
    if platform.system() == "Darwin":
        def sysctl(key):
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        info["cpu"] = sysctl("machdep.cpu.brand_string")
        info["ram_gb"] = int(sysctl("hw.memsize")) / 2**30
    else:
        info["cpu"] = platform.processor()
    return info


def load_avg() -> list[float]:
    return list(os.getloadavg())


def write_result(phase: int, name: str, lane: str, seed: int, config: dict, data: dict) -> Path:
    """Write results/phase<N>/<name>_<lane>_seed<seed>.json and return its path."""
    out_dir = REPO_ROOT / "results" / f"phase{phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{lane}_seed{seed}.json"
    record = {
        "phase": phase,
        "name": name,
        "lane": lane,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "hardware": hardware(),
        "config": config,
        **data,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
