"""The corpus manifest and the fixed test windows used by Phase 1."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.json"
TESTSET = ROOT / "configs" / "phase1_testset.yaml"
VIDEO_DIR = ROOT / "data" / "corpus"


@dataclass(frozen=True)
class Window:
    id: str
    genre: str
    path: Path
    start_s: float
    duration_s: float


def titles() -> list[dict]:
    return json.loads(MANIFEST.read_text())["titles"]


def video_path(title_id: str) -> Path:
    return VIDEO_DIR / f"{title_id}.mp4"


def is_downloaded(title: dict) -> bool:
    """File present and complete (byte size matches the manifest). fetch_corpus.py also checks MD5."""
    path = video_path(title["id"])
    return path.exists() and path.stat().st_size == title["bytes"]


def test_windows() -> list[Window]:
    """Phase 1 test set, as listed in configs/phase1_testset.yaml."""
    cfg = yaml.safe_load(TESTSET.read_text())
    genre = {t["id"]: t["genre"] for t in titles()}
    return [Window(i, genre[i], video_path(i), float(cfg["start_s"]), float(cfg["duration_s"])) for i in cfg["titles"]]
