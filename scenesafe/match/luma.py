"""Per-second brightness and contrast of the sampled frames (uniform 1 fps, the same seconds as the embeddings).
Used by the Phase 4b junk filter: a nearly black or nearly blank frame is not an ad slot."""

import numpy as np

from scenesafe.common.results import REPO_ROOT
from scenesafe.corpus import video_path
from scenesafe.decode.sampling import Uniform, sample

LUMA_DIR = REPO_ROOT / "data" / "luma"


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of grey level in [0, 1], on a coarse grid (every 8th pixel)."""
    g = frame[::8, ::8].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    return float(g.mean()), float(g.std())


def compute(title_id: str) -> str:
    out = LUMA_DIR / f"{title_id}.npz"
    if not out.exists():
        stats = [(t, *frame_stats(f)) for t, f in sample(video_path(title_id), Uniform(1.0), "cpu")]
        a = np.array(stats)
        LUMA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(out, times=a[:, 0], mean=a[:, 1], std=a[:, 2])
    return title_id


def load(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated (mean, std) in corpus order; checked against the embedding times."""
    means, stds = [], []
    for tid in ids:
        d = np.load(LUMA_DIR / f"{tid}.npz")
        e = np.load(REPO_ROOT / "data" / "embeddings" / f"{tid}.npz")["times"]
        assert len(d["times"]) == len(e) and np.allclose(d["times"], e), f"luma/embedding seconds differ: {tid}"
        means.append(d["mean"])
        stds.append(d["std"])
    return np.concatenate(means), np.concatenate(stds)
