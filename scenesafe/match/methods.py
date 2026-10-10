"""The three ways of picking scenes for an ad (EVAL-PROTOCOL.md, "Methods"). NumPy only; the CLIP search hits
come in from FAISS, which must not share a process with torch (BUGS.md #7)."""

import numpy as np

from scenesafe.index.ranking import collapse

K = 10


def clip_scenes(hit_rows: np.ndarray, hit_scores: np.ndarray, scene_of_second: np.ndarray, k: int = K) -> list[int]:
    """Best-first per-second hits -> the top k scenes, each once at its best second (the Phase 3 decision)."""
    scenes, _, _ = collapse(hit_rows, hit_scores, scene_of_second, k)
    return scenes.tolist()


def sampled_scenes(eligible_seconds: np.ndarray, scene_of_second: np.ndarray, rng: np.random.Generator,
                   k: int = K) -> list[int]:
    """Random moments (uniform over the eligible seconds) -> the scenes they fall in, skipping repeats, in draw
    order. Used by both the Genre and the Random baseline."""
    available = len(np.unique(scene_of_second[eligible_seconds]))
    picked: list[int] = []
    while len(picked) < min(k, available):
        s = int(scene_of_second[eligible_seconds[rng.integers(len(eligible_seconds))]])
        if s not in picked:
            picked.append(s)
    return picked
