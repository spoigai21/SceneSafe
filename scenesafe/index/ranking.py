"""Turning raw hits into results: collapsing seconds into scenes, and counting distinct moments. NumPy only."""

import numpy as np


def collapse(rows: np.ndarray, scores: np.ndarray, owner: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From hits sorted best-first, keep the best hit per owner (e.g. per scene) until k owners are found.

    Returns (owners, their best score, the row of that best hit). May return fewer than k if the hits run out.
    """
    seen, out_owner, out_score, out_row = set(), [], [], []
    for r, s in zip(rows, scores):
        if r < 0:
            break
        o = int(owner[r])
        if o not in seen:
            seen.add(o)
            out_owner.append(o)
            out_score.append(s)
            out_row.append(r)
            if len(out_owner) == k:
                break
    return np.array(out_owner, dtype=np.int64), np.array(out_score), np.array(out_row, dtype=np.int64)


def distinct(title: np.ndarray, start: np.ndarray, end: np.ndarray, gap_s: float) -> np.ndarray:
    """Mask over a ranked result list: True if the result is a new moment, i.e. not within gap_s seconds of a
    higher-ranked result from the same title."""
    keep = np.ones(len(title), dtype=bool)
    for i in range(len(title)):
        for j in range(i):
            if title[j] == title[i] and start[i] < end[j] + gap_s and start[j] < end[i] + gap_s:
                keep[i] = False
                break
    return keep


def greedy_distinct(scores: np.ndarray, title: np.ndarray, time: np.ndarray, k: int, gap_s: float) -> np.ndarray:
    """The k highest-scoring seconds that are pairwise more than gap_s apart (within a title). Used as the ceiling:
    the best k separate moments that per-second scores can point at."""
    picked = []
    for r in np.argsort(-scores, kind="stable"):
        if all(title[p] != title[r] or abs(time[p] - time[r]) >= 1.0 + gap_s for p in picked):
            picked.append(r)
            if len(picked) == k:
                break
    return np.array(picked, dtype=np.int64)
