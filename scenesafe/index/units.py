"""What one searchable entry ("unit") is: one second, a fixed 10-second chunk, or a detected scene.

The corpus embeddings are one per second (uniform 1 fps, Phase 2). Units are built from them without touching
the video again. Every unit is a contiguous run of seconds from one title, in corpus order, so per-second values
can be reduced per unit with np.*.reduceat over `first`.
"""

from dataclasses import dataclass

import numpy as np

from scenesafe.common.results import REPO_ROOT
from scenesafe.corpus import titles

EMB_DIR = REPO_ROOT / "data" / "embeddings"


@dataclass
class Corpus:
    emb: np.ndarray       # (N, 512) float32, unit-normalised, one row per sampled second
    title: np.ndarray     # (N,) index into `ids`
    time: np.ndarray      # (N,) seconds from the start of the title
    ids: list[str]


@dataclass
class Units:
    name: str
    vectors: np.ndarray   # (U, 512) float32, unit-normalised
    title: np.ndarray     # (U,) index into Corpus.ids
    start: np.ndarray     # (U,) seconds
    end: np.ndarray       # (U,) seconds (exclusive)
    first: np.ndarray     # (U,) row of the first second in the corpus
    count: np.ndarray     # (U,) number of seconds

    def __len__(self):
        return len(self.vectors)

    def of_second(self) -> np.ndarray:
        """For every corpus row, the unit that contains it."""
        return np.repeat(np.arange(len(self)), self.count)


def load_corpus(ids: list[str] | None = None) -> Corpus:
    ids = ids or [t["id"] for t in titles()]
    embs, title, time = [], [], []
    for i, tid in enumerate(ids):
        d = np.load(EMB_DIR / f"{tid}.npz")
        embs.append(d["emb"].astype(np.float32))
        time.append(d["times"].astype(np.float64))
        title.append(np.full(len(d["times"]), i))
    return Corpus(unit(np.concatenate(embs)), np.concatenate(title), np.concatenate(time), ids)


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def from_runs(name: str, corpus: Corpus, first: np.ndarray) -> Units:
    """Units from the corpus rows where each run starts (runs never cross titles)."""
    first = np.asarray(first, dtype=np.int64)
    count = np.diff(np.append(first, len(corpus.emb)))
    last = first + count - 1
    vectors = unit(np.add.reduceat(corpus.emb, first, axis=0))
    return Units(name, vectors, corpus.title[first], corpus.time[first], corpus.time[last] + 1.0, first, count)


def title_starts(corpus: Corpus) -> np.ndarray:
    return np.flatnonzero(np.r_[True, corpus.title[1:] != corpus.title[:-1]])


def per_second(corpus: Corpus) -> Units:
    return from_runs("second", corpus, np.arange(len(corpus.emb)))


def fixed_chunks(corpus: Corpus, seconds: int = 10) -> Units:
    starts = title_starts(corpus)
    ends = np.append(starts[1:], len(corpus.emb))
    first = np.concatenate([np.arange(s, e, seconds) for s, e in zip(starts, ends)])
    return from_runs(f"fixed{seconds}", corpus, first)


def scene_starts(emb: np.ndarray, threshold: float, min_len: int = 3) -> list[int]:
    """Cut where a second's embedding drops below `threshold` cosine to the running mean of the current scene.

    Comparing with the scene's mean (not just the previous second) keeps one odd frame from pulling the reference
    along. A scene must reach `min_len` seconds before it can end, so a single flash doesn't become a scene.
    """
    starts, total = [0], emb[0].astype(np.float64).copy()
    for t in range(1, len(emb)):
        if t - starts[-1] >= min_len and emb[t] @ (total / np.linalg.norm(total)) < threshold:
            starts.append(t)
            total = emb[t].astype(np.float64).copy()
        else:
            total += emb[t]
    return starts


def scenes(corpus: Corpus, threshold: float, min_len: int = 3) -> Units:
    starts = title_starts(corpus)
    ends = np.append(starts[1:], len(corpus.emb))
    first = np.concatenate([s + np.array(scene_starts(corpus.emb[s:e], threshold, min_len))
                            for s, e in zip(starts, ends)])
    return from_runs(f"scene@{threshold:.2f}", corpus, first)
