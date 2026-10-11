"""Phase 4f: matching ads to what is said in a scene, and combining that with the frame score. NumPy only, except
embed_text (sentence-transformers, run in a child process like the CLIP text embeddings, BUGS.md #7)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scenesafe.common.results import REPO_ROOT

TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_WORDS = 50       # a scene's dialogue is scored in windows of 50 words, stride 25; the best window counts
STRIDE_WORDS = 25

EMBED = """
import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
texts = json.load(open(sys.argv[1]))
m = SentenceTransformer(sys.argv[3], device="mps")
np.save(sys.argv[2], m.encode(texts, batch_size=128, normalize_embeddings=True).astype(np.float32))
"""


def embed_text(texts: list[str], cache: Path) -> np.ndarray:
    """Unit-normalised sentence embeddings, one row per text. Recomputed whenever the texts change."""
    key = cache.with_suffix(".json")
    if not (cache.exists() and key.exists() and json.loads(key.read_text()) == texts):
        cache.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(texts))
        subprocess.run([sys.executable, "-c", EMBED, str(key), str(cache), TEXT_MODEL], check=True, cwd=REPO_ROOT)
    out = np.load(cache)
    assert len(out) == len(texts)
    return out


def chunks(text: str) -> list[str]:
    """Overlapping word windows of a scene's dialogue (the whole text if it is short)."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text] if words else []
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words) - CHUNK_WORDS + STRIDE_WORDS, STRIDE_WORDS)]


def scene_dialogue_scores(chunk_vecs: np.ndarray, chunk_scene: np.ndarray, n_scenes: int,
                          ad_vecs: np.ndarray) -> np.ndarray:
    """(scenes x ads): the best chunk's cosine similarity; NaN for scenes with no speech."""
    sims = chunk_vecs @ ad_vecs.T
    out = np.full((n_scenes, ad_vecs.shape[0]), np.nan, np.float32)
    for c, s in enumerate(chunk_scene):
        out[s] = np.fmax(out[s], sims[c])
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Per column (ad), over the rows that are not NaN."""
    return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)


def combined(frame: np.ndarray, dialogue: np.ndarray) -> np.ndarray:
    """Equal-weight sum of per-ad z-scores. A scene with no speech gets 0 for dialogue (the average)."""
    return zscore(frame) + np.nan_to_num(zscore(dialogue), nan=0.0)


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Scene indices with the k highest scores (NaN never picked), best first."""
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    return [int(s) for s in order if np.isfinite(scores[s])][:k]
