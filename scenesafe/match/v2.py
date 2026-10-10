"""Phase 4b matcher (EVAL-PROTOCOL.md, "Phase 4b amendment"). General rules only, the same for every ad:
template ensemble, score relative to ordinary footage, junk filter (pixel brightness/contrast), one cut-off.
NumPy only."""

import numpy as np

TEMPLATES = (
    "{}",
    "a photo of {}.",
    "a still from a film showing {}.",
    "a black and white film still of {}.",
    "a movie scene of {}.",
)
NEUTRAL = (
    "people talking in a room",
    "a close-up of a person's face",
    "a landscape",
    "a group of people standing",
    "an indoor scene",
)
# A second is junk (never an ad slot) if its frame is nearly black or nearly blank. Grey level in [0, 1] on a
# coarse grid (scenesafe.match.luma). Chosen by eye on practice-film frames; CLIP text prompts for "black screen"
# or "credits" were tried first and flagged 44% of practice seconds, mostly ordinary scenes.
JUNK_MAX_MEAN = 0.05
JUNK_MAX_STD = 0.03


def template_texts(briefs: list[str]) -> list[str]:
    """All template x brief texts, brief-major: brief i occupies rows i*len(TEMPLATES) .. +len(TEMPLATES)."""
    return [t.format(b) for b in briefs for t in TEMPLATES]


def ensemble(template_vecs: np.ndarray, n_briefs: int) -> np.ndarray:
    """Average each brief's template embeddings (unit vectors) and re-normalise."""
    v = template_vecs.reshape(n_briefs, len(TEMPLATES), -1).mean(axis=1)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def second_scores(emb: np.ndarray, ads: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Per second and ad: similarity to the ad minus mean similarity to neutral footage."""
    return emb @ ads.T - (emb @ neutral.T).mean(axis=1, keepdims=True)


def junk_mask(luma_mean: np.ndarray, luma_std: np.ndarray) -> np.ndarray:
    return (luma_mean < JUNK_MAX_MEAN) | (luma_std < JUNK_MAX_STD)


def scene_scores(scores: np.ndarray, is_junk: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Per scene and ad: the best score over the scene's non-junk seconds (-inf if every second is junk)."""
    masked = np.where(is_junk[:, None], -np.inf, scores)
    return np.maximum.reduceat(masked, first, axis=0)


def pick(scene_score: np.ndarray, k: int, tau: float | None) -> list[int]:
    """Top k scenes by score for one ad; with tau, only those scoring above it (possibly none)."""
    order = np.argsort(-scene_score, kind="stable")
    keep = [int(s) for s in order if np.isfinite(scene_score[s]) and (tau is None or scene_score[s] > tau)]
    return keep[:k]


def best_f1_threshold(scores: np.ndarray, fits: np.ndarray) -> tuple[float, float]:
    """The tau maximising F1 of (score > tau) against the labels; ties go to the lower tau. Candidates are the
    midpoints between sorted distinct scores, plus one below the minimum (return everything)."""
    s = np.unique(scores)
    candidates = np.r_[s[0] - 1e-6, (s[:-1] + s[1:]) / 2]
    best_tau, best_f1 = candidates[0], -1.0
    for tau in candidates:
        pred = scores > tau
        tp = int((pred & fits).sum())
        f1 = 2 * tp / (pred.sum() + fits.sum()) if (pred.sum() + fits.sum()) else 0.0
        if f1 > best_f1 + 1e-12:
            best_tau, best_f1 = float(tau), f1
    return best_tau, float(best_f1)
