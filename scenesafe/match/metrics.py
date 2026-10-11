"""Scoring (EVAL-PROTOCOL.md, "Metrics"): precision@k, nDCG@k with binary or graded gains, bootstrap over ads, and
agreement between two labellers."""

import numpy as np


def precision_at_k(relevant: list[int], k: int) -> float:
    """relevant: 0/1 per returned item, in rank order. Missing items count as not relevant."""
    return sum(relevant[:k]) / k


def ndcg_at_k(relevant: list[int], n_relevant_known: int, k: int) -> float:
    """Binary gain, log2 discount. The ideal ranking holds every item known to be relevant (at most k)."""
    ideal = min(n_relevant_known, k)
    if ideal == 0:
        return 0.0
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevant[:k]))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal))
    return float(dcg / idcg)


def graded_ndcg_at_k(gains: list[int], known_gains: list[int], k: int) -> float:
    """Graded gain (e.g. 0/1/2), log2 discount. The ideal ranking is the k largest gains known for the ad."""
    ideal = sorted(known_gains, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k])) / idcg)


def bootstrap(per_ad: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean over ads. per_ad may be a paired difference."""
    rng = np.random.default_rng(seed)
    means = per_ad[rng.integers(len(per_ad), size=(n, len(per_ad)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def agreement(a: list[int], b: list[int]) -> dict:
    """Percent agreement and Cohen's kappa for two binary label lists over the same items."""
    a, b = np.asarray(a), np.asarray(b)
    observed = float((a == b).mean())
    p_a, p_b = a.mean(), b.mean()
    expected = float(p_a * p_b + (1 - p_a) * (1 - p_b))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": int(len(a)), "agreement": observed, "cohen_kappa": float(kappa),
            "fits_rate_a": float(p_a), "fits_rate_b": float(p_b)}
