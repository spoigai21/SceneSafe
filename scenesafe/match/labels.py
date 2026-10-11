"""Which pooled pairs the second labeller judges (EVAL-PROTOCOL.md, "Labels" and the Phase 4b amendment)."""

import numpy as np

AUTHOR_SAMPLE = 100


def author_sample(n_pairs: int, size: int = AUTHOR_SAMPLE, seed: int = 0) -> list[int]:
    """Indices, in pool order, of the pairs the second labeller judges. Phase 4: 100 pairs, default_rng(0);
    Phase 4b and 4c: 50 of the new pairs, default_rng(1) and (2)."""
    return sorted(np.random.default_rng(seed).choice(n_pairs, min(size, n_pairs), replace=False).tolist())
