import subprocess
import sys

import numpy as np

from scenesafe.common.results import REPO_ROOT
from scenesafe.index.ranking import collapse, distinct, greedy_distinct
from scenesafe.index.units import Corpus, fixed_chunks, per_second, scene_starts, scenes, unit

D = 16


def blocks(lengths, seed=0):
    """Embeddings made of blocks: every second in a block is its own direction plus a little noise."""
    rng = np.random.default_rng(seed)
    centres = unit(rng.standard_normal((len(lengths), D)))
    return unit(np.concatenate([c + 0.05 * rng.standard_normal((n, D)) for c, n in zip(centres, lengths)]))


def corpus_of(*titles_lengths):
    embs = [blocks(lengths, seed=i) for i, lengths in enumerate(titles_lengths)]
    title = np.concatenate([np.full(len(e), i) for i, e in enumerate(embs)])
    time = np.concatenate([np.arange(len(e), dtype=float) for e in embs])
    return Corpus(np.concatenate(embs), title, time, [f"t{i}" for i in range(len(embs))])


def test_scene_starts_finds_block_boundaries():
    assert scene_starts(blocks([8, 5, 12]), threshold=0.8) == [0, 8, 13]


def test_scene_starts_respects_min_len():
    # a 1-second flash between two blocks cannot become its own scene
    starts = scene_starts(blocks([6, 1, 6]), threshold=0.8, min_len=3)
    assert all(b - a >= 3 for a, b in zip(starts, starts[1:]))


def test_units_cover_every_second_once_and_never_cross_titles():
    c = corpus_of([8, 5], [23])
    for u in (per_second(c), fixed_chunks(c, 10), scenes(c, 0.8)):
        assert u.count.sum() == len(c.emb)
        assert np.array_equal(u.of_second(), np.repeat(np.arange(len(u)), u.count))
        for f, n, t in zip(u.first, u.count, u.title):
            assert (c.title[f:f + n] == t).all()
        assert np.allclose(np.linalg.norm(u.vectors, axis=1), 1, atol=1e-5)
    assert list(fixed_chunks(c, 10).count) == [10, 3, 10, 10, 3]
    assert list(scenes(c, 0.8).count) == [8, 5, 23]


def test_collapse_keeps_best_hit_per_owner():
    rows = np.array([4, 5, 0, 6, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    owner = np.array([0, 0, 1, 1, 2, 2, 3])   # rows 4,5 -> owner 2; row 0 -> 0; row 6 -> 3
    owners, best, best_rows = collapse(rows, scores, owner, k=3)
    assert list(owners) == [2, 0, 3]
    assert list(best) == [0.9, 0.7, 0.6]
    assert list(best_rows) == [4, 0, 6]


def test_distinct_and_greedy_agree_on_gap():
    title = np.array([0, 0, 0, 1])
    start = np.array([100.0, 120.0, 200.0, 100.0])
    keep = distinct(title, start, start + 1, gap_s=30)
    assert list(keep) == [True, False, True, True]
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert list(greedy_distinct(scores, title, start, k=3, gap_s=30)) == [0, 2, 3]


FAISS_CHECK = """
import numpy as np
from scenesafe.index.search import IndexSpec, build, set_nprobe
from scenesafe.index.units import unit
x = unit(np.random.default_rng(1).standard_normal((2000, 16)))
flat = build(IndexSpec("flat"), x)
_, ids = flat.search(x[:50], 10)
assert (ids[:, 0] == np.arange(50)).all(), "flat search must return each vector first for itself"
ivf = build(IndexSpec("ivf", nlist=16), x)
set_nprobe(ivf, 16)
_, ivf_ids = ivf.search(x[:50], 10)
assert (ivf_ids == ids).all(), "IVF probing every group must equal exact search"
"""


def test_flat_returns_self_first_and_ivf_full_probe_is_exact():
    # in a child process: other test modules load torch, and torch + faiss in one process abort (BUGS.md #7)
    out = subprocess.run([sys.executable, "-c", FAISS_CHECK], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
