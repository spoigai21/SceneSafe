import numpy as np
import pytest

from scenesafe.match import v2


def test_templates_and_ensemble_shape():
    texts = v2.template_texts(["a dog", "a cat"])
    assert len(texts) == 2 * len(v2.TEMPLATES) and texts[0] == "a dog" and texts[len(v2.TEMPLATES)] == "a cat"
    vecs = np.random.default_rng(0).standard_normal((len(texts), 8)).astype(np.float32)
    out = v2.ensemble(vecs, 2)
    assert out.shape == (2, 8) and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-6)


def test_scores_are_relative_to_neutral():
    emb = np.eye(3, dtype=np.float32)
    ads = np.array([[1, 0, 0]], np.float32)
    neutral = np.array([[0, 1, 0], [0, 0, 1]], np.float32)
    s = v2.second_scores(emb, ads, neutral)
    assert s[:, 0].tolist() == pytest.approx([1.0, -0.5, -0.5])


def test_junk_and_scene_scores_skip_junk_seconds():
    is_junk = v2.junk_mask(np.array([0.01, 0.3, 0.3, 0.3]), np.array([0.2, 0.2, 0.01, 0.2]))
    assert is_junk.tolist() == [True, False, True, False]
    scores = np.array([[9.0], [1.0], [8.0], [2.0]])
    ss = v2.scene_scores(scores, is_junk, np.array([0, 2]))
    assert ss[:, 0].tolist() == [1.0, 2.0]                     # the junk seconds' high scores are ignored
    all_junk = v2.scene_scores(scores, np.ones(4, bool), np.array([0, 2]))
    assert np.isneginf(all_junk).all()


def test_pick_respects_cutoff_and_can_abstain():
    s = np.array([0.3, -np.inf, 0.1, 0.2])
    assert v2.pick(s, 10, None) == [0, 3, 2]
    assert v2.pick(s, 10, 0.15) == [0, 3]
    assert v2.pick(s, 10, 0.5) == []


def test_best_f1_threshold():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    tau, f1 = v2.best_f1_threshold(scores, np.array([False, False, True, True]))
    assert 0.2 < tau < 0.3 and f1 == pytest.approx(1.0)
    tau, _ = v2.best_f1_threshold(scores, np.ones(4, bool))
    assert tau < 0.1                                           # everything fits: keep everything
