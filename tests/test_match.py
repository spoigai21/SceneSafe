import numpy as np
import pytest

from scenesafe.corpus import titles
from scenesafe.match import catalogue as cat
from scenesafe.match.methods import clip_scenes, sampled_scenes
from scenesafe.match.metrics import agreement, bootstrap, ndcg_at_k, precision_at_k


def test_split_is_disjoint_and_covers_corpus():
    test, practice = set(cat.test_films()), set(cat.practice_films())
    assert len(test) == 14 and not test & practice
    assert test | practice == {t["id"] for t in titles()}
    genre = cat.genre_of()
    assert sorted(genre[t] for t in test) == sorted([g for g in set(genre.values())] * 2)


def test_catalogue_is_well_formed():
    catalogue = cat.ads()
    assert len(catalogue) == 20 and len({a.id for a in catalogue}) == 20
    genres = set(cat.genre_of().values())
    vocab = {"violence", "weapons", "accidents", "death", "horror", "alcohol", "smoking"}
    for a in catalogue:
        assert a.brief and set(a.genres) <= genres and set(a.avoid) <= vocab


def test_precision_and_ndcg():
    assert precision_at_k([1, 0, 1], 10) == pytest.approx(0.2)        # missing items count as not relevant
    assert ndcg_at_k([1, 0, 1], 2, 3) == pytest.approx((1 + 1 / 2) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k([1, 1, 0], 2, 3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], 0, 3) == 0.0


def test_bootstrap_and_agreement():
    assert bootstrap(np.full(20, 0.4)) == pytest.approx((0.4, 0.4))
    lo, hi = bootstrap(np.r_[np.zeros(10), np.ones(10)])
    assert lo < 0.5 < hi
    same = agreement([1, 0, 1, 0], [1, 0, 1, 0])
    assert same["agreement"] == 1 and same["cohen_kappa"] == pytest.approx(1)
    # 3 of 4 agree; both say "fits" half the time -> expected 0.5, kappa 0.5
    assert agreement([1, 1, 0, 0], [1, 0, 0, 0])["agreement"] == 0.75
    assert agreement([1, 1, 0, 0], [1, 0, 1, 0])["cohen_kappa"] == pytest.approx(0.0)


def test_sampled_scenes_are_repeatable_distinct_and_eligible():
    scene_of_second = np.repeat(np.arange(30), 5)          # 30 scenes of 5 s
    eligible = np.arange(50, 150)                          # seconds of scenes 10..29 only
    a = sampled_scenes(eligible, scene_of_second, np.random.default_rng(7))
    b = sampled_scenes(eligible, scene_of_second, np.random.default_rng(7))
    assert a == b and len(a) == 10 and len(set(a)) == 10 and all(10 <= s < 30 for s in a)
    few = sampled_scenes(np.arange(0, 10), scene_of_second, np.random.default_rng(0))
    assert sorted(few) == [0, 1]                           # only 2 scenes available: stop, don't loop forever


def test_clip_scenes_keep_each_scene_once_in_score_order():
    scene_of_second = np.array([0, 0, 1, 1, 2, 2])
    rows, scores = np.array([3, 2, 0, 5, 1]), np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert clip_scenes(rows, scores, scene_of_second, k=10) == [1, 0, 2]
