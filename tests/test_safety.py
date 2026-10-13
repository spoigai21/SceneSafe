import numpy as np

from scenesafe.match.catalogue import Ad
from scenesafe.safety import rules as rl
from scenesafe.safety.tags import parse_tags


def test_parse_tags():
    assert parse_tags("none") == []
    assert parse_tags("Violence, death") == ["violence", "death"]
    assert parse_tags("a weapon and an accident") == ["weapons", "accidents"]


def test_breaks_stay_inside_a_film():
    assert rl.breaks(["a", "a", "a", "b", "b"]) == [(0, 1), (1, 2), (3, 4)]


def test_sellable_checks_both_neighbours_and_all_three_kinds():
    tags = [set(), {"accidents"}, set(), {"smoking"}, set()]
    brk = [(0, 1), (1, 2), (2, 3), (3, 4)]
    car = Ad("car", "automotive", "a car", (), ())                 # adjacency: no accidents
    tea = Ad("tea", "food_drink", "tea", (), ("smoking",))         # own avoid list: smoking
    ok = rl.sellable(tags, brk, [car, tea], block=set())
    assert ok[:, 0].tolist() == [False, False, True, True]
    assert ok[:, 1].tolist() == [True, True, False, False]
    assert not rl.sellable(tags, brk, [car], block={"smoking"})[2:, 0].any()


def test_allocate_respects_demand_and_takes_each_break_once():
    ok = np.ones((4, 2), bool)
    score = np.array([[4, 4], [3, 3], [2, 2], [1, 1]], float)
    sold = rl.allocate(ok, score, demand=2, seed=0)
    assert sold.tolist() == [2, 2]
    assert rl.allocate(ok, score, demand=3, seed=0).sum() == 4      # only 4 breaks to go round
    assert rl.allocate(np.zeros((4, 2), bool), score, demand=1).tolist() == [0, 0]
