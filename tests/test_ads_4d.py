"""Phase 4d catalogue: 60 distinct ads, the original 20 first, and a label-guide line for every one."""

from scenesafe.match import catalogue as cat


def test_sixty_ads_with_guide():
    ads = cat.ads_4d()
    ids = [a.id for a in ads]
    assert len(ads) == 60 and len(set(ids)) == 60
    assert ids[:20] == [a.id for a in cat.ads()]
    guide = cat.label_guide()
    assert set(guide) == set(ids)
    assert all(set(g) == {"fits", "not"} and g["fits"] and g["not"] for g in guide.values())


def test_themes_cover_every_ad():
    th = cat.themes()
    assert set(th) == {a.id for a in cat.ads_4d()}
    assert all(t["theme"] and len(t["query"]) == 3 and all(t["query"]) for t in th.values())


def test_graded_ndcg():
    from scenesafe.match.metrics import graded_ndcg_at_k
    assert graded_ndcg_at_k([2, 1, 0], [2, 1, 1], 3) < 1.0
    assert graded_ndcg_at_k([2, 1, 1], [2, 1, 1, 0], 3) == 1.0
    assert graded_ndcg_at_k([0, 0], [0], 2) == 0.0
