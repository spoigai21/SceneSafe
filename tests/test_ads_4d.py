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
