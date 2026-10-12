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


def test_dialogue_scoring():
    import numpy as np
    from scenesafe.match import dialogue as dl
    assert dl.chunks("") == [] and dl.chunks("a b c") == ["a b c"]
    long = " ".join(str(i) for i in range(120))
    ch = dl.chunks(long)
    assert all(len(c.split()) <= dl.CHUNK_WORDS for c in ch) and ch[-1].split()[-1] == "119"
    vecs = np.eye(3, dtype=np.float32)
    s = dl.scene_dialogue_scores(vecs, np.array([0, 0, 2]), 4, np.array([[0, 1, 0]], np.float32))
    assert s[0, 0] == 1 and np.isnan(s[1, 0]) and s[2, 0] == 0 and np.isnan(s[3, 0])
    comb = dl.combined(np.array([[1.0], [2.0], [3.0], [4.0]]), s)
    assert np.isfinite(comb).all()
    assert dl.top_k(np.array([0.1, np.nan, 0.5]), 5) == [2, 0]


def test_speech_filter():
    from scenesafe.match.transcribe import scene_text
    seg = lambda a, b, t: {"start": a, "end": b, "text": t, "no_speech_prob": 0.0, "avg_logprob": -0.3}
    segs = [seg(0, 2, "Hello there."), seg(2, 32, "Thank you."),          # 30 s of music read as "Thank you."
            seg(32, 33, "I can."), seg(33, 34, "I can."), seg(34, 35, "I can."), seg(35, 37, "Where to?")]
    assert scene_text(segs, 0, 40) == "Hello there. I can. Where to?"
    assert scene_text(segs, 33, 40) == "Where to?"


def test_card_text():
    from scenesafe.match.cards import card_text, cut
    card = "place: a saloon\nvisible: bar, glasses\nactivity: men drinking\ntalk: none\nmood: tense"
    assert card_text(card) == "a saloon bar, glasses men drinking"
    assert card_text("place: road\ntalk: money, a loan") == "road money, a loan"
    assert cut("a bc d", 4) == "a" and cut("short") == "short"


def test_judge_digit():
    from scenesafe.match.cards import parse_digit
    assert parse_digit("2") == 2 and parse_digit(" 1.") == 1 and parse_digit("0\n") == 0
    assert parse_digit("Answer: 2") == 2 and parse_digit("yes") == -1
