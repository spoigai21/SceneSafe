"""Phase 4g: does adding a written scene description ("scene card": a small vision-language model looks at the frames
and reads the dialogue) find scenes that suit the ad more often than frames alone? Rules: EVAL-PROTOCOL.md,
"Phase 4g amendment". Tested on the 12 Phase 4d films the card method was not tried on.

  cards   write a card for every scene of the 12 films (data/cards/). No results.
  test    three systems, top 10 per ad: frames (Phase 4d ViT-B/32 scene scores), cards (card text vs ad text),
          frames+cards (equal-weight z-scores). Every picked pair goes into one shuffled pool; each labelling card shows
          the 4 frames and the scene's dialogue, never the scene card or the system.
  score   frames vs cards vs frames+cards; writes results/phase4g/match_cards_cpu-laptop_seed0.json and a chart.
"""

import argparse
import json

import numpy as np

from run_phase4f import FRAMES, K, P4D, SCENE_THRESHOLD, ad_texts, describe, films, sheets
from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import cards as cd
from scenesafe.match import dialogue as dl
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d
from scenesafe.match.display import frame_strip
from scenesafe.match.metrics import agreement, bootstrap, graded_ndcg_at_k, precision_at_k
from scenesafe.match.transcribe import WHISPER_MODEL, scene_text, transcribe

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4g"
DATA = REPO_ROOT / "data" / "phase4g"
PRACTICE_FILMS = ("calvary", "Hitchcock_Secret_Agent")      # the card method was tried on these; left out of the test
AUTHOR_SAMPLE, AUTHOR_SEED = 100, 6
SYSTEMS = ("frames", "cards", "frames+cards")


def test_films() -> list[str]:
    return [t for t in films() if t not in PRACTICE_FILMS]


def setup():
    ids = test_films()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    tr = {t: transcribe(t)["segments"] for t in ids}
    text = [scene_text(tr[corpus.ids[units.title[u]]], units.start[u], units.end[u]) for u in range(len(units))]
    return ids, corpus, units, text


def make_all_cards():
    ids, corpus, units, text = setup()
    for t in ids:
        todo = []
        for u in range(len(units)):
            if corpus.ids[units.title[u]] != t:
                continue
            d = describe(corpus, units, u)
            img = FRAMES / f"{d['scene']}.jpg"
            if not img.exists():
                frame_strip(t, d["start_s"], d["end_s"], img)
            todo.append((d["scene"], img, text[u]))
        print(t, len(todo), "scenes", flush=True)
        cd.make_cards(t, todo)


def test():
    catalogue = ads_4d()
    ids, corpus, units, text = setup()
    names = [describe(corpus, units, u)["scene"] for u in range(len(units))]
    cards = {}
    for t in ids:
        cards |= json.loads((cd.CARD_DIR / f"{t}.json").read_text())
    missing = [n for n in names if n not in cards]
    assert not missing, f"scenes without a card: {missing[:5]} (run the cards step)"

    q = np.load(REPO_ROOT / "data" / "phase4d" / "briefs_b32.npy")      # the Phase 4d frame queries, unchanged
    frame = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    card_vecs = dl.embed_text([cd.card_text(cards[n]["card"]) for n in names], DATA / "cards.npy")
    ad_vecs = dl.embed_text(ad_texts(catalogue), DATA / "ads.npy")
    card = card_vecs @ ad_vecs.T
    score = {"frames": frame, "cards": card, "frames+cards": dl.zscore(frame) + dl.zscore(card)}

    picks = {s: {ad.id: [{**describe(corpus, units, u), "score": float(score[s][u, i])}
                         for u in dl.top_k(score[s][:, i], K)] for i, ad in enumerate(catalogue)} for s in SYSTEMS}
    pairs = {}
    for s in SYSTEMS:
        for ad_id, lst in picks[s].items():
            for x in lst:
                pairs.setdefault((ad_id, x["scene"]), {k: x[k] for k in ("scene", "title", "start_s", "end_s")})
    keys = list(pairs)
    by_ad = {a.id: a for a in catalogue}
    index = {n: u for u, n in enumerate(names)}
    order = np.random.default_rng(0).permutation(len(keys))
    pool = [{"pair": f"G{n:04d}", "ad": keys[j][0], "category": by_ad[keys[j][0]].category,
             "brief": by_ad[keys[j][0]].brief, **pairs[keys[j]], "dialogue": text[index[keys[j][1]]]}
            for n, j in enumerate(order, 1)]
    sheets(pool, out=DATA / "sheets")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"films": ids, "practice_films_left_out": list(PRACTICE_FILMS), "ads": len(catalogue),
            "scenes": len(units), "scenes_with_speech": int(sum(bool(t) for t in text)),
            "card_model": cd.CARD_MODEL, "text_model": dl.TEXT_MODEL, "whisper_model": WHISPER_MODEL,
            "frame_model": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}", "ad_text": "brief + '. ' + theme line",
            "card_fields_matched": list(cd.MATCHED_FIELDS)}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {len(units)} scenes in {len(ids)} films; {len(pool)} unique pairs to label; sheets in {DATA / 'sheets'}")


def compare(picks: dict, label: dict, seed: int):
    per_ad, cols = {}, {f"{m}_{s}": [] for m in ("thematic", "literal", "ndcg") for s in SYSTEMS}
    for ad in ads_4d():
        known = [v for (a, _), v in label.items() if a == ad.id]
        row = {}
        for s in SYSTEMS:
            g = [label[(ad.id, x["scene"])] for x in picks[s][ad.id]]
            row[s] = {"gains": g, "thematic_precision@10": precision_at_k([int(v >= 1) for v in g], K),
                      "literal_precision@10": precision_at_k([int(v == 2) for v in g], K),
                      "graded_ndcg@10": graded_ndcg_at_k(g, known, K)}
            for m, key in (("thematic", "thematic_precision@10"), ("literal", "literal_precision@10"),
                           ("ndcg", "graded_ndcg@10")):
                cols[f"{m}_{s}"].append(row[s][key])
        per_ad[ad.id] = row
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for m in ("thematic", "literal", "ndcg"):
        summary[m] = {s: {"mean": float(arr[f"{m}_{s}"].mean()), "ci95": bootstrap(arr[f"{m}_{s}"], seed=seed)}
                      for s in SYSTEMS}
        for a, b in (("frames", "frames+cards"), ("frames", "cards"), ("cards", "frames+cards")):
            d = arr[f"{m}_{b}"] - arr[f"{m}_{a}"]
            ci = bootstrap(d, seed=seed)
            summary[m][f"{b} - {a}"] = {"mean": float(d.mean()), "ci95": ci, "above_zero": bool(ci[0] > 0),
                                        "below_zero": bool(ci[1] < 0), "ads_better": int((d > 0).sum()),
                                        "ads_same": int((d == 0).sum()), "ads_worse": int((d < 0).sum())}
    return summary, per_ad


def score(seed: int, labels: str = "primary"):
    """labels: "primary" (the pre-registered labels) or "relabel" (the relabel under the author's rules, Phase 4g
    addendum); each writes its own result file."""
    meta = json.loads((OUT / "pool.json").read_text())
    pool = meta["pairs"]
    picks = json.loads((OUT / "picks.json").read_text())["picks"]
    primary = json.loads((OUT / f"labels_{labels}.json").read_text())["labels"]
    missing = [p["pair"] for p in pool if p["pair"] not in primary]
    assert not missing, f"unlabelled pairs: {missing[:10]}"
    label = {(p["ad"], p["scene"]): primary[p["pair"]] for p in pool}
    summary, per_ad = compare(picks, label, seed)
    data = {"summary": summary, "per_ad": per_ad, "pairs_labelled": len(pool), "films": meta["films"],
            "scenes": meta["scenes"], "scenes_with_speech": meta["scenes_with_speech"],
            "label_counts": {str(v): sum(1 for x in primary.values() if x == v) for v in (0, 1, 2)}}
    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p["pair"] for p in pool if p["pair"] in author]
        a, b = [primary[k] for k in both], [author[k] for k in both]
        data["agreement"] = {"n": len(both), "exact_3_level": float(np.mean(np.array(a) == np.array(b))),
                             "thematic_or_better": agreement([int(x >= 1) for x in a], [int(x >= 1) for x in b]),
                             "literal": agreement([int(x == 2) for x in a], [int(x == 2) for x in b]),
                             "disagreements": [k for k in both if primary[k] != author[k]]}
        mixed = label | {(p["ad"], p["scene"]): author[p["pair"]] for p in pool if p["pair"] in author}
        data["sensitivity_author_labels_where_present"] = compare(picks, mixed, seed)[0]["thematic"]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4g amendment)", "labels": labels, "k": K, "bootstrap_resamples": 10_000,
              "card_model": cd.CARD_MODEL, "text_model": dl.TEXT_MODEL, "whisper_model": WHISPER_MODEL,
              "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    stem = "match_cards" if labels == "primary" else f"match_cards_{labels}"
    print("  ->", write_result(PHASE, stem, LANE, seed, config, data).rename(OUT / f"{stem}_{LANE}_seed{seed}.json"))
    for m, name in (("thematic", "thematic precision@10"), ("literal", "literal precision@10"), ("ndcg", "graded nDCG@10")):
        s = summary[m]
        print(f"{name}: " + "  ".join(f"{k} {s[k]['mean']:.3f}" for k in SYSTEMS))
        for k in ("frames+cards - frames", "cards - frames", "frames+cards - cards"):
            d = s[k]
            print(f"   {k}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
                  f"better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("label counts:", data["label_counts"])
    if "agreement" in data:
        g = data["agreement"]
        print(f"agreement on {g['n']}: exact {g['exact_3_level']:.2f}; thematic-or-better kappa "
              f"{g['thematic_or_better']['cohen_kappa']:.2f}; literal kappa {g['literal']['cohen_kappa']:.2f}")
    plot(per_ad, summary, OUT / f"{stem}.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, ax = plt.subplots(figsize=(18, 4.8))
    x = np.arange(len(names))
    for j, (s, c) in enumerate((("frames", "#9e9e9e"), ("cards", "#c77c1e"), ("frames+cards", "#2b55a8"))):
        ax.bar(x + (j - 1) * 0.27, [per_ad[a][s]["thematic_precision@10"] for a in names], 0.27, color=c, label=s)
    ax.set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("thematic precision@10")
    s = summary["thematic"]
    ax.set_title("Phase 4g: frames {:.2f}, cards {:.2f}, frames+cards {:.2f}".format(
        s["frames"]["mean"], s["cards"]["mean"], s["frames+cards"]["mean"]), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cards", "test", "score"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", choices=["primary", "relabel"], default="primary")
    args = ap.parse_args()
    if args.step == "cards":
        make_all_cards()
    else:
        test() if args.step == "test" else score(args.seed, args.labels)


if __name__ == "__main__":
    main()
