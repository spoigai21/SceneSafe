"""Phase 4h: confirmation on fresh films. Do scene cards still beat frames on films nothing has been tuned on, and does
letting the card model judge a shortlist (re-rank) add to that? Rules: EVAL-PROTOCOL.md, "Phase 4h amendment".

  embed       ViT-B/32 embeddings of the films in corpus/manifest_4h.json, exactly as in Phase 2 (data/embeddings/).
  transcribe  Whisper on each film (data/transcripts/).
  cards       a scene card for every scene (data/cards/).
  test        three systems, top 10 per ad: frames, frames+cards (both exactly as in Phase 4g), and rerank (frames+cards'
              top 30, re-ordered by the card model's 0/1/2 judgement, ties by the frames+cards score). Every picked
              pair goes into one shuffled pool with labelling sheets.
  score       writes results/phase4h/match_confirm_cpu-laptop_seed0.json and a chart; with --labels relabel, the
              relabel under the Phase 4h addendum (match_confirm_relabel_*).
  repeat      re-runs the card writer and the judge on a random sample and checks the outputs are identical.
"""

import argparse
import json

import numpy as np

from run_phase4f import FRAMES, K, SCENE_THRESHOLD, ad_texts, describe, sheets
from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import cards as cd
from scenesafe.match import dialogue as dl
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d, label_guide, themes
from scenesafe.match.display import frame_strip
from scenesafe.match.metrics import agreement, bootstrap, graded_ndcg_at_k, precision_at_k
from scenesafe.match.transcribe import WHISPER_MODEL, scene_text, transcribe

PHASE = 4
LANE = "cpu-laptop"
MANIFEST = REPO_ROOT / "corpus" / "manifest_4h.json"
OUT = REPO_ROOT / "results" / "phase4h"
DATA = REPO_ROOT / "data" / "phase4h"
SHORTLIST = 30
AUTHOR_SAMPLE, AUTHOR_SEED = 100, 7
SYSTEMS = ("frames", "frames+cards", "rerank")
DECISIONS = (("frames", "frames+cards"), ("frames+cards", "rerank"))


def films() -> list[str]:
    return sorted(t["id"] for t in json.loads(MANIFEST.read_text())["titles"])


def embed():
    from scripts.embed_corpus import OUT_DIR, embed_title
    from scenesafe.corpus import video_path
    from scenesafe.embed.clip import ClipEmbedder

    b32 = ClipEmbedder("mps")
    b32.model.half()                                          # images: Phase 2 config, as in Phase 4d
    for tid in films():
        ref = OUT_DIR / f"{tid}.npz"
        if not ref.exists():
            times, e = embed_title(b32, video_path(tid))
            np.savez(ref, times=times, emb=e)
        print(f"  {tid}: {len(np.load(ref)['times'])} seconds", flush=True)


def setup():
    ids = films()
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
    card = dl.embed_text([cd.card_text(cards[n]["card"]) for n in names], DATA / "cards.npy") @ \
        dl.embed_text(ad_texts(catalogue), DATA / "ads.npy").T
    both = dl.zscore(frame) + dl.zscore(card)

    guide, th = label_guide(), themes()
    short = {ad.id: dl.top_k(both[:, i], SHORTLIST) for i, ad in enumerate(catalogue)}
    items = [{"key": f"{ad.id}|{names[u]}", "img": FRAMES / f"{names[u]}.jpg", "dialogue": text[u], "brief": ad.brief,
              "fits": guide[ad.id]["fits"], "theme": th[ad.id]["theme"]} for ad in catalogue for u in short[ad.id]]
    verdict = cd.judge(items, DATA / "judge.json")

    picks = {"frames": {}, "frames+cards": {}, "rerank": {}}
    for i, ad in enumerate(catalogue):
        picks["frames"][ad.id] = [{**describe(corpus, units, u), "score": float(frame[u, i])}
                                  for u in dl.top_k(frame[:, i], K)]
        picks["frames+cards"][ad.id] = [{**describe(corpus, units, u), "score": float(both[u, i])}
                                        for u in short[ad.id][:K]]
        order = sorted(short[ad.id], key=lambda u: (-max(verdict[f"{ad.id}|{names[u]}"], 0), -both[u, i]))
        picks["rerank"][ad.id] = [{**describe(corpus, units, u), "judge": verdict[f"{ad.id}|{names[u]}"],
                                   "score": float(both[u, i])} for u in order[:K]]

    pairs = {}
    for s in SYSTEMS:
        for ad_id, lst in picks[s].items():
            for x in lst:
                pairs.setdefault((ad_id, x["scene"]), {k: x[k] for k in ("scene", "title", "start_s", "end_s")})
    keys = list(pairs)
    by_ad = {a.id: a for a in catalogue}
    index = {n: u for u, n in enumerate(names)}
    order = np.random.default_rng(0).permutation(len(keys))
    pool = [{"pair": f"H{n:04d}", "ad": keys[j][0], "category": by_ad[keys[j][0]].category,
             "brief": by_ad[keys[j][0]].brief, **pairs[keys[j]], "dialogue": text[index[keys[j][1]]]}
            for n, j in enumerate(order, 1)]
    sheets(pool, out=DATA / "sheets")
    OUT.mkdir(parents=True, exist_ok=True)
    digits = [verdict[it["key"]] for it in items]
    meta = {"films": ids, "ads": len(catalogue), "scenes": len(units),
            "scenes_with_speech": int(sum(bool(t) for t in text)), "card_model": cd.CARD_MODEL,
            "text_model": dl.TEXT_MODEL, "whisper_model": WHISPER_MODEL,
            "frame_model": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}", "shortlist": SHORTLIST,
            "judge_answers": {str(v): digits.count(v) for v in (-1, 0, 1, 2)},
            "ads_judged_all_zero": [ad.id for ad in catalogue
                                    if all(verdict[f"{ad.id}|{names[u]}"] <= 0 for u in short[ad.id])]}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {len(units)} scenes in {len(ids)} films; {len(pool)} unique pairs to label; judge answers "
          f"{meta['judge_answers']}; sheets in {DATA / 'sheets'}")


def repeat(n_cards: int = 30, n_judge: int = 60):
    """Re-run the card writer and the judge on a random sample (default_rng(0)) from scratch, in a fresh cache, and
    count how many outputs are identical to the stored ones. Decoding is greedy, so all should be."""
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    stored = {}
    for t in films():
        stored |= {(t, k): v for k, v in json.loads((cd.CARD_DIR / f"{t}.json").read_text()).items()}
    keys = [list(stored)[j] for j in rng.choice(len(stored), n_cards, replace=False)]
    judged = json.loads((DATA / "judge.json").read_text())
    jkeys = [list(judged)[j] for j in rng.choice(len(judged), n_judge, replace=False)]
    by_ad, guide, th = {a.id: a for a in ads_4d()}, label_guide(), themes()
    dialogue = {k: v["dialogue"] for (_, k), v in stored.items()}
    with tempfile.TemporaryDirectory() as tmp:
        cd.CARD_DIR = Path(tmp)
        again = {}
        for t in sorted({t for t, _ in keys}):
            again |= cd.make_cards(t, [(k, FRAMES / f"{k}.jpg", stored[(t, k)]["dialogue"]) for tt, k in keys if tt == t])
        items = [{"key": k, "img": FRAMES / f"{k.split('|')[1]}.jpg", "dialogue": dialogue[k.split("|")[1]],
                  "brief": by_ad[k.split("|")[0]].brief, "fits": guide[k.split("|")[0]]["fits"],
                  "theme": th[k.split("|")[0]]["theme"]} for k in jkeys]
        cd.judge(items, Path(tmp) / "judge.json")
        answers = json.loads((Path(tmp) / "judge.json").read_text())
    same_cards = [k for t, k in keys if again[k]["card"] == stored[(t, k)]["card"]]
    same_judge = [k for k in jkeys if answers[k]["answer"] == judged[k]["answer"]]
    out = {"cards": {"n": n_cards, "identical": len(same_cards),
                     "different": [k for _, k in keys if k not in same_cards]},
           "judge": {"n": n_judge, "identical": len(same_judge), "different": [k for k in jkeys if k not in same_judge]},
           "sample": "default_rng(0)", "decoding": "greedy (temperature 0)"}
    (OUT / "repeat.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"repeat: cards {len(same_cards)}/{n_cards} identical, judge {len(same_judge)}/{n_judge} identical")


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
        for a, b in DECISIONS + (("frames", "rerank"),):
            d = arr[f"{m}_{b}"] - arr[f"{m}_{a}"]
            ci = bootstrap(d, seed=seed)
            summary[m][f"{b} - {a}"] = {"mean": float(d.mean()), "ci95": ci, "above_zero": bool(ci[0] > 0),
                                        "below_zero": bool(ci[1] < 0), "ads_better": int((d > 0).sum()),
                                        "ads_same": int((d == 0).sum()), "ads_worse": int((d < 0).sum())}
    return summary, per_ad


def score(seed: int, labels: str = "primary"):
    """labels: "primary" (the pre-registered labels) or "relabel" (the relabel under the Phase 4h addendum)."""
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
            "judge_answers": meta["judge_answers"], "ads_judged_all_zero": meta["ads_judged_all_zero"],
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
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4h amendment)", "labels": labels, "k": K, "shortlist": SHORTLIST,
              "bootstrap_resamples": 10_000, "card_model": cd.CARD_MODEL, "text_model": dl.TEXT_MODEL,
              "whisper_model": WHISPER_MODEL, "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    stem = "match_confirm" if labels == "primary" else f"match_confirm_{labels}"
    print("  ->", write_result(PHASE, stem, LANE, seed, config, data).rename(OUT / f"{stem}_{LANE}_seed{seed}.json"))
    for m, title in (("thematic", "thematic precision@10"), ("literal", "literal precision@10"),
                     ("ndcg", "graded nDCG@10")):
        s = summary[m]
        print(f"{title}: " + "  ".join(f"{k} {s[k]['mean']:.3f}" for k in SYSTEMS))
        for k in ("frames+cards - frames", "rerank - frames+cards", "rerank - frames"):
            d = s[k]
            print(f"   {k}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
                  f"better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("label counts:", data["label_counts"], " judge answers:", data["judge_answers"])
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
    for j, (s, c) in enumerate((("frames", "#9e9e9e"), ("frames+cards", "#2b55a8"), ("rerank", "#c77c1e"))):
        ax.bar(x + (j - 1) * 0.27, [per_ad[a][s]["thematic_precision@10"] for a in names], 0.27, color=c, label=s)
    ax.set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("thematic precision@10")
    s = summary["thematic"]
    ax.set_title("Phase 4h (fresh films): frames {:.2f}, frames+cards {:.2f}, rerank {:.2f}".format(
        s["frames"]["mean"], s["frames+cards"]["mean"], s["rerank"]["mean"]), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "transcribe", "cards", "test", "score", "repeat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", choices=["primary", "relabel"], default="primary")
    args = ap.parse_args()
    if args.step == "embed":
        embed()
    elif args.step == "transcribe":
        for t in films():
            print(t, len(transcribe(t)["segments"]), flush=True)
    elif args.step == "cards":
        make_all_cards()
    elif args.step == "repeat":
        repeat()
    else:
        test() if args.step == "test" else score(args.seed, args.labels)


if __name__ == "__main__":
    main()
