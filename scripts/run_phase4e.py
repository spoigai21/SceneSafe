"""Phase 4e: does describing each ad by its theme (not only its literal subject) find scenes that suit the ad more
often? Graded labels: 2 literal fit, 1 thematic fit, 0 neither. Rules: EVAL-PROTOCOL.md, "Phase 4e amendment".

  test    the theme system (ViT-B/32; the brief plus the ad's 3 theme phrases, averaged) runs the v1 method once on the
          Phase 4d films. The literal system is Phase 4d's ViT-B/32 run, reused. Pairs judged a literal fit in 4d keep
          label 2; every other pair from either system goes into one shuffled pool for graded labelling.
  score   literal vs theme system; writes results/phase4e/match_theme_cpu-laptop_seed0.json and a chart.
"""

import argparse
import json

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d, label_guide, themes
from scenesafe.match.display import frame_strip, sheets
from scenesafe.match.metrics import agreement, bootstrap, graded_ndcg_at_k, precision_at_k
from scenesafe.match.textvec import embed_texts

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4e"
P4D = REPO_ROOT / "results" / "phase4d"
DATA = REPO_ROOT / "data" / "phase4e"
FRAMES = REPO_ROOT / "data" / "phase4d" / "frames"     # shared with 4d, so seen scenes reuse strips
SCENE_THRESHOLD = 0.75
K = 10
AUTHOR_SAMPLE, AUTHOR_SEED = 100, 4
SYSTEMS = ("literal", "theme")


def theme_vectors(catalogue) -> np.ndarray:
    """Per ad: the unit mean of the literal brief and its 3 theme phrases (ViT-B/32 text embeddings)."""
    th = themes()
    texts = [t for a in catalogue for t in [a.brief, *th[a.id]["query"]]]
    v = embed_texts(texts, DATA / "theme_texts.npy").reshape(len(catalogue), 4, -1).mean(axis=1)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def describe(corpus, units, u: int) -> dict:
    title = corpus.ids[units.title[u]]
    return {"scene": f"{title}@{units.start[u]:.0f}", "title": title,
            "start_s": round(float(units.start[u]), 2), "end_s": round(float(units.end[u]), 2)}


def test():
    catalogue = ads_4d()
    d4 = json.loads((P4D / "pool.json").read_text())
    ids = d4["films"]
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    q = theme_vectors(catalogue)
    ss = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    theme_picks = {ad.id: [{**describe(corpus, units, u), "score": float(ss[u, i])} for u in v2.pick(ss[:, i], K, None)]
                   for i, ad in enumerate(catalogue)}

    lit4d = json.loads((P4D / "labels_primary.json").read_text())["labels"]
    literal_fit = {(p["ad"], p["scene"]) for p in d4["pairs"] if lit4d[p["pair"]] == 1}
    pairs = {}
    for p in d4["pairs"]:
        pairs.setdefault((p["ad"], p["scene"]), {k: p[k] for k in ("scene", "title", "start_s", "end_s")})
    for ad_id, picks in theme_picks.items():
        for s in picks:
            pairs.setdefault((ad_id, s["scene"]), {k: s[k] for k in ("scene", "title", "start_s", "end_s")})
    keys = [k for k in pairs if k not in literal_fit]
    by_ad = {a.id: a for a in catalogue}
    order = np.random.default_rng(0).permutation(len(keys))
    pool = [{"pair": f"E{n:04d}", "ad": keys[j][0], "category": by_ad[keys[j][0]].category,
             "brief": by_ad[keys[j][0]].brief, **pairs[keys[j]]} for n, j in enumerate(order, 1)]
    for key, p in {p["scene"]: p for p in pool}.items():
        if not (FRAMES / f"{key}.jpg").exists():
            frame_strip(p["title"], p["start_s"], p["end_s"], FRAMES / f"{key}.jpg")
    guide, th = label_guide(), themes()
    sheets(pool, FRAMES, DATA / "sheets", {a: {**guide[a], "theme": th[a]["theme"]} for a in guide})
    OUT.mkdir(parents=True, exist_ok=True)
    new = sum(1 for k in pairs if k not in {(p["ad"], p["scene"]) for p in d4["pairs"]})
    meta = {"films": ids, "ads": len(catalogue), "scene_threshold": SCENE_THRESHOLD, "model": DEFAULT_MODEL,
            "pretrained": DEFAULT_PRETRAINED, "literal_system": "Phase 4d ViT-B/32 picks (reused)",
            "literal_fits_from_4d": len(literal_fit), "new_pairs_from_theme_system": new}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": {"theme": theme_picks}}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {sum(len(v) for v in theme_picks.values())} theme picks, {new} pairs not seen in 4d; "
          f"{len(literal_fit)} pairs keep label 2 from 4d; {len(pool)} pairs to label; sheets in {DATA / 'sheets'}")


def labels(kind: str = "primary") -> dict[tuple[str, str], int]:
    """(ad, scene) -> 0/1/2. 4d literal fits are 2; every other pair takes its Phase 4e label."""
    d4 = json.loads((P4D / "pool.json").read_text())["pairs"]
    lit4d = json.loads((P4D / "labels_primary.json").read_text())["labels"]
    out = {(p["ad"], p["scene"]): 2 for p in d4 if lit4d[p["pair"]] == 1}
    lab = json.loads((OUT / f"labels_{kind}.json").read_text())["labels"]
    for p in json.loads((OUT / "pool.json").read_text())["pairs"]:
        if p["pair"] in lab:
            out[(p["ad"], p["scene"])] = lab[p["pair"]]
    return out


def compare(picks: dict, label: dict, seed: int):
    per_ad, cols = {}, {f"{m}_{s}": [] for m in ("thematic", "literal", "ndcg") for s in SYSTEMS}
    for ad in ads_4d():
        known = [v for (a, _), v in label.items() if a == ad.id]
        row = {"thematic_or_better_known": sum(v >= 1 for v in known), "literal_known": sum(v == 2 for v in known)}
        for s in SYSTEMS:
            g = [label[(ad.id, x["scene"])] for x in picks[s][ad.id]]
            row[s] = {"gains": g, "thematic_precision@10": precision_at_k([int(v >= 1) for v in g], K),
                      "literal_precision@10": precision_at_k([int(v == 2) for v in g], K),
                      "graded_ndcg@10": graded_ndcg_at_k(g, known, K)}
            cols[f"thematic_{s}"].append(row[s]["thematic_precision@10"])
            cols[f"literal_{s}"].append(row[s]["literal_precision@10"])
            cols[f"ndcg_{s}"].append(row[s]["graded_ndcg@10"])
        per_ad[ad.id] = row
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for metric in ("thematic", "literal", "ndcg"):
        a, b = arr[f"{metric}_literal"], arr[f"{metric}_theme"]
        d = b - a
        ci = bootstrap(d, seed=seed)
        summary[metric] = {"literal": {"mean": float(a.mean()), "ci95": bootstrap(a, seed=seed)},
                           "theme": {"mean": float(b.mean()), "ci95": bootstrap(b, seed=seed)},
                           "theme-literal": {"mean": float(d.mean()), "ci95": ci, "above_zero": bool(ci[0] > 0),
                                             "below_zero": bool(ci[1] < 0), "ads_better": int((d > 0).sum()),
                                             "ads_same": int((d == 0).sum()), "ads_worse": int((d < 0).sum())}}
    return summary, per_ad


def score(seed: int):
    pool = json.loads((OUT / "pool.json").read_text())["pairs"]
    primary_raw = json.loads((OUT / "labels_primary.json").read_text())["labels"]
    missing = [p["pair"] for p in pool if p["pair"] not in primary_raw]
    assert not missing, f"unlabelled pairs: {missing[:10]}"
    picks = {"literal": json.loads((P4D / "picks.json").read_text())["picks"]["b32"],
             "theme": json.loads((OUT / "picks.json").read_text())["picks"]["theme"]}
    label = labels("primary")
    summary, per_ad = compare(picks, label, seed)
    lit4d = json.loads((P4D / "labels_primary.json").read_text())["labels"]
    d4 = {(p["ad"], p["scene"]): lit4d[p["pair"]] for p in json.loads((P4D / "pool.json").read_text())["pairs"]}
    relabelled_2 = sum(1 for p in pool if (p["ad"], p["scene"]) in d4 and primary_raw[p["pair"]] == 2)
    data = {"summary": summary, "per_ad": per_ad, "pairs_labelled": len(pool),
            "label_counts": {str(v): sum(1 for x in primary_raw.values() if x == v) for v in (0, 1, 2)},
            "4d_non_fits_now_labelled_literal": relabelled_2}

    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p["pair"] for p in pool if p["pair"] in author]
        a, b = [primary_raw[k] for k in both], [author[k] for k in both]
        data["agreement"] = {
            "n": len(both), "exact_3_level": float(np.mean(np.array(a) == np.array(b))),
            "thematic_or_better": agreement([int(x >= 1) for x in a], [int(x >= 1) for x in b]),
            "literal": agreement([int(x == 2) for x in a], [int(x == 2) for x in b]),
            "disagreements": [k for k in both if primary_raw[k] != author[k]]}
        mixed = label | {k: v for k, v in labels("author").items() if k in {(p["ad"], p["scene"]) for p in pool
                                                                               if p["pair"] in author}}
        data["sensitivity_author_labels_where_present"] = compare(picks, mixed, seed)[0]["thematic"]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4e amendment)", "k": K, "bootstrap_resamples": 10_000,
              "model": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}", "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    print("  ->", write_result(PHASE, "match_theme", LANE, seed, config, data)
          .rename(OUT / f"match_theme_{LANE}_seed{seed}.json"))
    for metric, name in (("thematic", "thematic precision@10"), ("literal", "literal precision@10"),
                         ("ndcg", "graded nDCG@10")):
        s = summary[metric]
        d = s["theme-literal"]
        print(f"{name}: literal {s['literal']['mean']:.3f}  theme {s['theme']['mean']:.3f}  diff {d['mean']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  ads better/same/worse "
              f"{d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("label counts:", data["label_counts"], " 4d non-fits now labelled 2:", relabelled_2)
    if "agreement" in data:
        g = data["agreement"]
        print(f"agreement on {g['n']}: exact {g['exact_3_level']:.2f}; thematic-or-better "
              f"{g['thematic_or_better']['agreement']:.2f} kappa {g['thematic_or_better']['cohen_kappa']:.2f}; "
              f"literal {g['literal']['agreement']:.2f} kappa {g['literal']['cohen_kappa']:.2f}")
        s = data["sensitivity_author_labels_where_present"]
        print(f"sensitivity (thematic): literal {s['literal']['mean']:.3f}  theme {s['theme']['mean']:.3f}  "
              f"diff {s['theme-literal']['mean']:+.3f} [{s['theme-literal']['ci95'][0]:+.3f}, "
              f"{s['theme-literal']['ci95'][1]:+.3f}]")
    plot(per_ad, summary, OUT / "match_theme.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    x = np.arange(len(names))
    for ax, metric, title in ((axes[0], "thematic_precision@10", "thematic or literal fit"),
                              (axes[1], "literal_precision@10", "literal fit only")):
        ax.bar(x - 0.2, [per_ad[a]["literal"][metric] for a in names], 0.4, color="#9e9e9e", label="literal system")
        ax.bar(x + 0.2, [per_ad[a]["theme"][metric] for a in names], 0.4, color="#2b55a8", label="theme system")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("precision@10")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    axes[1].set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    s = summary["thematic"]
    fig.suptitle(f"Phase 4e: thematic precision@10 literal {s['literal']['mean']:.2f} vs theme {s['theme']['mean']:.2f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["test", "score"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test() if args.step == "test" else score(args.seed)


if __name__ == "__main__":
    main()
