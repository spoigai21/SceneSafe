"""Phase 4b: do general improvements make CLIP's picks fit more often? Rules: EVAL-PROTOCOL.md, Phase 4b amendment.

  practice-pool  v2's top 15 scenes per ad on the practice films (no cut-off), pooled and shuffled, with
                 labelling sheets. Scores are stored but never shown on the sheets.
  calibrate      reads results/phase4b/practice_labels_primary.json and picks tau (best F1, ties to lower).
  test           runs v2 once on the test films with the frozen tau; pairs not labelled in Phase 4 go into a
                 new shuffled pool with sheets.
  score          v1 vs v2 on the test films; writes results/phase4b/match_v2_cpu-laptop_seed0.json and a chart.
"""

import argparse
import json

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import luma, v2
from scenesafe.match.catalogue import ads, practice_films, test_films
from scenesafe.match.display import frame_strip, sheets
from scenesafe.match.metrics import agreement, bootstrap, ndcg_at_k, precision_at_k
from scenesafe.match.textvec import embed_texts

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4b"
V1 = REPO_ROOT / "results" / "phase4"
DATA = REPO_ROOT / "data" / "phase4b"
FRAMES = REPO_ROOT / "data" / "phase4" / "test" / "frames"    # shared with Phase 4, so seen scenes reuse strips
SCENE_THRESHOLD = 0.75
K = 10
PRACTICE_DEPTH = 15
AUTHOR_SAMPLE = 50


def vectors(catalogue):
    briefs = [a.brief for a in catalogue]
    t = embed_texts(v2.template_texts(briefs), DATA / "templates.npy")
    return v2.ensemble(t, len(briefs)), embed_texts(list(v2.NEUTRAL), DATA / "neutral.npy")


def score_films(ids):
    """Corpus, scenes and the (scene x ad) v2 scores for these films, plus how many seconds are junk."""
    catalogue = ads()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    ad_vecs, neutral = vectors(catalogue)
    s = v2.second_scores(corpus.emb, ad_vecs, neutral)
    is_junk = v2.junk_mask(*luma.load(ids))
    return catalogue, corpus, units, v2.scene_scores(s, is_junk, units.first), is_junk


def describe(corpus, units, u: int) -> dict:
    title = corpus.ids[units.title[u]]
    return {"scene": f"{title}@{units.start[u]:.0f}", "title": title,
            "start_s": round(float(units.start[u]), 2), "end_s": round(float(units.end[u]), 2)}


def make_pool(pairs: list[tuple], prefix: str, catalogue) -> list[dict]:
    """(ad id, scene dict) pairs -> shuffled pool entries with ids; no scores, no method."""
    by_ad = {a.id: a for a in catalogue}
    order = np.random.default_rng(0).permutation(len(pairs))
    return [{"pair": f"{prefix}{n:03d}", "ad": pairs[j][0], "category": by_ad[pairs[j][0]].category,
             "brief": by_ad[pairs[j][0]].brief, **pairs[j][1]} for n, j in enumerate(order, 1)]


def strips(pool: list[dict], frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for key, p in {p["scene"]: p for p in pool}.items():
        if not (frames_dir / f"{key}.jpg").exists():
            frame_strip(p["title"], p["start_s"], p["end_s"], frames_dir / f"{key}.jpg")


def practice_pool():
    catalogue, corpus, units, ss, is_junk = score_films(practice_films())
    pairs, scores = [], {}
    for i, ad in enumerate(catalogue):
        for u in v2.pick(ss[:, i], PRACTICE_DEPTH, None):
            d = describe(corpus, units, u)
            pairs.append((ad.id, d))
            scores[(ad.id, d["scene"])] = float(ss[u, i])
    pool = make_pool(pairs, "Q", catalogue)
    frames = DATA / "practice" / "frames"
    strips(pool, frames)
    sheets(pool, frames, DATA / "practice" / "sheets")
    OUT.mkdir(parents=True, exist_ok=True)
    for p in pool:
        p["score"] = scores[(p["ad"], p["scene"])]      # stored for calibration, never drawn on the sheets
    (OUT / "practice_pool.json").write_text(json.dumps({"films": "practice", "pairs": pool}, indent=1) + "\n")
    print(f"practice: {len(corpus.ids)} films, {len(units)} scenes, {int(is_junk.sum())} of {len(is_junk)} seconds "
          f"junk; {len(pool)} pairs to label in {DATA / 'practice' / 'sheets'}")


def calibrate():
    pool = json.loads((OUT / "practice_pool.json").read_text())["pairs"]
    labels = json.loads((OUT / "practice_labels_primary.json").read_text())["labels"]
    scores = np.array([p["score"] for p in pool])
    fits = np.array([labels[p["pair"]] == 1 for p in pool])
    tau, f1 = v2.best_f1_threshold(scores, fits)
    kept = scores > tau
    out = {"tau": tau, "f1": f1, "pairs": len(pool), "fits": int(fits.sum()), "kept": int(kept.sum()),
           "precision_kept": float(fits[kept].mean()) if kept.any() else None,
           "recall_kept": float(fits[kept].sum() / fits.sum()) if fits.any() else None,
           "rule": "maximise F1 of (score > tau) vs fits over practice pairs; ties to the lower tau"}
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def test():
    tau = json.loads((OUT / "calibration.json").read_text())["tau"]
    catalogue, corpus, units, ss, is_junk = score_films(test_films())
    v1_pool = json.loads((V1 / "pool.json").read_text())["pairs"]
    seen = {(p["ad"], p["scene"]) for p in v1_pool}
    picks, new = {}, []
    for i, ad in enumerate(catalogue):
        chosen = v2.pick(ss[:, i], K, tau)
        picks[ad.id] = [{**describe(corpus, units, u), "score": float(ss[u, i])} for u in chosen]
        new += [(ad.id, {k: d[k] for k in ("scene", "title", "start_s", "end_s")})
                for d in picks[ad.id] if (ad.id, d["scene"]) not in seen]
    pool = make_pool(new, "R", catalogue)
    strips(pool, FRAMES)
    sheets(pool, FRAMES, DATA / "test" / "sheets")
    meta = {"films": "test", "titles": test_films(), "tau": tau, "scene_threshold": SCENE_THRESHOLD,
            "junk_seconds": int(is_junk.sum()), "seconds": int(len(is_junk)), "model": DEFAULT_MODEL,
            "pretrained": DEFAULT_PRETRAINED, "templates": v2.TEMPLATES, "neutral": v2.NEUTRAL,
            "junk_rule": {"max_mean": v2.JUNK_MAX_MEAN, "max_std": v2.JUNK_MAX_STD}}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    returned = sum(len(v) for v in picks.values())
    print(f"test: {returned} scenes returned for {len(catalogue)} ads "
          f"({sum(1 for v in picks.values() if not v)} ads with none); {len(pool)} new pairs to label")


def score(seed: int):
    catalogue = ads()
    v1_picks = json.loads((V1 / "picks.json").read_text())["picks"]
    v1_pool = json.loads((V1 / "pool.json").read_text())["pairs"]
    v2_picks = json.loads((OUT / "picks.json").read_text())["picks"]
    v2_pool = json.loads((OUT / "pool.json").read_text())["pairs"]
    v1_labels = json.loads((V1 / "labels_primary.json").read_text())["labels"]
    v2_labels = json.loads((OUT / "labels_primary.json").read_text())["labels"]
    label = {(p["ad"], p["scene"]): v1_labels[p["pair"]] for p in v1_pool}
    label |= {(p["ad"], p["scene"]): v2_labels[p["pair"]] for p in v2_pool}

    per_ad, cols = {}, {k: [] for k in ("p_v1", "p_v2", "n_v1", "n_v2")}
    returned = fitting_returned = abstain_correct = abstain = 0
    for ad in catalogue:
        known = sum(v for (a, _), v in label.items() if a == ad.id)
        r1 = [label[(ad.id, s["scene"])] for s in v1_picks[ad.id]["clip"]]
        r2 = [label[(ad.id, s["scene"])] for s in v2_picks[ad.id]]
        row = {"fitting_scenes_known": known, "v1": r1, "v2": r2, "v2_returned": len(r2),
               "precision@10": {"v1": precision_at_k(r1, K), "v2": precision_at_k(r2, K)},
               "ndcg@10": {"v1": ndcg_at_k(r1, known, K), "v2": ndcg_at_k(r2, known, K)}}
        per_ad[ad.id] = row
        cols["p_v1"].append(row["precision@10"]["v1"])
        cols["p_v2"].append(row["precision@10"]["v2"])
        cols["n_v1"].append(row["ndcg@10"]["v1"])
        cols["n_v2"].append(row["ndcg@10"]["v2"])
        returned += len(r2)
        fitting_returned += sum(r2)
        if not r2:
            abstain += 1
            abstain_correct += known == 0
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for metric, (a, b) in {"precision@10": ("p_v1", "p_v2"), "ndcg@10": ("n_v1", "n_v2")}.items():
        d = arr[b] - arr[a]
        summary[metric] = {"v1": {"mean": float(arr[a].mean()), "ci95": bootstrap(arr[a], seed=seed)},
                           "v2": {"mean": float(arr[b].mean()), "ci95": bootstrap(arr[b], seed=seed)},
                           "v2-v1": {"mean": float(d.mean()), "ci95": bootstrap(d, seed=seed),
                                     "improves": bool(bootstrap(d, seed=seed)[0] > 0),
                                     "ads_better": int((d > 0).sum()), "ads_same": int((d == 0).sum()),
                                     "ads_worse": int((d < 0).sum())}}
    v1_returned = sum(len(v["clip"]) for v in v1_picks.values())
    v1_fit = sum(label[(a, s["scene"])] for a, v in v1_picks.items() for s in v["clip"])
    summary["precision_of_returned"] = {"v1": v1_fit / v1_returned, "v1_returned": v1_returned,
                                        "v2": fitting_returned / returned if returned else None, "v2_returned": returned}
    summary["abstentions"] = {"ads_with_none_returned": abstain, "correct": abstain_correct}

    data = {"summary": summary, "per_ad": per_ad, "new_pairs_labelled": len(v2_pool)}
    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p for p in v2_pool if p["pair"] in author]
        data["agreement"] = agreement([v2_labels[p["pair"]] for p in both], [author[p["pair"]] for p in both])
        data["agreement"]["disagreements"] = [p["pair"] for p in both if v2_labels[p["pair"]] != author[p["pair"]]]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4b amendment)", "k": K, "bootstrap_resamples": 10_000,
              "tau": json.loads((OUT / "calibration.json").read_text())["tau"]}
    print("  ->", write_result(PHASE, "match_v2", LANE, seed, config, data).rename(OUT / f"match_v2_{LANE}_seed{seed}.json"))
    for metric in ("precision@10", "ndcg@10"):
        s = summary[metric]
        d = s["v2-v1"]
        print(f"{metric}: v1 {s['v1']['mean']:.3f}  v2 {s['v2']['mean']:.3f}  v2-v1 {d['mean']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] improves={d['improves']}  "
              f"ads better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("precision of returned:", summary["precision_of_returned"], " abstentions:", summary["abstentions"])
    if "agreement" in data:
        a = data["agreement"]
        print(f"agreement on {a['n']} pairs: {a['agreement']:.2f}, kappa {a['cohen_kappa']:.2f}")
    plot(per_ad, summary, OUT / "match_v2.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [per_ad[a]["precision@10"]["v1"] for a in names], 0.4, color="#9e9e9e", label="v1 (Phase 4)")
    ax.bar(x + 0.2, [per_ad[a]["precision@10"]["v2"] for a in names], 0.4, color="#d62728", label="v2 (4b)")
    for i, a in enumerate(names):
        ax.text(i + 0.2, per_ad[a]["precision@10"]["v2"] + 0.02, str(per_ad[a]["v2_returned"]), ha="center", fontsize=7)
    ax.set_xticks(x, names, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("precision@10 (not returned = miss)")
    s = summary["precision@10"]
    ax.set_title(f"v1 {s['v1']['mean']:.2f} vs v2 {s['v2']['mean']:.2f} (numbers above red bars: scenes v2 returned)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["practice-pool", "calibrate", "test", "score"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"practice-pool": practice_pool, "calibrate": calibrate, "test": test}.get(args.step, lambda: score(args.seed))()


if __name__ == "__main__":
    main()
