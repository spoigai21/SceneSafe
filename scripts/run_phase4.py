"""Phase 4: does CLIP find better ad placements than random scenes or the film's genre? Rules: EVAL-PROTOCOL.md.

  retrieve  each method picks 10 scenes per ad; the picks are pooled into (ad, scene) pairs, shuffled, and each
            scene gets a 4-frame strip for judging. Writes results/phase4/{picks,pool}.json (test films) and
            labelling sheets in data/phase4/<films>/sheets/. --films practice only checks that the pipeline works.
  score     reads results/phase4/labels_primary.json (and labels_author.json if present) and writes
            results/phase4/match_cpu-laptop_seed0.json and match.png.
"""

import argparse
import json
import subprocess
import sys

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match.catalogue import ADS, ads, genre_of, practice_films, test_films
from scenesafe.match.display import frame_strip, sheets
from scenesafe.match.methods import K, clip_scenes, sampled_scenes
from scenesafe.match.metrics import agreement, bootstrap, ndcg_at_k, precision_at_k

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4"
DATA = REPO_ROOT / "data" / "phase4"
SCENE_THRESHOLD = 0.75     # Phase 3 decision
OVERFETCH = 1000           # seconds fetched before grouping into scenes (Phase 3)
METHODS = ("clip", "genre", "random")

# torch and faiss must not share a process (BUGS.md #7), so the briefs are embedded in a child process.
EMBED_BRIEFS = """
import sys, numpy as np, yaml
from scenesafe.embed.clip import ClipEmbedder
briefs = [a["brief"] for a in yaml.safe_load(open(sys.argv[1]).read())["ads"]]
np.save(sys.argv[2], ClipEmbedder("mps").embed_text(briefs).numpy().astype(np.float32))
"""


def scene_key(title: str, start: float) -> str:
    return f"{title}@{start:.0f}"


def brief_vectors() -> np.ndarray:
    cache = DATA / "briefs.npy"
    if not cache.exists() or cache.stat().st_mtime < ADS.stat().st_mtime:
        DATA.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-c", EMBED_BRIEFS, str(ADS), str(cache)], check=True, cwd=REPO_ROOT)
    return np.load(cache)


def retrieve(films: str):
    import faiss
    from scenesafe.index.search import IndexSpec, build

    ids = test_films() if films == "test" else practice_films()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    scene_of_second = units.of_second()
    genre = genre_of()
    catalogue = ads()
    q = brief_vectors()
    assert len(q) == len(catalogue)

    index = build(IndexSpec("flat"), corpus.emb)
    hit_scores, hit_rows = index.search(q, OVERFETCH)
    all_seconds = np.arange(len(corpus.emb))
    title_genre = np.array([genre[corpus.ids[t]] for t in corpus.title])

    def describe(u: int) -> dict:
        return {"scene": scene_key(corpus.ids[units.title[u]], units.start[u]), "title": corpus.ids[units.title[u]],
                "genre": genre[corpus.ids[units.title[u]]], "start_s": round(float(units.start[u]), 2),
                "end_s": round(float(units.end[u]), 2)}

    picks = {}
    for i, ad in enumerate(catalogue):
        in_genre = all_seconds[np.isin(title_genre, ad.genres)]
        picks[ad.id] = {
            "clip": [describe(u) for u in clip_scenes(hit_rows[i], hit_scores[i], scene_of_second)],
            "genre": [describe(u) for u in sampled_scenes(in_genre, scene_of_second, np.random.default_rng(1000 + i))],
            "random": [describe(u) for u in sampled_scenes(all_seconds, scene_of_second, np.random.default_rng(1000 + i))],
        }

    # pool: unique (ad, scene) pairs, shuffled; no method information
    by_ad = {a.id: a for a in catalogue}
    pairs = {}
    for ad_id, methods in picks.items():
        for scenes_ in methods.values():
            for s in scenes_:
                pairs.setdefault((ad_id, s["scene"]), s)
    keys = list(pairs)
    order = np.random.default_rng(0).permutation(len(keys))
    pool = []
    for n, j in enumerate(order, 1):
        ad_id, key = keys[j]
        s = pairs[(ad_id, key)]
        pool.append({"pair": f"P{n:03d}", "ad": ad_id, "category": by_ad[ad_id].category, "brief": by_ad[ad_id].brief,
                     "scene": key, "title": s["title"], "start_s": s["start_s"], "end_s": s["end_s"]})

    base = DATA / films
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for key, s in {p["scene"]: p for p in pool}.items():
        out = frames_dir / f"{key}.jpg"
        if not out.exists():
            frame_strip(s["title"], s["start_s"], s["end_s"], out)
    sheets(pool, frames_dir, base / "sheets")

    dest = OUT if films == "test" else base
    dest.mkdir(parents=True, exist_ok=True)
    meta = {"films": films, "titles": ids, "scene_threshold": SCENE_THRESHOLD, "overfetch": OVERFETCH,
            "seconds": int(len(corpus.emb)), "scenes": int(len(units)), "faiss_version": faiss.__version__,
            "model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED}
    (dest / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (dest / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"{films}: {len(ids)} films, {len(corpus.emb)} seconds, {len(units)} scenes; {len(catalogue)} ads; "
          f"{len(pool)} unique pairs to label; sheets in {base / 'sheets'}")


def score(seed: int):
    picks = json.loads((OUT / "picks.json").read_text())
    pool = json.loads((OUT / "pool.json").read_text())["pairs"]
    primary = json.loads((OUT / "labels_primary.json").read_text())["labels"]
    label = {(p["ad"], p["scene"]): primary[p["pair"]] for p in pool}
    missing = [p["pair"] for p in pool if p["pair"] not in primary]
    assert not missing, f"unlabelled pairs: {missing[:10]}"

    catalogue = ads()
    per_ad, p_at, n_at = {}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
    for ad in catalogue:
        known = sum(v for (a, _), v in label.items() if a == ad.id)
        row = {"fitting_scenes_in_pool": known}
        for m in METHODS:
            rel = [label[(ad.id, s["scene"])] for s in picks["picks"][ad.id][m]]
            row[m] = {"relevant": rel, "precision@10": precision_at_k(rel, K), "ndcg@10": ndcg_at_k(rel, known, K)}
            p_at[m].append(row[m]["precision@10"])
            n_at[m].append(row[m]["ndcg@10"])
        per_ad[ad.id] = row

    summary = {}
    for metric, table in (("precision@10", p_at), ("ndcg@10", n_at)):
        arr = {m: np.array(v) for m, v in table.items()}
        summary[metric] = {m: {"mean": float(a.mean()), "ci95": bootstrap(a, seed=seed)} for m, a in arr.items()}
        for other in ("genre", "random"):
            d = arr["clip"] - arr[other]
            lo, hi = bootstrap(d, seed=seed)
            summary[metric][f"clip-{other}"] = {"mean": float(d.mean()), "ci95": (lo, hi),
                                                "clip_beats": bool(lo > 0),
                                                "ads_clip_wins": int((d > 0).sum()), "ads_tied": int((d == 0).sum()),
                                                "ads_clip_loses": int((d < 0).sum())}

    data = {"summary": summary, "per_ad": per_ad, "pairs_labelled": len(pool),
            "fits_rate_primary": float(np.mean(list(primary.values())))}
    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p for p in pool if p["pair"] in author]
        data["agreement"] = agreement([primary[p["pair"]] for p in both], [author[p["pair"]] for p in both])
        data["agreement"]["disagreements"] = [p["pair"] for p in both if primary[p["pair"]] != author[p["pair"]]]
    config = {"protocol": "EVAL-PROTOCOL.md", "k": K, "methods": METHODS, "bootstrap_resamples": 10_000,
              "labels_primary": "results/phase4/labels_primary.json",
              "labels_author": "results/phase4/labels_author.json" if author_file.exists() else None}
    print("  ->", write_result(PHASE, "match", LANE, seed, config, data))
    for metric in ("precision@10", "ndcg@10"):
        s = summary[metric]
        print(f"{metric}: " + "  ".join(f"{m} {s[m]['mean']:.3f} [{s[m]['ci95'][0]:.3f}, {s[m]['ci95'][1]:.3f}]"
                                         for m in METHODS))
        for other in ("genre", "random"):
            d = s[f"clip-{other}"]
            print(f"   clip-{other}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  beats={d['clip_beats']}"
                  f"  ads won/tied/lost {d['ads_clip_wins']}/{d['ads_tied']}/{d['ads_clip_loses']}")
    if "agreement" in data:
        a = data["agreement"]
        print(f"agreement on {a['n']} pairs: {a['agreement']:.2f}, kappa {a['cohen_kappa']:.2f}")
    plot(summary, per_ad, OUT / "match.png")


def plot(summary: dict, per_ad: dict, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"clip": "#d62728", "genre": "#1f77b4", "random": "#7f7f7f"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    for i, metric in enumerate(("precision@10", "ndcg@10")):
        for j, m in enumerate(METHODS):
            s = summary[metric][m]
            x = i + (j - 1) * 0.25
            ax.bar(x, s["mean"], 0.23, color=colors[m], label=m if i == 0 else None)
            ax.errorbar(x, s["mean"], yerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]], color="black",
                        capsize=3)
    ax.set_xticks([0, 1], ["precision@10", "nDCG@10"])
    ax.set_ylim(0, 1)
    ax.set_title("mean over 20 ads, 95% bootstrap interval", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    names = list(per_ad)
    for j, m in enumerate(METHODS):
        ax.bar(np.arange(len(names)) + (j - 1) * 0.27, [per_ad[a][m]["precision@10"] for a in names], 0.26,
               color=colors[m], label=m)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("precision@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("per ad", fontsize=9)
    fig.suptitle("Phase 4: scenes picked by CLIP vs genre-only vs random (14 held-out films, simulated ads)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["retrieve", "score"])
    ap.add_argument("--films", choices=["test", "practice"], default="test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.step == "retrieve":
        retrieve(args.films)
    else:
        score(args.seed)


if __name__ == "__main__":
    main()
