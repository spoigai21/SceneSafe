"""Phase 4c: does a stronger image-text model (SigLIP2 B/16) pick better scenes than CLIP ViT-B/32, with the matching
method held fixed? Rules: EVAL-PROTOCOL.md, "Phase 4c amendment".

  embed   embeds the test films at the same 1 fps seconds as the corpus embeddings, and the 20 briefs, with the
          alternative model. Writes data/embeddings_siglip2/<title>.npz and data/phase4c/briefs.npy. No results.
  test    runs the v1 method (exact search over every second, each scene once at its best second, top 10) with the
          alternative model on the Phase 3 scenes. Pairs not labelled in Phase 4 or 4b go into a new shuffled pool.
  score   v1 (B/32, Phase 4) vs SigLIP2 on the test films; writes results/phase4c/match_model_cpu-laptop_seed0.json
          and a chart.
"""

import argparse
import json

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import ALT_MODEL, ALT_PRETRAINED, DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import luma, v2
from scenesafe.match.catalogue import ads, test_films
from scenesafe.match.display import frame_strip, sheets
from scenesafe.match.labels import author_sample
from scenesafe.match.metrics import agreement, bootstrap, ndcg_at_k, precision_at_k

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4c"
P4 = REPO_ROOT / "results" / "phase4"
P4B = REPO_ROOT / "results" / "phase4b"
DATA = REPO_ROOT / "data" / "phase4c"
EMB_DIR = REPO_ROOT / "data" / "embeddings_siglip2"
FRAMES = REPO_ROOT / "data" / "phase4" / "test" / "frames"    # shared, so seen scenes reuse strips
SCENE_THRESHOLD = 0.75
K = 10
BATCH = 64
AUTHOR_SAMPLE = 50
AUTHOR_SEED = 2


def embed():
    import open_clip
    import torch
    from PIL import Image
    from scripts.embed_corpus import batches
    from scenesafe.corpus import video_path

    dev = torch.device("mps")
    model, _, prep = open_clip.create_model_and_transforms(ALT_MODEL, pretrained=ALT_PRETRAINED)
    model = model.eval().half().to(dev)
    with torch.inference_mode():
        t = model.encode_text(open_clip.get_tokenizer(ALT_MODEL)([a.brief for a in ads()]).to(dev)).float()
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "briefs.npy", (t / t.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for tid in test_films():
        out = EMB_DIR / f"{tid}.npz"
        if out.exists():
            print(f"  {tid}: cached")
            continue
        times, embs = [], []
        with torch.inference_mode():
            for ts, frames in batches(video_path(tid), BATCH):
                x = torch.stack([prep(Image.fromarray(f)) for f in frames]).half().to(dev)
                e = model.encode_image(x).float()
                embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
                times += ts
        ref = np.load(REPO_ROOT / "data" / "embeddings" / f"{tid}.npz")["times"]
        assert len(ref) == len(times) and np.allclose(ref, times), f"seconds differ from the corpus: {tid}"
        np.savez(out, times=np.array(times), emb=np.concatenate(embs))
        print(f"  {tid}: {len(times)} seconds", flush=True)


def load_alt(ids: list[str]) -> np.ndarray:
    e = np.concatenate([np.load(EMB_DIR / f"{tid}.npz")["emb"].astype(np.float32) for tid in ids])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def describe(corpus, units, u: int) -> dict:
    title = corpus.ids[units.title[u]]
    return {"scene": f"{title}@{units.start[u]:.0f}", "title": title,
            "start_s": round(float(units.start[u]), 2), "end_s": round(float(units.end[u]), 2)}


def labelled() -> set[tuple[str, str]]:
    return {(p["ad"], p["scene"]) for d in (P4, P4B) for p in json.loads((d / "pool.json").read_text())["pairs"]}


def test():
    catalogue = ads()
    ids = test_films()
    corpus = load_corpus(ids)                       # B/32: only for the Phase 3 scenes, unchanged
    units = scenes(corpus, SCENE_THRESHOLD)
    emb = load_alt(ids)
    assert len(emb) == len(corpus.emb)
    q = np.load(DATA / "briefs.npy")
    assert len(q) == len(catalogue)
    ss = v2.scene_scores(emb @ q.T, np.zeros(len(emb), bool), units.first)   # each scene at its best second

    seen = labelled()
    picks, new = {}, []
    for i, ad in enumerate(catalogue):
        picks[ad.id] = [{**describe(corpus, units, u), "score": float(ss[u, i])} for u in v2.pick(ss[:, i], K, None)]
        new += [(ad.id, {k: d[k] for k in ("scene", "title", "start_s", "end_s")})
                for d in picks[ad.id] if (ad.id, d["scene"]) not in seen]
    by_ad = {a.id: a for a in catalogue}
    order = np.random.default_rng(0).permutation(len(new))
    pool = [{"pair": f"S{n:03d}", "ad": new[j][0], "category": by_ad[new[j][0]].category,
             "brief": by_ad[new[j][0]].brief, **new[j][1]} for n, j in enumerate(order, 1)]
    FRAMES.mkdir(parents=True, exist_ok=True)
    for key, p in {p["scene"]: p for p in pool}.items():
        if not (FRAMES / f"{key}.jpg").exists():
            frame_strip(p["title"], p["start_s"], p["end_s"], FRAMES / f"{key}.jpg")
    sheets(pool, FRAMES, DATA / "test" / "sheets")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"films": "test", "titles": ids, "scene_threshold": SCENE_THRESHOLD, "seconds": int(len(emb)),
            "scenes": int(len(units)), "model": ALT_MODEL, "pretrained": ALT_PRETRAINED,
            "scenes_from": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED} (Phase 3)"}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {sum(len(v) for v in picks.values())} scenes picked; {len(pool)} new pairs to label; "
          f"sheets in {DATA / 'test' / 'sheets'}")


def labels(kind: str) -> dict[tuple[str, str], int]:
    """(ad, scene) -> label from every phase's pool; kind 'primary' or 'author' (author covers only a sample)."""
    out = {}
    for d in (P4, P4B, OUT):
        f = d / f"labels_{kind}.json"
        if f.exists():
            lab = json.loads(f.read_text())["labels"]
            out |= {(p["ad"], p["scene"]): lab[p["pair"]]
                    for p in json.loads((d / "pool.json").read_text())["pairs"] if p["pair"] in lab}
    return out


def compare(v1_picks, alt_picks, label, seed):
    per_ad, cols = {}, {k: [] for k in ("p_v1", "p_alt", "n_v1", "n_alt")}
    for ad in ads():
        known = sum(v for (a, _), v in label.items() if a == ad.id)
        r1 = [label[(ad.id, s["scene"])] for s in v1_picks[ad.id]["clip"]]
        r2 = [label[(ad.id, s["scene"])] for s in alt_picks[ad.id]]
        per_ad[ad.id] = {"fitting_scenes_known": known, "v1": r1, "alt": r2,
                         "precision@10": {"v1": precision_at_k(r1, K), "alt": precision_at_k(r2, K)},
                         "ndcg@10": {"v1": ndcg_at_k(r1, known, K), "alt": ndcg_at_k(r2, known, K)}}
        for m, short in (("precision@10", "p"), ("ndcg@10", "n")):
            cols[f"{short}_v1"].append(per_ad[ad.id][m]["v1"])
            cols[f"{short}_alt"].append(per_ad[ad.id][m]["alt"])
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for metric, short in (("precision@10", "p"), ("ndcg@10", "n")):
        a, b = arr[f"{short}_v1"], arr[f"{short}_alt"]
        d = b - a
        ci = bootstrap(d, seed=seed)
        summary[metric] = {"v1": {"mean": float(a.mean()), "ci95": bootstrap(a, seed=seed)},
                           "alt": {"mean": float(b.mean()), "ci95": bootstrap(b, seed=seed)},
                           "alt-v1": {"mean": float(d.mean()), "ci95": ci, "improves": bool(ci[0] > 0),
                                      "ads_better": int((d > 0).sum()), "ads_same": int((d == 0).sum()),
                                      "ads_worse": int((d < 0).sum())}}
    return summary, per_ad


def mostly_junk(picks: dict) -> int:
    """Picks whose seconds are mostly near-black or blank by the Phase 4b pixel rule (descriptive, not a decision)."""
    ids = test_films()
    corpus = load_corpus(ids)
    is_junk = v2.junk_mask(*luma.load(ids))
    n = 0
    for scenes_ in picks.values():
        for s in scenes_:
            sel = (corpus.title == ids.index(s["title"])) & (corpus.time >= s["start_s"]) & (corpus.time < s["end_s"])
            n += is_junk[sel].mean() > 0.5
    return int(n)


def score(seed: int):
    v1_picks = json.loads((P4 / "picks.json").read_text())["picks"]
    alt_picks = json.loads((OUT / "picks.json").read_text())["picks"]
    pool = json.loads((OUT / "pool.json").read_text())["pairs"]
    primary = labels("primary")
    missing = [p["pair"] for p in pool if (p["ad"], p["scene"]) not in primary]
    assert not missing, f"unlabelled pairs: {missing[:10]}"
    summary, per_ad = compare(v1_picks, alt_picks, primary, seed)
    data = {"summary": summary, "per_ad": per_ad, "new_pairs_labelled": len(pool),
            "mostly_junk_picks": {"v1": mostly_junk({a: v["clip"] for a, v in v1_picks.items()}),
                                  "alt": mostly_junk(alt_picks), "of": 200}}

    author_file = OUT / "labels_author.json"
    if author_file.exists():
        own = json.loads((OUT / "labels_primary.json").read_text())["labels"]
        author = json.loads(author_file.read_text())["labels"]
        both = [p for p in pool if p["pair"] in author]
        data["agreement"] = agreement([own[p["pair"]] for p in both], [author[p["pair"]] for p in both])
        data["agreement"]["disagreements"] = [p["pair"] for p in both if own[p["pair"]] != author[p["pair"]]]
        sens, _ = compare(v1_picks, alt_picks, primary | labels("author"), seed)
        data["sensitivity_author_labels_where_present"] = sens["precision@10"]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4c amendment)", "k": K, "bootstrap_resamples": 10_000,
              "v1": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}", "alt": f"{ALT_MODEL} {ALT_PRETRAINED}",
              "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    print("  ->", write_result(PHASE, "match_model", LANE, seed, config, data)
          .rename(OUT / f"match_model_{LANE}_seed{seed}.json"))
    for metric in ("precision@10", "ndcg@10"):
        s = summary[metric]
        d = s["alt-v1"]
        print(f"{metric}: v1 {s['v1']['mean']:.3f}  SigLIP2 {s['alt']['mean']:.3f}  diff {d['mean']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] improves={d['improves']}  "
              f"ads better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("mostly near-black picks:", data["mostly_junk_picks"])
    if "agreement" in data:
        a = data["agreement"]
        print(f"agreement on {a['n']} pairs: {a['agreement']:.2f}, kappa {a['cohen_kappa']:.2f}")
        s = data["sensitivity_author_labels_where_present"]
        print(f"sensitivity: v1 {s['v1']['mean']:.3f}  SigLIP2 {s['alt']['mean']:.3f}  diff {s['alt-v1']['mean']:+.3f} "
              f"[{s['alt-v1']['ci95'][0]:+.3f}, {s['alt-v1']['ci95'][1]:+.3f}]")
    plot(per_ad, summary, OUT / "match_model.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [per_ad[a]["precision@10"]["v1"] for a in names], 0.4, color="#9e9e9e", label="CLIP ViT-B/32 (Phase 4)")
    ax.bar(x + 0.2, [per_ad[a]["precision@10"]["alt"] for a in names], 0.4, color="#1f77b4", label="SigLIP2 B/16 (4c)")
    ax.set_xticks(x, names, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("precision@10")
    s = summary["precision@10"]
    ax.set_title(f"Same method, different model: {s['v1']['mean']:.2f} vs {s['alt']['mean']:.2f}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "test", "score"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"embed": embed, "test": test}.get(args.step, lambda: score(args.seed))()


if __name__ == "__main__":
    main()
