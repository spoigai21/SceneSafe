"""Phase 4d: CLIP ViT-B/32 vs SigLIP2 B/16, same method, on 14 fresh films and 60 ads, with a label guide.
Rules: EVAL-PROTOCOL.md, "Phase 4d amendment".

  embed   for each film in corpus/manifest_4d.json: ViT-B/32 embeddings exactly as in Phase 2 (data/embeddings/),
          SigLIP2 embeddings at the same seconds (data/embeddings_siglip2/), and per-second brightness (data/luma/).
          Also both models' embeddings of the 60 briefs (data/phase4d/). No results.
  test    both models run the v1 method once (exact search over every second, each scene once at its best second,
          top 10 per ad) on the Phase 3 scenes of the new films. Every picked (ad, scene) pair goes into one shuffled
          pool with labelling sheets. Writes results/phase4d/{picks,pool}.json.
  score   ViT-B/32 vs SigLIP2; writes results/phase4d/match_model_cpu-laptop_seed0.json and a chart.
"""

import argparse
import json

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import ALT_MODEL, ALT_PRETRAINED, DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import luma, v2
from scenesafe.match.catalogue import ads_4d, label_guide
from scenesafe.match.display import frame_strip, sheets
from scenesafe.match.metrics import agreement, bootstrap, ndcg_at_k, precision_at_k

PHASE = 4
LANE = "cpu-laptop"
MANIFEST = REPO_ROOT / "corpus" / "manifest_4d.json"
OUT = REPO_ROOT / "results" / "phase4d"
DATA = REPO_ROOT / "data" / "phase4d"
FRAMES = DATA / "frames"
ALT_DIR = REPO_ROOT / "data" / "embeddings_siglip2"
SCENE_THRESHOLD = 0.75
K = 10
BATCH = 64
MODELS = ("b32", "siglip2")
NAMES = {"b32": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}", "siglip2": f"{ALT_MODEL} {ALT_PRETRAINED}"}


def films() -> list[str]:
    return sorted(t["id"] for t in json.loads(MANIFEST.read_text())["titles"])


def embed():
    import open_clip
    import torch
    from PIL import Image
    from scripts.embed_corpus import OUT_DIR, batches, embed_title
    from scenesafe.corpus import video_path
    from scenesafe.embed.clip import ClipEmbedder

    briefs = [a.brief for a in ads_4d()]
    DATA.mkdir(parents=True, exist_ok=True)
    b32 = ClipEmbedder("mps")
    np.save(DATA / "briefs_b32.npy", b32.embed_text(briefs).numpy().astype(np.float32))   # fp32, as in Phase 4
    b32.model.half()                                                                         # images: Phase 2 config
    dev = torch.device("mps")
    alt, _, prep = open_clip.create_model_and_transforms(ALT_MODEL, pretrained=ALT_PRETRAINED)
    alt = alt.eval().half().to(dev)
    with torch.inference_mode():
        t = alt.encode_text(open_clip.get_tokenizer(ALT_MODEL)(briefs).to(dev)).float()
    np.save(DATA / "briefs_siglip2.npy", (t / t.norm(dim=-1, keepdim=True)).cpu().numpy())

    ALT_DIR.mkdir(parents=True, exist_ok=True)
    for tid in films():
        ref = OUT_DIR / f"{tid}.npz"
        if not ref.exists():                                  # Phase 2 configuration, unchanged
            times, e = embed_title(b32, video_path(tid))
            np.savez(ref, times=times, emb=e)
        out = ALT_DIR / f"{tid}.npz"
        if not out.exists():
            times, embs = [], []
            with torch.inference_mode():
                for ts, frames in batches(video_path(tid), BATCH):
                    x = torch.stack([prep(Image.fromarray(f)) for f in frames]).half().to(dev)
                    e = alt.encode_image(x).float()
                    embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
                    times += ts
            r = np.load(ref)["times"]
            assert len(r) == len(times) and np.allclose(r, times), f"seconds differ between models: {tid}"
            np.savez(out, times=np.array(times), emb=np.concatenate(embs))
        luma.compute(tid)
        print(f"  {tid}: {len(np.load(ref)['times'])} seconds", flush=True)


def load_alt(ids: list[str]) -> np.ndarray:
    e = np.concatenate([np.load(ALT_DIR / f"{tid}.npz")["emb"].astype(np.float32) for tid in ids])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def describe(corpus, units, u: int) -> dict:
    title = corpus.ids[units.title[u]]
    return {"scene": f"{title}@{units.start[u]:.0f}", "title": title,
            "start_s": round(float(units.start[u]), 2), "end_s": round(float(units.end[u]), 2)}


def test():
    catalogue = ads_4d()
    ids = films()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)                   # Phase 3 detector, on the ViT-B/32 embeddings
    embs = {"b32": corpus.emb, "siglip2": load_alt(ids)}
    no_junk = np.zeros(len(corpus.emb), bool)
    picks, pairs = {m: {} for m in MODELS}, {}
    for m in MODELS:
        q = np.load(DATA / f"briefs_{m}.npy")
        assert len(q) == len(catalogue) and len(embs[m]) == len(corpus.emb)
        ss = v2.scene_scores(embs[m] @ q.T, no_junk, units.first)    # each scene at its best second
        for i, ad in enumerate(catalogue):
            chosen = v2.pick(ss[:, i], K, None)
            picks[m][ad.id] = [{**describe(corpus, units, u), "score": float(ss[u, i])} for u in chosen]
            for u in chosen:
                d = describe(corpus, units, u)
                pairs.setdefault((ad.id, d["scene"]), d)

    by_ad = {a.id: a for a in catalogue}
    keys = list(pairs)
    order = np.random.default_rng(0).permutation(len(keys))
    pool = [{"pair": f"D{n:04d}", "ad": keys[j][0], "category": by_ad[keys[j][0]].category,
             "brief": by_ad[keys[j][0]].brief, **pairs[keys[j]]} for n, j in enumerate(order, 1)]
    FRAMES.mkdir(parents=True, exist_ok=True)
    for key, p in {p["scene"]: p for p in pool}.items():
        if not (FRAMES / f"{key}.jpg").exists():
            frame_strip(p["title"], p["start_s"], p["end_s"], FRAMES / f"{key}.jpg")
    sheets(pool, FRAMES, DATA / "sheets", label_guide())
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"films": ids, "ads": len(catalogue), "scene_threshold": SCENE_THRESHOLD, "seconds": int(len(corpus.emb)),
            "scenes": int(len(units)), "models": NAMES}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {len(ids)} films, {len(corpus.emb)} seconds, {len(units)} scenes; {len(catalogue)} ads; "
          f"{len(pool)} unique pairs to label; sheets in {DATA / 'sheets'}")


def compare(picks, label, seed):
    per_ad, cols = {}, {f"{s}_{m}": [] for s in ("p", "n") for m in MODELS}
    for ad in ads_4d():
        known = sum(v for (a, _), v in label.items() if a == ad.id)
        row = {"fitting_scenes_known": known}
        for m in MODELS:
            rel = [label[(ad.id, s["scene"])] for s in picks[m][ad.id]]
            row[m] = {"relevant": rel, "precision@10": precision_at_k(rel, K), "ndcg@10": ndcg_at_k(rel, known, K)}
            cols[f"p_{m}"].append(row[m]["precision@10"])
            cols[f"n_{m}"].append(row[m]["ndcg@10"])
        per_ad[ad.id] = row
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for metric, s in (("precision@10", "p"), ("ndcg@10", "n")):
        a, b = arr[f"{s}_b32"], arr[f"{s}_siglip2"]
        d = b - a
        ci = bootstrap(d, seed=seed)
        summary[metric] = {"b32": {"mean": float(a.mean()), "ci95": bootstrap(a, seed=seed)},
                           "siglip2": {"mean": float(b.mean()), "ci95": bootstrap(b, seed=seed)},
                           "siglip2-b32": {"mean": float(d.mean()), "ci95": ci, "improves": bool(ci[0] > 0),
                                           "ads_better": int((d > 0).sum()), "ads_same": int((d == 0).sum()),
                                           "ads_worse": int((d < 0).sum())}}
    return summary, per_ad


def mostly_junk(model_picks: dict) -> int:
    """Picks whose seconds are mostly near-black or blank by the Phase 4b pixel rule (descriptive, not a decision)."""
    ids = films()
    corpus = load_corpus(ids)
    is_junk = v2.junk_mask(*luma.load(ids))
    n = 0
    for scenes_ in model_picks.values():
        for s in scenes_:
            sel = (corpus.title == ids.index(s["title"])) & (corpus.time >= s["start_s"]) & (corpus.time < s["end_s"])
            n += is_junk[sel].mean() > 0.5
    return int(n)


def score(seed: int):
    picks = json.loads((OUT / "picks.json").read_text())["picks"]
    pool = json.loads((OUT / "pool.json").read_text())["pairs"]
    primary = json.loads((OUT / "labels_primary.json").read_text())["labels"]
    missing = [p["pair"] for p in pool if p["pair"] not in primary]
    assert not missing, f"unlabelled pairs: {missing[:10]}"
    label = {(p["ad"], p["scene"]): primary[p["pair"]] for p in pool}
    summary, per_ad = compare(picks, label, seed)
    data = {"summary": summary, "per_ad": per_ad, "pairs_labelled": len(pool),
            "mostly_junk_picks": {m: mostly_junk(picks[m]) for m in MODELS} | {"of": K * len(per_ad)}}

    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p for p in pool if p["pair"] in author]
        data["agreement"] = agreement([primary[p["pair"]] for p in both], [author[p["pair"]] for p in both])
        data["agreement"]["disagreements"] = [p["pair"] for p in both if primary[p["pair"]] != author[p["pair"]]]
        mixed = label | {(p["ad"], p["scene"]): author[p["pair"]] for p in both}
        data["sensitivity_author_labels_where_present"] = compare(picks, mixed, seed)[0]["precision@10"]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4d amendment)", "k": K, "bootstrap_resamples": 10_000,
              "models": NAMES, "author_sample": {"size": 100, "seed": 3}}
    print("  ->", write_result(PHASE, "match_model_4d", LANE, seed, config, data)
          .rename(OUT / f"match_model_{LANE}_seed{seed}.json"))
    for metric in ("precision@10", "ndcg@10"):
        s = summary[metric]
        d = s["siglip2-b32"]
        print(f"{metric}: ViT-B/32 {s['b32']['mean']:.3f}  SigLIP2 {s['siglip2']['mean']:.3f}  diff {d['mean']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] improves={d['improves']}  "
              f"ads better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("mostly near-black picks:", data["mostly_junk_picks"])
    if "agreement" in data:
        a = data["agreement"]
        print(f"agreement on {a['n']} pairs: {a['agreement']:.2f}, kappa {a['cohen_kappa']:.2f}")
        s = data["sensitivity_author_labels_where_present"]
        print(f"sensitivity: ViT-B/32 {s['b32']['mean']:.3f}  SigLIP2 {s['siglip2']['mean']:.3f}  "
              f"diff {s['siglip2-b32']['mean']:+.3f} [{s['siglip2-b32']['ci95'][0]:+.3f}, {s['siglip2-b32']['ci95'][1]:+.3f}]")
    plot(per_ad, summary, OUT / "match_model.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, ax = plt.subplots(figsize=(18, 4.8))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [per_ad[a]["b32"]["precision@10"] for a in names], 0.4, color="#9e9e9e", label="CLIP ViT-B/32")
    ax.bar(x + 0.2, [per_ad[a]["siglip2"]["precision@10"] for a in names], 0.4, color="#1f77b4", label="SigLIP2 B/16")
    ax.set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("precision@10")
    s = summary["precision@10"]
    ax.set_title(f"Phase 4d, 14 fresh films, 60 ads: {s['b32']['mean']:.2f} vs {s['siglip2']['mean']:.2f}", fontsize=9)
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
