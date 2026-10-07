"""Phase 1 quality: how much does each sampling policy cost in retrieval quality?

1. Reference: decode EVERY frame of the test windows and embed each one (cached in data/phase1/).
2. Each policy keeps a subset of those frames (same selection code as the speed benchmark).
3. Score 10-second segments against text queries (segment score = best frame in it), and compare each
   policy's top-k segments with the every-frame reference: recall@k.

Writes results/phase1/quality_<lane>_seed<seed>.json and results/phase1/sampling_curve.png.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.corpus import Window, test_windows
from scenesafe.decode.decoder import iter_frames
from scenesafe.decode.sampling import parse, select_indices, thumbnail
from scenesafe.embed.clip import DEFAULT_MODEL, DEFAULT_PRETRAINED, ClipEmbedder

PHASE = 1
CACHE = REPO_ROOT / "data" / "phase1"
POLICIES = ["all", "uniform@0.25", "uniform@0.5", "uniform@1", "uniform@2", "keyframes",
            "scene@0.03", "scene@0.05", "scene@0.08", "scene@0.12"]


def reference(w: Window, emb: ClipEmbedder, batch_size: int) -> dict:
    """Every frame in the window: time, keyframe flag, thumbnail, embedding. Cached per window."""
    path = CACHE / f"ref_{w.id}_{w.start_s:g}_{w.duration_s:g}.npz"
    if path.exists():
        return dict(np.load(path))
    times, keys, smalls, embs, buf = [], [], [], [], []

    def flush():
        embs.append(emb.embed(emb.preprocess_on_device(np.stack(buf))).numpy())
        buf.clear()

    t0 = time.perf_counter()
    for d in iter_frames(w.path, "cpu", False, w.start_s, w.duration_s):
        times.append(d.t)
        keys.append(d.is_key)
        smalls.append(thumbnail(d.frame))
        buf.append(d.frame.to_ndarray(format="rgb24"))
        if len(buf) == batch_size:
            flush()
    if buf:
        flush()
    ref = {"times": np.array(times), "is_key": np.array(keys), "smalls": np.stack(smalls),
           "emb": np.concatenate(embs).astype(np.float32)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ref)
    print(f"  {w.id}: {len(times)} frames embedded in {time.perf_counter() - t0:.0f}s")
    return ref


def segment_scores(ref_list, windows, text: np.ndarray, keep: list[np.ndarray], segment_s: float) -> np.ndarray:
    """(n_segments, n_queries) best-frame similarity per segment; -inf where the policy kept no frame."""
    rows = []
    for ref, w, idx in zip(ref_list, windows, keep):
        n_seg = int(np.ceil(w.duration_s / segment_s))
        scores = np.full((n_seg, text.shape[0]), -np.inf, dtype=np.float32)
        if len(idx):
            seg = np.minimum(((ref["times"][idx] - w.start_s) // segment_s).astype(int), n_seg - 1)
            sims = ref["emb"][idx] @ text.T
            np.maximum.at(scores, seg, sims)
        rows.append(scores)
    return np.concatenate(rows)


def recall_at_k(ref_scores: np.ndarray, scores: np.ndarray, k: int) -> float:
    per_query = []
    for q in range(ref_scores.shape[1]):
        truth = set(np.argsort(-ref_scores[:, q], kind="stable")[:k])
        got = set(np.argsort(-scores[:, q], kind="stable")[:k])
        per_query.append(len(truth & got) / k)
    return float(np.mean(per_query))


def plot(rows: list[dict], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for family, marker in (("uniform", "o"), ("scene", "s"), ("keyframes", "^"), ("all", "*")):
        pts = sorted((r for r in rows if r["policy"].startswith(family)), key=lambda r: r["kept_per_hour"])
        if pts:
            ax.plot([r["kept_per_hour"] for r in pts], [r["recall@10"] for r in pts], marker=marker,
                    label=family, linestyle="-" if len(pts) > 1 else "none")
            for r in pts:
                ax.annotate(r["policy"].split("@")[-1] if "@" in r["policy"] else "", (r["kept_per_hour"], r["recall@10"]),
                            textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("frames kept per hour of video (log scale)")
    ax.set_ylabel("recall@10 vs every-frame reference")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Sampling policy: frames kept vs retrieval quality")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-s", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    windows = test_windows()
    queries = yaml.safe_load((REPO_ROOT / "configs" / "phase1_queries.yaml").read_text())["queries"]
    emb = ClipEmbedder(args.device)

    print("reference (every frame):")
    refs = [reference(w, emb, args.batch_size) for w in windows]
    text = emb.embed_text(queries).numpy()
    hours = sum(w.duration_s for w in windows) / 3600

    all_idx = [np.arange(len(r["times"])) for r in refs]
    ref_scores = segment_scores(refs, windows, text, all_idx, args.segment_s)

    rows = []
    for spec in POLICIES:
        keep = [select_indices(parse(spec), r["times"], r["is_key"], r["smalls"]) for r in refs]
        scores = segment_scores(refs, windows, text, keep, args.segment_s)
        n = sum(len(k) for k in keep)
        rows.append({"policy": spec, "kept_frames": n, "kept_per_hour": n / hours,
                     "kept_fraction": n / sum(len(r["times"]) for r in refs),
                     "segment_coverage": float(np.isfinite(scores[:, 0]).mean()),
                     "recall@10": recall_at_k(ref_scores, scores, 10),
                     "recall@50": recall_at_k(ref_scores, scores, 50)})
        r = rows[-1]
        print(f"  {spec:13s} kept/h {r['kept_per_hour']:8.0f}  coverage {r['segment_coverage']:.3f}  "
              f"R@10 {r['recall@10']:.3f}  R@50 {r['recall@50']:.3f}")

    config = {"windows": [{"id": w.id, "genre": w.genre, "start_s": w.start_s, "duration_s": w.duration_s}
                          for w in windows],
              "hours": hours, "segment_s": args.segment_s, "n_segments": int(ref_scores.shape[0]),
              "queries": queries, "model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED,
              "preprocess": "on-device (preprocess_on_device)", "device": args.device, "decode": "cpu, every frame"}
    frames = {w.id: int(len(r["times"])) for w, r in zip(windows, refs)}
    lane = "gpu-apple-m5" if args.device == "mps" else "cpu-laptop"
    print("  ->", write_result(PHASE, "quality", lane, args.seed, config, {"frames_per_window": frames, "policies": rows}))
    out = REPO_ROOT / "results" / "phase1" / "sampling_curve.png"
    plot(rows, out)
    print("  ->", out)


if __name__ == "__main__":
    main()
