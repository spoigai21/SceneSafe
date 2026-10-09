"""Phase 2: what does fast inference cost in quality?

Same model (CLIP ViT-B/32 image encoder), several inference configurations (runtime x precision x device).
  1. Inputs: Phase 1 test windows sampled at 1 fps, resized/normalised ONCE, so every configuration sees
     identical pixels. Calibration frames for int8 come from OTHER titles.
  2. Accuracy: embed all test frames with each configuration; compare with the reference (pytorch-fp32-mps):
     cosine similarity per frame, and recall@k of the 20 Phase 1 searches (10-second segments).
  3. Speed: frames/sec per configuration and batch size, on a fixed timing subset, 3 repeats after a warm-up pass.
Writes results/phase2/inference_<lane>_seed<seed>.json and results/phase2/frontier.png.
"""

import argparse
import json
import statistics
from collections import defaultdict

import numpy as np
import torch
import yaml

from scenesafe.common.results import REPO_ROOT, Timer, load_avg, write_result
from scenesafe.corpus import is_downloaded, test_windows, titles, video_path
from scenesafe.decode.sampling import Uniform, sample
from scenesafe.embed.clip import DEFAULT_MODEL, DEFAULT_PRETRAINED, ClipEmbedder
from scenesafe.embed.runtimes import BY_NAME, CONFIGS, REFERENCE, Runner

PHASE = 2
CACHE = REPO_ROOT / "data" / "phase2"
SEGMENT_S = 10.0


def preprocess_windows(emb: ClipEmbedder, specs) -> tuple[np.ndarray, list]:
    """Decode (CPU, uniform 1 fps), resize+normalise on MPS. Stored as float16 to halve memory (same for all configs)."""
    pix, meta = [], []
    for title_id, start, dur, fps in specs:
        frames, times = [], []
        for t, f in sample(video_path(title_id), Uniform(fps), "cpu", start, dur):
            frames.append(f)
            times.append(t)
        for i in range(0, len(frames), 128):
            pix.append(emb.preprocess_on_device(np.stack(frames[i:i + 128])).cpu().numpy().astype(np.float16))
        meta += [(title_id, start, t) for t in times]
    return np.concatenate(pix), meta


def inputs(emb: ClipEmbedder):
    CACHE.mkdir(parents=True, exist_ok=True)
    test_p, cal_p = CACHE / "pixels_test.npy", CACHE / "pixels_calibration.npy"
    if not test_p.exists():
        specs = [(w.id, w.start_s, w.duration_s, 1.0) for w in test_windows()]
        pix, meta = preprocess_windows(emb, specs)
        np.save(test_p, pix)
        np.save(CACHE / "meta_test.npy", np.array(meta, dtype=object), allow_pickle=True)
    if not cal_p.exists():
        # one title per genre that is NOT in the test set and is downloaded; 48 frames each (1 per 20 s from 20:00)
        test_ids, seen, specs = {w.id for w in test_windows()}, set(), []
        for t in titles():
            if t["id"] not in test_ids and t["genre"] not in seen and is_downloaded(t):
                seen.add(t["genre"])
                specs.append((t["id"], 1200.0, 960.0, 0.05))
        pix, _ = preprocess_windows(emb, specs)
        np.save(cal_p, pix)
        (CACHE / "calibration_titles.txt").write_text("\n".join(s[0] for s in specs) + "\n")
    meta = np.load(CACHE / "meta_test.npy", allow_pickle=True)
    return np.load(test_p), np.load(cal_p), meta


def segment_scores(emb: np.ndarray, meta, text: np.ndarray) -> np.ndarray:
    """Best-frame similarity per (title, 10-second segment) for each query."""
    keys = [(m[0], int((float(m[2]) - float(m[1])) // SEGMENT_S)) for m in meta]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    seg = np.array([index[k] for k in keys])
    scores = np.full((len(index), text.shape[0]), -np.inf, np.float32)
    np.maximum.at(scores, seg, emb @ text.T)
    return scores


def recall_at_k(ref: np.ndarray, got: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(np.argsort(-ref[:, q], kind="stable")[:k]) &
                              set(np.argsort(-got[:, q], kind="stable")[:k])) / k for q in range(ref.shape[1])]))


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def accuracy(pixels, calibration, meta, text, batch):
    embs = {}
    for cfg in CONFIGS:
        path = CACHE / f"emb_{cfg.name}.npy"
        if not path.exists():
            runner = Runner(cfg, batch, calibration.astype(np.float32))
            out = np.concatenate([runner.embed_all(pixels[i:i + 1024].astype(np.float32))
                                  for i in range(0, len(pixels), 1024)])
            np.save(path, out)
        embs[cfg.name] = unit(np.load(path).astype(np.float32))
    ref = embs[REFERENCE]
    ref_seg = segment_scores(ref, meta, text)
    rows = {}
    for name, e in embs.items():
        cos = (e * ref).sum(1)
        seg = segment_scores(e, meta, text)
        rows[name] = {"cosine_mean": float(cos.mean()), "cosine_p1": float(np.percentile(cos, 1)),
                      "cosine_min": float(cos.min()), "recall@10": recall_at_k(ref_seg, seg, 10),
                      "recall@50": recall_at_k(ref_seg, seg, 50)}
        r = rows[name]
        print(f"  {name:28s} cos mean {r['cosine_mean']:.6f} min {r['cosine_min']:.5f}  "
              f"R@10 {r['recall@10']:.3f}  R@50 {r['recall@50']:.3f}", flush=True)
    return rows


def speed(pixels, calibration, batches, repeats, n_frames):
    timing = pixels[:n_frames].astype(np.float32)
    rows = defaultdict(dict)
    for cfg in CONFIGS:
        for b in batches:
            runner = Runner(cfg, b, calibration.astype(np.float32))
            chunks = [timing[i:i + b] for i in range(0, len(timing) - b + 1, b)]  # whole batches only
            n = len(chunks) * b
            for c in chunks:  # warm-up: one full untimed pass
                runner.run(c)
            runner.sync()
            fps, xfer = [], []
            for _ in range(repeats):
                with Timer() as t:
                    for c in chunks:
                        runner.run(c)
                    runner.sync()
                fps.append(n / t.seconds)
                with Timer() as tx:
                    for c in chunks:
                        runner.transfer_only(c)
                xfer.append(tx.seconds / t.seconds)
            rows[cfg.name][b] = {"fps": {"runs": fps, "median": statistics.median(fps), "min": min(fps), "max": max(fps)},
                                 "transfer_share": statistics.median(xfer) if cfg.device == "mps" else None}
            print(f"  {cfg.name:28s} batch {b:4d}  {statistics.median(fps):7.1f} fps ({min(fps):.1f}-{max(fps):.1f})",
                  flush=True)
            del runner
    return rows


def plot(acc, spd, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"cpu-laptop": ("o", "#1f77b4"), "gpu-apple-m5": ("s", "#d62728"), "ane-apple-m5": ("^", "#2ca02c")}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    points = sorted(((max(v["fps"]["median"] for v in spd[n].values()), a["recall@10"], n) for n, a in acc.items()))
    top = 0
    for fps, rec, name in points:
        marker, color = style[BY_NAME[name].lane]
        ax.scatter(fps, rec, marker=marker, color=color, s=60, zorder=3)
        dy = 6
        if rec > 0.9:  # stagger labels of the points that sit together at the top
            dy, top = -14 - 11 * (top % 3), top + 1
        ax.annotate(f"{name} ({fps:.0f})", (fps, rec), textcoords="offset points", xytext=(4, dy), fontsize=7)
    for lane, (marker, color) in style.items():
        ax.scatter([], [], marker=marker, color=color, label=lane)
    ax.legend(loc="center right", title="lane", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("frames/sec at best batch size (log scale)")
    ax.set_ylabel("recall@10 vs pytorch-fp32-mps (20 searches)")
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.set_title("Same model, different inference configurations (Apple M5 laptop)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_from_results(seed: int):
    acc, spd = {}, {}
    for f in (REPO_ROOT / "results" / "phase2").glob(f"inference_*_seed{seed}.json"):
        for name, row in json.loads(f.read_text())["configs"].items():
            acc[name], spd[name] = row["accuracy"], row["speed"]
    plot(acc, spd, REPO_ROOT / "results" / "phase2" / "frontier.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--accuracy-batch", type=int, default=32)
    ap.add_argument("--timing-frames", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true", help="redraw frontier.png from saved results")
    args = ap.parse_args()
    if args.plot_only:
        plot_from_results(args.seed)
        return
    torch.manual_seed(args.seed)

    emb = ClipEmbedder("mps")
    pixels, calibration, meta = inputs(emb)
    queries = yaml.safe_load((REPO_ROOT / "configs" / "phase1_queries.yaml").read_text())["queries"]
    text = emb.embed_text(queries).numpy()
    del emb
    print(f"inputs: {len(pixels)} test frames, {len(calibration)} calibration frames")

    print("accuracy (all test frames, vs reference):")
    acc = accuracy(pixels, calibration, meta, text, args.accuracy_batch)
    print("speed:")
    load_before = load_avg()
    spd = speed(pixels, calibration, args.batches, args.repeats, args.timing_frames)
    load_after = load_avg()

    config = {"model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED, "reference": REFERENCE,
              "test_frames": int(len(pixels)), "calibration_frames": int(len(calibration)),
              "calibration_titles": (CACHE / "calibration_titles.txt").read_text().split(),
              "input": "Phase 1 test windows, uniform 1 fps, preprocess_on_device, stored float16, fed as float32",
              "timing_frames": args.timing_frames, "repeats": args.repeats, "batches": args.batches,
              "accuracy_batch": args.accuracy_batch, "segment_s": SEGMENT_S,
              "torch_version": torch.__version__, "onnxruntime_version": __import__("onnxruntime").__version__}
    by_lane = defaultdict(dict)
    for cfg in CONFIGS:
        by_lane[cfg.lane][cfg.name] = {"runtime": cfg.runtime, "precision": cfg.precision, "device": cfg.device,
                                       "accuracy": acc[cfg.name], "speed": spd[cfg.name]}
    for lane, rows in by_lane.items():
        print("  ->", write_result(PHASE, "inference", lane, args.seed, config,
                                   {"configs": rows, "load_avg_before": load_before, "load_avg_after": load_after}))
    out = REPO_ROOT / "results" / "phase2" / "frontier.png"
    plot(acc, spd, out)
    print("  ->", out)


if __name__ == "__main__":
    main()
