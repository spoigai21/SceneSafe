"""Embed the whole corpus with the chosen inference configuration (Phase 2 decision).

Configuration: CPU decode at uniform 1 fps (Phase 1), on-device preprocessing, CLIP ViT-B/32 in PyTorch fp16 on MPS,
batch 128. Decode runs in a background thread so the CPU decodes the next batch while the GPU embeds the current one.

Output: data/embeddings/<title id>.npz with `times` (s) and `emb` (unit-normalised, float16, N x 512).
Resumable: titles already embedded are skipped. Also spot-checks fp16 against fp32 on a few titles.
Writes results/phase2/corpus_embed_gpu-apple-m5_seed<seed>.json.
"""

import argparse
import queue
import threading

import numpy as np
import torch

from scenesafe.common.results import REPO_ROOT, Timer, load_avg, write_result
from scenesafe.corpus import titles, video_path
from scenesafe.decode.sampling import Uniform, sample
from scenesafe.embed.clip import DEFAULT_MODEL, DEFAULT_PRETRAINED, ClipEmbedder

PHASE = 2
OUT_DIR = REPO_ROOT / "data" / "embeddings"
FPS = 1.0
BATCH = 128


def batches(path, batch):
    """Background-thread decoder: yields (times, stacked RGB frames) batches."""
    q: queue.Queue = queue.Queue(maxsize=4)

    def produce():
        try:
            times, frames = [], []
            for t, f in sample(path, Uniform(FPS), "cpu"):
                times.append(t)
                frames.append(f)
                if len(frames) == batch:
                    q.put((times, np.stack(frames)))
                    times, frames = [], []
            if frames:
                q.put((times, np.stack(frames)))
        except Exception as e:  # surface decode errors in the main thread
            q.put(e)
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


@torch.inference_mode()
def embed_title(emb: ClipEmbedder, path) -> tuple[np.ndarray, np.ndarray]:
    times, out = [], []
    for t, frames in batches(path, BATCH):
        x = emb.preprocess_on_device(frames).half()
        e = emb.model.encode_image(x).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
        times += t
    return np.array(times), np.concatenate(out)


@torch.inference_mode()
def spot_check(emb16: ClipEmbedder, title_ids: list[str], n_frames: int) -> list[dict]:
    """Re-embed the first n_frames of a few titles in fp32 and compare with the stored fp16 embeddings."""
    emb32 = ClipEmbedder("mps")
    rows = []
    for tid in title_ids:
        stored = np.load(OUT_DIR / f"{tid}.npz")["emb"][:n_frames].astype(np.float32)
        frames = [f for _, f in zip(range(n_frames), (f for _, f in sample(video_path(tid), Uniform(FPS), "cpu")))]
        ref = []
        for i in range(0, len(frames), BATCH):
            e = emb32.model.encode_image(emb32.preprocess_on_device(np.stack(frames[i:i + BATCH])))
            ref.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        cos = (np.concatenate(ref) * stored).sum(1)
        rows.append({"title": tid, "frames": len(cos), "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min())})
        print(f"  spot check {tid}: cos mean {cos.mean():.6f} min {cos.min():.5f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-check-titles", type=int, default=3)
    ap.add_argument("--spot-check-frames", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb = ClipEmbedder("mps")
    emb.model.half()
    rows, load_before = [], load_avg()
    for t in titles():
        out = OUT_DIR / f"{t['id']}.npz"
        if out.exists():
            print(f"  {t['id']}: cached")
            continue
        with Timer() as timer:
            times, e = embed_title(emb, video_path(t["id"]))
        np.savez(out, times=times, emb=e)
        rows.append({"title": t["id"], "video_s": t["duration_s"], "frames": int(len(times)), "wall_s": timer.seconds,
                     "s_per_video_hour": 3600 * timer.seconds / t["duration_s"]})
        print(f"  {t['id']}: {len(times)} frames in {timer.seconds:.0f}s "
              f"({rows[-1]['s_per_video_hour']:.1f} s/video-hour)", flush=True)
    load_after = load_avg()

    all_files = sorted(OUT_DIR.glob("*.npz"))
    total_frames = sum(len(np.load(f)["times"]) for f in all_files)
    checks = spot_check(emb, [t["id"] for t in titles()][-args.spot_check_titles:], args.spot_check_frames)
    if rows:
        video_s = sum(r["video_s"] for r in rows)
        wall_s = sum(r["wall_s"] for r in rows)
        summary = {"titles_embedded_this_run": len(rows), "video_hours": video_s / 3600, "wall_s": wall_s,
                   "s_per_video_hour": 3600 * wall_s / video_s}
        print(f"total: {len(rows)} titles, {video_s / 3600:.1f} video-hours in {wall_s / 60:.1f} min "
              f"= {summary['s_per_video_hour']:.1f} s/video-hour; {total_frames} frames stored")
        config = {"decode": "cpu, uniform 1 fps, background thread", "preprocess": "on-device (MPS)",
                  "model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED, "inference": "pytorch-fp16-mps",
                  "batch": BATCH, "torch_version": torch.__version__}
        print("  ->", write_result(PHASE, "corpus_embed", "gpu-apple-m5", args.seed, config,
                                   {"summary": summary, "titles": rows, "total_frames_stored": total_frames,
                                    "spot_check_fp16_vs_fp32": checks,
                                    "load_avg_before": load_before, "load_avg_after": load_after}))


if __name__ == "__main__":
    main()
