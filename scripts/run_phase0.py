"""Phase 0: measure decode and model rates on the laptop, before renting any GPU.

Decodes a fixed 10-minute window (every frame) several times, keeps one frame per second,
then times preprocessing and CLIP inference on those frames on each available device.
Writes results/phase0/*.json.
"""

import argparse
import statistics
from pathlib import Path

import torch

from scenesafe.common.results import Timer, load_avg, write_result
from scenesafe.decode.cpu import decode_window, probe
from scenesafe.embed.clip import DEFAULT_MODEL, DEFAULT_PRETRAINED, ClipEmbedder

PHASE = 0


def summarise(values: list[float]) -> dict:
    return {"runs": values, "median": statistics.median(values), "min": min(values), "max": max(values)}


def bench_decode(video: Path, start_s: float, duration_s: float, repeats: int, src_fps: float):
    rates, sampled = [], []
    step = round(src_fps)  # keep ~1 frame per second, from the first run only
    load_before = load_avg()
    for r in range(repeats):
        n = 0
        with Timer() as t:
            for frame in decode_window(video, start_s, duration_s):
                if r == 0 and n % step == 0:
                    sampled.append(frame)
                n += 1
        rates.append(n / t.seconds)
        print(f"  decode run {r + 1}: {n} frames in {t.seconds:.1f}s = {n / t.seconds:.1f} fps")
    return {"frames_per_run": n, "fps": summarise(rates), "load_avg_before": load_before,
            "load_avg_after": load_avg()}, sampled


def bench_embed(frames, device: str, batch_size: int, repeats: int):
    emb = ClipEmbedder(device)
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    pre_rates, tensors = [], None
    for _ in range(repeats):
        with Timer() as t:
            tensors = [emb.preprocess_batch(b) for b in batches]
        pre_rates.append(len(frames) / t.seconds)

    for x in tensors:  # warm-up: one full untimed pass (one batch was not enough on MPS)
        emb.embed(x)
    emb.sync()
    load_before = load_avg()
    model_rates = []
    for r in range(repeats):
        with Timer() as t:
            for x in tensors:
                emb.embed(x)
            emb.sync()
        model_rates.append(len(frames) / t.seconds)
        print(f"  {device} model run {r + 1}: {len(frames) / t.seconds:.1f} fps")
    return {"frames": len(frames), "preprocess_fps": summarise(pre_rates), "model_fps": summarise(model_rates),
            "load_avg_before": load_before, "load_avg_after": load_avg()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/raw/notld_1968_1024x768.mp4"))
    ap.add_argument("--start", type=float, default=600.0, help="window start (s); skips opening titles")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    meta = probe(args.video)
    print(f"video: {meta}")
    config = {"video": str(args.video), "video_meta": meta, "window_start_s": args.start,
              "window_duration_s": args.duration, "repeats": args.repeats}

    print("decode (cpu, PyAV/ffmpeg, all frames, includes YUV->RGB):")
    decode, sampled = bench_decode(args.video, args.start, args.duration, args.repeats, meta["fps"])
    print("  ->", write_result(PHASE, "decode", "cpu-laptop", args.seed,
                               {**config, "backend": "pyav-ffmpeg", "threads": "AUTO",
                                "av_version": __import__("av").__version__}, decode))

    lanes = {"cpu": "cpu-laptop"}
    if torch.backends.mps.is_available():
        lanes["mps"] = "gpu-apple-m5"
    for device, lane in lanes.items():
        print(f"embed ({device}, {len(sampled)} frames sampled at ~1 fps):")
        res = bench_embed(sampled, device, args.batch_size, args.repeats)
        model_cfg = {**config, "model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED, "runtime": "pytorch-eager",
                     "precision": "fp32", "batch_size": args.batch_size, "device": device,
                     "torch_version": torch.__version__, "preprocess_device": "cpu"}
        print("  ->", write_result(PHASE, "embed", lane, args.seed, model_cfg, res))


if __name__ == "__main__":
    main()
