"""Phase 1 speed: decode throughput per backend x sampling policy, and scaling across parallel processes.

Every run decodes the same test windows (scenesafe.corpus.test_windows) and converts each KEPT frame to RGB,
which is what the model needs. Writes results/phase1/decode_<lane>_seed<seed>.json.

Run nothing else heavy while this runs: it is a timing benchmark.
"""

import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor

from scenesafe.common.results import Timer, load_avg, write_result
from scenesafe.corpus import Window, test_windows
from scenesafe.decode.sampling import parse, sample

PHASE = 1
LANES = {"cpu": "cpu-laptop", "videotoolbox": "hwdec-apple-m5", "nvdec": "gpu-nvidia"}


def run_window(args: tuple[Window, str, str]) -> dict:
    w, backend, spec = args
    stats = {}
    for _ in sample(w.path, parse(spec), backend, w.start_s, w.duration_s, stats):
        pass
    return stats


def run_once(windows, backend, spec, workers) -> dict:
    jobs = [(w, backend, spec) for w in windows]
    with Timer() as t:
        if workers == 1:
            stats = [run_window(j) for j in jobs]
        else:
            with ProcessPoolExecutor(workers) as pool:
                stats = list(pool.map(run_window, jobs))
    video_s = sum(w.duration_s for w in windows)
    return {"wall_s": t.seconds, "decoded": sum(s["decoded"] for s in stats), "kept": sum(s["kept"] for s in stats),
            "x_realtime": video_s / t.seconds, "s_per_video_hour": 3600 * t.seconds / video_s}


def summarise(runs: list[dict]) -> dict:
    sph = [r["s_per_video_hour"] for r in runs]
    return {"runs": runs, "s_per_video_hour": {"median": statistics.median(sph), "min": min(sph), "max": max(sph)},
            "decoded_fps_median": statistics.median(r["decoded"] / r["wall_s"] for r in runs),
            "decoded": runs[0]["decoded"], "kept": runs[0]["kept"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cpu", "videotoolbox"])
    ap.add_argument("--policies", nargs="+", default=["all", "uniform@1", "keyframes", "scene@0.05"])
    ap.add_argument("--parallel-policies", nargs="+", default=["uniform@1", "keyframes"])
    ap.add_argument("--workers", nargs="+", type=int, default=[2, 4, 7])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    windows = test_windows()
    hours = sum(w.duration_s for w in windows) / 3600
    for backend in args.backends:
        load_before = load_avg()
        results = []
        cases = [(spec, 1) for spec in args.policies] + [(spec, n) for spec in args.parallel_policies for n in args.workers]
        for spec, workers in cases:
            runs = [run_once(windows, backend, spec, workers) for _ in range(args.repeats)]
            s = summarise(runs)
            results.append({"policy": spec, "workers": workers, **s})
            m = s["s_per_video_hour"]
            print(f"{backend:12s} {spec:11s} workers={workers}  {m['median']:6.1f} s/video-hour "
                  f"({m['min']:.1f}-{m['max']:.1f})  decoded {s['decoded']}  kept {s['kept']}", flush=True)
        config = {"backend": backend, "windows": [{"id": w.id, "start_s": w.start_s, "duration_s": w.duration_s}
                                                  for w in windows],
                  "hours": hours, "repeats": args.repeats, "rgb_conversion": "kept frames only",
                  "av_version": __import__("av").__version__}
        print("  ->", write_result(PHASE, "decode", LANES[backend], args.seed, config,
                                   {"cases": results, "load_avg_before": load_before, "load_avg_after": load_avg()}))


if __name__ == "__main__":
    main()
