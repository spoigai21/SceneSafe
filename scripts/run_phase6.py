"""Phase 6: cost. What does it take to run the final pipeline over 1,000 hours of video? Nothing new is built: every
stage of the pipeline used by Phases 4h and 5d is timed on the laptop (3 runs each, median and range), then turned into
machine-hours and dollars per 1,000 video-hours with cloud prices recorded in configs/prices.yaml.

  counts   per video-hour workload of the 19 Phase 5 films: scenes, safety checks.
           Writes results/phase6/counts_cpu-laptop_seed0.json.
  video    the stages that scale with running time, on 3 films picked with the seed: 1 fps decode + CLIP (the Phase 2
           configuration for the lane), audio decode, and Whisper on a 10-minute window of each film.
           Writes results/phase6/video_<lane>_seed0.json. --lane gpu-apple-m5 or cpu-laptop.
  scenes   the stages that scale with the number of scenes, on 60 scenes picked with the seed: strip and grid images,
           scene cards, the 5d safety passes, card text embedding, scene detection.
           Writes results/phase6/scenes_gpu-apple-m5_seed0.json.
  cost     combines the three files with the prices. Writes results/phase6/cost_seed0.json (no single lane: each figure
           in it names its lane).

Model loading and one warm-up call are excluded from every timing (paid once per machine, not per hour of video).
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from scenesafe.common.results import REPO_ROOT, Timer, load_avg, write_result

PHASE = 6
OUT = REPO_ROOT / "results" / "phase6"
PRICES = REPO_ROOT / "configs" / "prices.yaml"
RUNS = 3
N_FILMS = 3
WINDOW = (1200.0, 600.0)          # Whisper window: start and length in seconds (20:00-30:00 of each film)
N_SCENES = 60
STRIP_TAGS = ("violence", "weapons")   # strip verifier: 2 checks per sampled scene (the time per check is what is used)
NTSC_FPS = 29.97                  # frames per video-hour for the every-frame baseline (most corpus files are ~30 fps)


def films() -> list[str]:
    from run_phase5 import films as f5
    return f5()


def stats(xs: list[float]) -> dict:
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "runs": xs}


def counts():
    from run_phase5 import DATA, setup
    corpus, units, names, film, cards = setup()
    hours = len(corpus.emb) / 3600                          # one embedding per second of video
    strip_checks = len(json.loads((DATA / "verify.json").read_text()))
    first = json.loads((DATA / "tags_5d_first.json").read_text())
    grid_checks = sum("weapons" in first[n]["tags"] for n in names)
    data = {"films": len(set(film)), "video_hours": hours, "scenes": len(names), "scenes_per_hour": len(names) / hours,
            "strip_checks": strip_checks, "strip_checks_per_hour": strip_checks / hours,
            "grid_checks": grid_checks, "grid_checks_per_hour": grid_checks / hours,
            "scenes_with_speech": sum(bool(cards[n]["dialogue"]) for n in names)}
    print(json.dumps(data, indent=1))
    write_result(PHASE, "counts", "cpu-laptop", 0, {"source": "19 Phase 5 films, data/phase5"}, data)


def pick_films(seed: int) -> list[str]:
    return sorted(np.random.default_rng(seed).choice(films(), N_FILMS, replace=False).tolist())


def video(seed: int, lane: str):
    import torch
    import mlx.core as mx
    from scripts.embed_corpus import BATCH, batches
    from scenesafe.corpus import video_path
    from scenesafe.embed.clip import ClipEmbedder
    from scenesafe.match.transcribe import RATE, WHISPER_MODEL, load_audio

    gpu = lane == "gpu-apple-m5"
    if not gpu:
        mx.set_default_device(mx.cpu)
    import mlx_whisper
    ids = pick_films(seed)
    emb = ClipEmbedder("mps" if gpu else "cpu")
    if gpu:
        emb.model.half()                     # Phase 2 choice for the GPU: fp16 on MPS; fp32 on the CPU

    @torch.inference_mode()
    def embed(path) -> int:
        n = 0
        for _, frames in batches(path, BATCH):
            x = emb.preprocess_on_device(frames)
            e = emb.model.encode_image(x.half() if gpu else x).float()
            (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
            n += len(frames)
        return n

    def whisper(audio):
        return mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, language="en",
                                      condition_on_previous_text=False, verbose=None)

    embed(video_path(ids[0]))                                           # warm-up (untimed)
    audio = {t: load_audio(video_path(t)) for t in ids}
    whisper(audio[ids[0]][:30 * RATE])                                  # loads the model (untimed)
    seconds = {t: len(audio[t]) / RATE for t in ids}
    hours = sum(seconds.values()) / 3600
    clip, aud, wsp, frames, loads = [], [], [], 0, []
    for r in range(RUNS):
        loads.append(load_avg())
        with Timer() as tc:
            frames = sum(embed(video_path(t)) for t in ids)
        with Timer() as ta:
            for t in ids:
                load_audio(video_path(t))
        a, b = int(WINDOW[0] * RATE), int(sum(WINDOW) * RATE)
        with Timer() as tw:
            words = sum(len(whisper(audio[t][a:b])["text"].split()) for t in ids)
        clip.append(tc.seconds / hours)
        aud.append(ta.seconds / hours)
        wsp.append(tw.seconds / (N_FILMS * WINDOW[1] / 3600))
        print(f"run {r + 1}: decode+CLIP {clip[-1]:.1f} s/h, audio decode {aud[-1]:.1f} s/h, "
              f"Whisper {wsp[-1]:.0f} s per audio-hour ({words} words)", flush=True)
    data = {"films": ids, "video_hours": hours, "frames_embedded": frames, "whisper_words": words,
            "decode_clip_s_per_video_hour": stats(clip), "audio_decode_s_per_video_hour": stats(aud),
            "whisper_s_per_audio_hour": stats(wsp), "load_avg_per_run": loads}
    config = {"runs": RUNS, "sampling": "uniform 1 fps", "decode": "PyAV/ffmpeg on the CPU, in a background thread",
              "clip": ("ViT-B/32 fp16, PyTorch MPS" if gpu else "ViT-B/32 fp32, PyTorch CPU") + f", batch {BATCH}",
              "whisper": f"{WHISPER_MODEL} (mlx-whisper, {'GPU' if gpu else 'CPU'} device)",
              "whisper_window_s": WINDOW, "excluded": "model loading and one warm-up call"}
    write_result(PHASE, "video", lane, seed, config, data)


def scene_sample(seed: int):
    from run_phase5 import setup
    corpus, units, names, film, cards = setup()
    pick = sorted(np.random.default_rng(seed).choice(len(names), N_SCENES, replace=False).tolist())
    return corpus, units, names, film, cards, pick


def scenes_step(seed: int):
    import mlx.core as mx
    from run_phase4f import SCENE_THRESHOLD
    from scenesafe.index.units import scenes
    from scenesafe.match import cards as cd
    from scenesafe.match import dialogue as dl
    from scenesafe.match.display import frame_grid, frame_strip
    from scenesafe.safety.tags import (GRID_OBVIOUS_PROMPT, GRID_TAG_PROMPT, TAGS_OBVIOUS, tag_scenes, verify,
                                       verify_p)

    corpus, units, names, film, cards, pick = scene_sample(seed)
    dlg = {names[u]: cards[names[u]]["dialogue"] for u in pick}
    all_cards = [cd.card_text(cards[n]["card"]) for n in names]
    cd._model()
    rows = {k: [] for k in ("scene_detection", "strip_image", "grid_image", "card", "strip_tags", "strip_check",
                            "grid_tags", "grid_check", "card_text_embedding")}
    loads = []
    with tempfile.TemporaryDirectory() as warm:                          # warm-up: one call of each kind (untimed)
        w = Path(warm)
        u = pick[0]
        frame_strip(film[u], units.start[u], units.end[u], w / "s.jpg")
        tag_scenes([("w", w / "s.jpg", "")], w / "t.json")
        dl.embed_text(all_cards[:8], w / "e.npy")
    for r in range(RUNS):
        loads.append(load_avg())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with Timer() as t:
                scenes(corpus, SCENE_THRESHOLD)
            rows["scene_detection"].append(t.seconds / (len(corpus.emb) / 3600))      # s per video-hour
            strips, grids_ = {}, {}
            with Timer() as t:
                for u in pick:
                    strips[names[u]] = tmp / f"{names[u]}_s.jpg"
                    frame_strip(film[u], units.start[u], units.end[u], strips[names[u]])
            rows["strip_image"].append(t.seconds / N_SCENES)
            with Timer() as t:
                for u in pick:
                    grids_[names[u]] = tmp / f"{names[u]}_g.jpg"
                    frame_grid(film[u], units.start[u], units.end[u], grids_[names[u]])
            rows["grid_image"].append(t.seconds / N_SCENES)
            cd.CARD_DIR = tmp / "cards"
            with Timer() as t:
                cd.make_cards("timing", [(n, strips[n], dlg[n]) for n in dlg])
            rows["card"].append(t.seconds / N_SCENES)
            with Timer() as t:
                tag_scenes([(n, strips[n], dlg[n]) for n in dlg], tmp / "tags.json")
            rows["strip_tags"].append(t.seconds / N_SCENES)
            with Timer() as t:
                verify([(n, tg, strips[n], dlg[n]) for n in dlg for tg in STRIP_TAGS], tmp / "verify.json")
            rows["strip_check"].append(t.seconds / (N_SCENES * len(STRIP_TAGS)))
            with Timer() as t:
                tag_scenes([(n, grids_[n], dlg[n]) for n in dlg], tmp / "grid_tags.json", GRID_TAG_PROMPT)
            rows["grid_tags"].append(t.seconds / N_SCENES)
            with Timer() as t:
                verify_p([(n, "weapons", grids_[n], dlg[n]) for n in dlg], tmp / "grid_check.json",
                         GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
            rows["grid_check"].append(t.seconds / N_SCENES)
            with Timer() as t:                                             # every card of the 19 films, one batch job
                dl.embed_text(all_cards, tmp / "cards.npy")
            rows["card_text_embedding"].append(t.seconds / len(all_cards))
            mx.clear_cache()
        print(f"run {r + 1}: " + ", ".join(f"{k} {v[-1]:.3f}" for k, v in rows.items()), flush=True)
    data = {"scenes": [names[u] for u in pick], "load_avg_per_run": loads,
            "s_per_scene": {k: stats(v) for k, v in rows.items() if k not in ("scene_detection", "strip_check",
                                                                               "grid_check")},
            "s_per_check": {"strip_check": stats(rows["strip_check"]), "grid_check": stats(rows["grid_check"])},
            "s_per_video_hour": {"scene_detection": stats(rows["scene_detection"])}}
    config = {"runs": RUNS, "n_scenes": N_SCENES, "strip_checks_per_scene": list(STRIP_TAGS),
              "vlm": f"{cd.CARD_MODEL} (mlx-vlm, M5 GPU), greedy", "text_model": f"{dl.TEXT_MODEL} (MPS)",
              "images": "PyAV decode on the CPU, 4 frames per scene; strip 4 x 192x144, grid 2 x 2 x 384x288",
              "card_text_embedding": "all 4,922 cards per run, includes the child process start and model load",
              "excluded": "VLM loading and one warm-up call of each kind"}
    write_result(PHASE, "scenes", "gpu-apple-m5", seed, config, data)


def load(name: str, lane: str, seed: int) -> dict:
    return json.loads((OUT / f"{name}_{lane}_seed{seed}.json").read_text())


def cost(seed: int):
    prices = yaml.safe_load(PRICES.read_text())
    c = load("counts", "cpu-laptop", 0)
    sc = load("scenes", "gpu-apple-m5", seed)
    per_h = {}                                                  # stage -> (lane, seconds per video-hour)
    for lane in ("gpu-apple-m5", "cpu-laptop"):
        v = load("video", lane, seed)
        per_h[lane] = {"decode + CLIP (1 fps)": v["decode_clip_s_per_video_hour"]["median"],
                       "audio decode": v["audio_decode_s_per_video_hour"]["median"],
                       "Whisper": v["whisper_s_per_audio_hour"]["median"]}
    s, k = sc["s_per_scene"], sc["s_per_check"]
    sph = c["scenes_per_hour"]
    scene_stages = {
        "scene detection": ("cpu-laptop", sc["s_per_video_hour"]["scene_detection"]["median"]),
        "strip + grid images": ("cpu-laptop", sph * (s["strip_image"]["median"] + s["grid_image"]["median"])),
        "scene cards": ("gpu-apple-m5", sph * s["card"]["median"]),
        "card text embedding": ("gpu-apple-m5", sph * s["card_text_embedding"]["median"]),
        "safety: strip tags + checks": ("gpu-apple-m5", sph * s["strip_tags"]["median"]
                                        + c["strip_checks_per_hour"] * k["strip_check"]["median"]),
        "safety: grid tags + weapons check": ("gpu-apple-m5", sph * s["grid_tags"]["median"]
                                              + c["grid_checks_per_hour"] * k["grid_check"]["median"]),
    }
    laptop = {"decode + CLIP (1 fps)": ("gpu-apple-m5 model, cpu-laptop decode",
                                        per_h["gpu-apple-m5"]["decode + CLIP (1 fps)"]),
              "audio decode": ("cpu-laptop", per_h["gpu-apple-m5"]["audio decode"]),
              "Whisper": ("gpu-apple-m5", per_h["gpu-apple-m5"]["Whisper"])} | scene_stages
    total = sum(x for _, x in laptop.values())
    v = load("video", "gpu-apple-m5", seed)
    spread = {}                     # the same sum with every stage at its fastest / slowest run (a bound, not a run)
    for end in ("min", "max"):
        spread[end] = (v["decode_clip_s_per_video_hour"][end] + v["audio_decode_s_per_video_hour"][end]
                       + v["whisper_s_per_audio_hour"][end] + sc["s_per_video_hour"]["scene_detection"][end]
                       + sph * sum(s[k][end] for k in ("strip_image", "grid_image", "card", "card_text_embedding",
                                                        "strip_tags", "grid_tags"))
                       + c["strip_checks_per_hour"] * k["strip_check"][end]
                       + c["grid_checks_per_hour"] * k["grid_check"][end])
    search_only = laptop["decode + CLIP (1 fps)"][1] + laptop["scene detection"][1]

    def money(sec_per_h, price):
        mh = sec_per_h * 1000 / 3600
        return {"machine_hours_per_1000_video_hours": mh, "dollars_per_1000_video_hours": mh * price}

    mac = prices["apple_m4_cloud"]
    cpu = prices["cpu_cloud"]
    cpu_lane = per_h["cpu-laptop"]
    cpu_search = cpu_lane["decode + CLIP (1 fps)"] + laptop["scene detection"][1]
    cpu_speech = cpu_lane["audio decode"] + cpu_lane["Whisper"]
    frames_per_hour = NTSC_FPS * 3600
    naive = {"gpu-apple-m5": 51.6 + frames_per_hour / 307, "cpu-laptop": 51.6 + frames_per_hour / 130}
    out = {
        "stages_s_per_video_hour": {k: {"lane": l, "s": x} for k, (l, x) in laptop.items()},
        "laptop_full_pipeline": {"s_per_video_hour": total, "s_per_video_hour_range": spread,
                                 **money(total, mac["usd_per_hour"]),
                                 "price": mac},
        "laptop_search_only": {"s_per_video_hour": search_only, **money(search_only, mac["usd_per_hour"])},
        "cpu_lane": {"search_only_s_per_video_hour": cpu_search, **money(cpu_search, cpu["usd_per_hour"]),
                     "speech_s_per_video_hour": cpu_speech,
                     "speech_dollars_per_1000_video_hours": money(cpu_speech, cpu["usd_per_hour"])[
                         "dollars_per_1000_video_hours"],
                     "vlm_stages": "not measured: mlx-vlm stops with a Metal GPU timeout on the CPU device "
                                   "(BUGS.md #9)", "price": cpu},
        "naive_phase1_baseline": {lane: {"s_per_video_hour": x, **money(x, mac["usd_per_hour"] if lane == "gpu-apple-m5"
                                                                          else cpu["usd_per_hour"])}
                                  for lane, x in naive.items()},
        "nvidia_lane": {"measured": False, "price": prices["nvidia_l4"]},
        "counts": c,
    }
    gpu_file = OUT / f"gpu_gpu-nvidia-a10_seed{seed}.json"
    if gpu_file.exists():                       # the rented NVIDIA session (gpu/run_gpu.py); absent until it is run
        out["nvidia_lane"] = nvidia_lane(json.loads(gpu_file.read_text()), c, laptop["scene detection"][1],
                                         prices["nvidia_a10_lambda"], money)
    print(f"{'stage':36s} {'lane':40s} s/video-hour")
    for k, (l, x) in laptop.items():
        print(f"{k:36s} {l:40s} {x:8.1f}")
    print(f"{'TOTAL (one M5 laptop, stages in sequence)':77s} {total:8.1f} ({spread['min']:.1f}-{spread['max']:.1f})")
    for k in ("laptop_full_pipeline", "laptop_search_only"):
        o = out[k]
        print(f"{k}: {o['s_per_video_hour']:.1f} s/h -> {o['machine_hours_per_1000_video_hours']:.0f} machine-h, "
              f"${o['dollars_per_1000_video_hours']:.0f} per 1,000 video-hours")
    print("cpu lane:", json.dumps({k: v for k, v in out["cpu_lane"].items() if k != "price"}, indent=1))
    print("naive:", json.dumps(out["naive_phase1_baseline"], indent=1))
    if out["nvidia_lane"]["measured"]:
        for mode, o in out["nvidia_lane"]["modes"].items():
            tot = o["s_per_video_hour"]
            print(f"nvidia a10, VLM {mode}: " + (f"{tot:.1f} s/h -> {o['machine_hours_per_1000_video_hours']:.0f} "
                                                 f"machine-h, ${o['dollars_per_1000_video_hours']:.0f} per 1,000 "
                                                 "video-hours" if tot is not None else f"incomplete: {o['missing']}"))
    write_result(PHASE, "cost", "laptop", seed,
                 {"prices": str(PRICES.relative_to(REPO_ROOT)), "frames_per_video_hour_naive": frames_per_hour,
                  "naive_sources": "NUMBERS.md: Phase 1 every-frame decode 51.6 s/h; Phase 2 fp32 MPS 307 fps, "
                                   "fp32 CPU 130 fps"}, out)


def nvidia_lane(g: dict, c: dict, scene_detection: float, price: dict, money) -> dict:
    """Per video-hour cost of the whole pipeline on the rented NVIDIA A10, built like the laptop total: per-scene and
    per-check timings scaled by the same workload counts. Two figures, one per VLM mode (sequential: one request at a
    time, as on the laptop; batched: all requests at once). Scene detection is NumPy on the CPU and is taken from the
    laptop (under 0.1 s per video-hour). A stage that failed on the GPU makes that total incomplete, not estimated."""
    sph = c["scenes_per_hour"]
    med = lambda d, k: d[k]["median"] if k in d else None                     # noqa: E731
    s = g.get("s_per_scene", {})
    common = {"decode + CLIP (1 fps), CPU decode": med(g, "decode_cpu_clip_s_per_video_hour"),
              "audio decode": med(g, "audio_decode_s_per_video_hour"),
              "Whisper (faster-whisper)": med(g, "whisper_s_per_audio_hour"),
              "scene detection (laptop figure)": scene_detection,
              "strip + grid images": (sph * (s["strip_image"]["median"] + s["grid_image"]["median"])
                                      if "strip_image" in s and "grid_image" in s else None),
              "card text embedding": sph * s["card_text_embedding"]["median"] if "card_text_embedding" in s else None}
    modes = {}
    for mode, v in g.get("vlm", {}).items():
        ps, pc = v["s_per_scene"], v["s_per_check"]
        ok = all(k in ps for k in ("card", "strip_tags", "grid_tags")) and all(k in pc for k in PER_CHECK_KINDS)
        stages = dict(common)
        stages["scene cards"] = sph * ps["card"]["median"] if "card" in ps else None
        stages["safety: strip tags + checks"] = (sph * ps["strip_tags"]["median"] + c["strip_checks_per_hour"]
                                                 * pc["strip_check"]["median"]) if ok else None
        stages["safety: grid tags + weapons check"] = (sph * ps["grid_tags"]["median"] + c["grid_checks_per_hour"]
                                                       * pc["grid_check"]["median"]) if ok else None
        missing = [k for k, x in stages.items() if x is None]
        total = None if missing else sum(stages.values())
        modes[mode] = {"stages_s_per_video_hour": stages, "missing": missing, "s_per_video_hour": total,
                       **(money(total, price["usd_per_hour"]) if total is not None
                          else {"machine_hours_per_1000_video_hours": None, "dollars_per_1000_video_hours": None})}
    return {"measured": True, "lane": "gpu-nvidia-a10 (VLM, Whisper, CLIP, text model) + its host CPU (decode, images)",
            "nvdec_decode_clip_s_per_video_hour": med(g, "decode_nvdec_clip_s_per_video_hour"),
            "modes": modes, "errors": g.get("errors", {}), "gpu": g.get("gpu"), "price": price,
            "differences_from_laptop": "VLM bf16 on vLLM (laptop: MLX 4-bit); Whisper on faster-whisper (laptop: "
                                       "mlx-whisper); image timings on 60 scenes of the 3 shipped films"}


PER_CHECK_KINDS = ("strip_check", "grid_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["counts", "video", "scenes", "cost"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lane", choices=["gpu-apple-m5", "cpu-laptop"], default="gpu-apple-m5")
    args = ap.parse_args()
    if args.step == "counts":
        counts()
    elif args.step == "video":
        video(args.seed, args.lane)
    elif args.step == "scenes":
        scenes_step(args.seed)
    else:
        cost(args.seed)


if __name__ == "__main__":
    main()
