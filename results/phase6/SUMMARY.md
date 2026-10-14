# Phase 6 — cost: results

**Question:** would anyone actually run this?

**Rules:** SPEC.md §5, Phase 6. Nothing new is built. Every stage of the final pipeline is timed on the laptop, 3 runs
each (median, range), and combined with cloud list prices recorded in `configs/prices.yaml` (looked up 2026-10-13).
Script: `scripts/run_phase6.py` (`counts`, `video --lane …`, `scenes`, `cost`).

## The planning figure

**One M5-class Mac runs the whole pipeline at about 11 minutes of machine time per hour of video: 666 s per
video-hour (656–678), which is 185 machine-hours and about $227 per 1,000 hours of video** at the AWS `mac-m4.metal`
on-demand price ($1.23 per hour, us-east-1, 2026-10-13).

"Whole pipeline" means everything a new film needs before ads can be matched and checked:
- frame search: decode at 1 fps and CLIP
- transcript: Whisper
- scene descriptions: Qwen3-VL-4B, 4-bit
- the final brand-safety system (5d)

It does not include per-ad work such as the optional re-ranking judge, which scales with ads, not hours of video.

Assumptions, stated plainly:
- **Speed:** the price is for an M4 machine, but the speed was measured on the M5 laptop. We assume they run at the same
  speed; this was not measured.
- **Sequential stages:** stages run one after another on one machine. Running stages side by side could lower the
  total, but that was not tried.
- **Billing:** a cloud Mac is billed for at least 24 hours, which does not matter at 185 hours.

## Where the time goes (one M5 laptop, median of 3 runs)

| Stage | Lane | Seconds per video-hour | Share |
|---|---|---|---|
| Scene descriptions (Qwen3-VL-4B, 1.52 s per scene) | gpu-apple-m5 | 292.3 | 44% |
| Safety, grid pass (0.77 s per scene) + weapons check (0.58 s per check) | gpu-apple-m5 | 151.8 | 23% |
| Safety, strip pass (0.37 s per scene) + per-tag checks (0.25 s per check) | gpu-apple-m5 | 90.8 | 14% |
| Transcript (Whisper large-v3-turbo) | gpu-apple-m5 | 84.9 | 13% |
| Strip and grid images (4 frames per scene) | cpu-laptop | 31.9 | 5% |
| Frame search: decode at 1 fps + CLIP ViT-B/32 fp16 | cpu-laptop decode + gpu-apple-m5 model | 10.0 | 1.5% |
| Audio decode | cpu-laptop | 3.2 | 0.5% |
| Scene-description text embedding (mpnet) | gpu-apple-m5 | 1.0 | 0.2% |
| Scene detection | cpu-laptop | < 0.1 | — |
| **Total** | one machine, stages in sequence | **665.8** (656.2–678.2) | |

**Workload:** the per-scene costs are scaled by the workload of the 19 Phase 5 films, which total 25.7 hours of video:
- 191.8 scenes per video-hour
- 79.7 strip safety checks per video-hour
- 7.8 grid weapons checks per video-hour

**Samples:**
- **Per-scene timings:** 60 scenes picked with seed 0. The strip checks were timed as 2 per scene (violence, weapons);
  the time per check is what is used.
- **Frame search and audio decode:** 3 whole films picked with seed 0, 4.0 hours of video.
- **Whisper:** a 10-minute window (20:00–30:00) of each of the 3 films.

**Excluded:** model loading and one warm-up call.

**Machine load:** the machine was in light use during the runs, with a load average of 1.3–4.6 on 10 cores.

**What this says:** three model stages (scene descriptions, the two safety passes, Whisper) are 93% of the cost.
- **Frame search:** the part Phases 0–3 optimised is now 1.5% of it.
- **Scene descriptions:** the one change that clearly improved matching costs 29× more than frame search.
- **Brand safety:** the final system's two passes cost 24× more than frame search.

## Losses and limits

- **Cost against the naive baseline:** the full pipeline costs **1.65× the naive Phase 1 baseline**: every frame
  decoded and embedded in fp32 takes 403 s per video-hour on the same machine. All of the Phase 1–2 speed work is
  outweighed by what Phases 4g–5d added.
- **No NVIDIA lane:** no NVIDIA GPU was rented, so there is no NVIDIA figure. The L4 price ($0.805 per hour, `g6.xlarge`)
  is recorded, but no cost is computed from it. The scene-description and safety models run on mlx, which is Apple-only,
  so an NVIDIA run would first need them moved to another runtime.
- **CPU lane is partial:**
  - The scene-description model would not run on the CPU (BUGS.md #9), so there is no CPU total.
  - What was measured on the CPU:
    - **Frame search:** 38.6 s per video-hour (37.4–40.9), fp32.
    - **Whisper:** 838 s per hour of audio (830–841), about 10× slower than on the M5 GPU (84.9).
  - **Cloud CPU price:** the CPU lane is priced at AWS `c7i.4xlarge` ($0.714 per hour). This assumes the instance matches
    the laptop CPU's speed; not measured.
- **Price date:** prices are on-demand list prices on one date. Discounted or spot pricing would lower every dollar
  figure. Prices change.

## Every lane, per 1,000 hours of video

| What | Lane | s per video-hour | Machine-hours | Dollars | Price used |
|---|---|---|---|---|---|
| Whole pipeline | one M5 laptop (cpu-laptop + gpu-apple-m5) | 665.8 | 185 | $227 | mac-m4.metal $1.23/h |
| Frame search only (1 fps, fp16) | cpu-laptop decode + gpu-apple-m5 model | 10.0 | 2.8 | $3.4 | mac-m4.metal $1.23/h |
| Frame search only (1 fps, fp32) | cpu-laptop | 38.6 | 10.7 | $7.7 | c7i.4xlarge $0.714/h |
| Transcript (audio decode + Whisper) | cpu-laptop | 841.3 | 234 | $167 | c7i.4xlarge $0.714/h |
| Naive Phase 1 baseline (every frame, fp32) | cpu-laptop decode + gpu-apple-m5 model | 403.0 | 112 | $138 | mac-m4.metal $1.23/h |
| Naive Phase 1 baseline (every frame, fp32) | cpu-laptop | 881.5 | 245 | $175 | c7i.4xlarge $0.714/h |
| Anything | NVIDIA L4 | not measured | — | — | g6.xlarge $0.805/h (recorded only) |

**How the naive baseline is derived:** it is not re-timed. It is computed from NUMBERS.md rows:
- every-frame decode: 51.6 s per video-hour (Phase 1)
- CLIP fp32: 307 fps on gpu-apple-m5, 130 fps on cpu-laptop (Phase 2)
- 29.97 frames per second of video, so 107,892 frames per video-hour

## Against PREDICTION.md

| Prediction | Verdict | Evidence |
|---|---|---|
| P1: at 1 fps every lane is decode-bound | **Half right.** True on the M5 GPU: decode 11.6 s per video-hour vs model 3.6. Wrong on the CPU: model 27.7 vs decode 11.6. | Phase 1 decode and Phase 2 model rows; model time is 3,600 frames ÷ fps |
| P5: the laptop does 1,000 hours of video in about 6 hours (about 20 s per video-hour); wrong if over 40 | **Right for frame search, which beat it:** 10.0 s per video-hour, 2.8 hours per 1,000 (uniform 1 fps rather than the keyframe path P5 assumed). **The whole pipeline is 666 s per video-hour**, which the prediction did not foresee: it was written before the scene-description and safety models existed. | this phase |
| P6: a rented NVIDIA GPU is 1–4× faster than the laptop | **Not tested** (no GPU session) | — |
| P7: the GPU session fits in 3 hours | **Not tested** | — |

P2–P4 and P8 were Phase 1 questions and are answered in `results/phase1/SUMMARY.md`.

## Files

- **Workload:** `counts_cpu-laptop_seed0.json`
- **Video timings:** `video_gpu-apple-m5_seed0.json`, `video_cpu-laptop_seed0.json` (frame search, audio decode, Whisper)
- **Scene timings:** `scenes_gpu-apple-m5_seed0.json` (per-scene and per-check timings)
- **Combined:** `cost_laptop_seed0.json`, which combines all of these with `configs/prices.yaml`
