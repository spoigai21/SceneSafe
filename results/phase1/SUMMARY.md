# Phase 1 — decode and sample: results

**Question:** how fast can frames be got off disk, and how many are needed?

## Setup

**Test set:** 7 public-domain films, one per genre, minutes 10–25 of each. That's 1.75 hours of
video and 188,811 frames (29.97 fps, 480p H.264). The titles are listed in
`configs/phase1_testset.yaml`.

**Machine:** Apple M5 laptop. **It was heavily shared during the timing runs:** load average
8.6–16 on 10 cores, with a virtual machine and other apps running. Treat the timings as
pessimistic. Re-running on an idle machine should only make them faster.

**Timing method:** 3 repeats per case; median with the range in brackets. Each run converts every
*kept* frame to RGB, because the model needs RGB.

## 1. Speed: decode seconds per hour of video (lower is better)

| Policy | Frames kept / video-hour | CPU (ffmpeg) | VideoToolbox (Apple hardware decoder) |
|---|---|---|---|
| every frame | 107,892 | **51.6** (45.6–52.5) | 96.7 (95.9–98.4) |
| uniform 1 fps | 3,600 | **11.6** (11.3–12.5) | 61.0 (60.7–61.7) |
| keyframes | 769 | **1.6** (1.6–1.8) | 2.0 (1.9–2.1) |
| scene change @0.05 | 5,157 | 57.9 (55.1–62.0) | 133.4 (131.3–143.3) |

**Parallel processes** (several films decoded at once), in s/video-hour:

| Policy | Backend | 1 process | 2 | 4 | 7 |
|---|---|---|---|---|---|
| uniform 1 fps | CPU | 11.6 | **8.9** | 9.8 | 10.2 |
| uniform 1 fps | VideoToolbox | 61.0 | 40.0 | 25.1 | 21.1 |
| keyframes | CPU | 1.6 | 1.2 | 1.0 | **0.7** |
| keyframes | VideoToolbox | 2.0 | 1.4 | 1.1 | 0.9 |

Sources: `decode_cpu-laptop_seed0.json` and `decode_hwdec-apple-m5_seed0.json`.

## 2. Quality: how much each policy misses

**Answer key:** every frame of the test set embedded with CLIP ViT-B/32.
**Units:** 630 ten-second segments, each scored by its best-matching frame.
**Searches:** 20 text searches in `configs/phase1_queries.yaml`, written before any results.
**Score:** recall@k, the share of the every-frame top-k segments that the policy also puts in its
top-k.

| Policy | Frames / video-hour | Segments with ≥1 frame | Recall@10 | Recall@50 |
|---|---|---|---|---|
| every frame (the reference) | 107,892 | 100% | 1.000 | 1.000 |
| uniform 2 fps | 7,200 | 100% | 0.695 | 0.785 |
| **uniform 1 fps** | 3,600 | 100% | **0.630** | **0.736** |
| uniform 0.5 fps | 1,800 | 100% | 0.535 | 0.680 |
| uniform 0.25 fps | 900 | 100% | 0.450 | 0.606 |
| keyframes | 769 | 100% | 0.425 | 0.570 |
| scene change @0.03 | 10,170 | 97.8% | 0.715 | 0.782 |
| scene change @0.05 | 5,157 | 93.2% | 0.575 | 0.717 |
| scene change @0.08 | 2,381 | 84.0% | 0.500 | 0.633 |
| scene change @0.12 | 1,070 | 67.8% | 0.420 | 0.540 |

Source: `quality_gpu-apple-m5_seed0.json`. Curve: `sampling_curve.png`.

## Findings

1. **Converting frames to RGB was most of the "decode" cost, not decoding itself.** Every frame
   takes 51.6 s/video-hour. Decoding every frame but converting only 1 per second takes 11.6.
   Phase 0 converted every frame, so it overstated the decode cost of a sampled pipeline.
2. **Keyframes are the fastest by far:** 1.6 s/video-hour, 32× faster than every frame and 7×
   faster than uniform 1 fps. At the same frame budget their quality matches uniform sampling
   (769 frames → 0.425, against 900 → 0.450). Their only weakness is that the encoder decides how
   many you get: here one every 4.7 s.
3. **Quality has no free cut-off point.** Recall rises steadily with each doubling of frames. No
   sampling rate is "enough". Choosing a rate is a straight speed vs quality trade.
4. **Scene-change sampling loses to plain uniform sampling** at every frame budget. It's also the
   slowest policy, because it has to look at every frame, and at stricter thresholds it leaves up
   to a third of segments with no frame at all. **This is a negative result.**
5. **The Apple hardware decoder is slower than the CPU** on this low-resolution video: 1.9× slower
   for every frame, 5× slower at 1 fps. It only catches up with many parallel processes, and even
   then the CPU wins.
6. **Parallel decoding barely helps on the CPU.** ffmpeg already spreads one film across cores,
   and the machine was busy with other work.

## Decision for later phases

**Default policy: uniform 1 fps on the CPU, 2 processes**, which is about 9 s of decode per
video-hour. Keyframes would be 7× faster, but recall@10 falls from 0.63 to 0.43, and on a 50-hour
corpus the time difference is minutes. Revisit in Phase 4 if matching quality turns out not to
need 1 fps.

## Predictions checked (from `PREDICTION.md`)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | 1 fps sampling stays decode-bound | Decode at 1 fps is 11.6 s/video-hour; the model is about 15 s (Phase 0 M5 GPU rate). The model is now slower. | **Wrong.** Phase 0 counted RGB conversion of every frame as decode cost, and this video is lower resolution. |
| P2 | Keyframes 10–30× faster than every frame | 32× | **Wrong**, just above the range. Keyframes are sparser in these files (every 4.7 s, against 1.5 s in the Phase 0 film). |
| P3 | With keyframes, the bottleneck moves to the model | Keyframe decode 1.6 s; model about 3.2 s per video-hour | **Right** |
| P4 | VideoToolbox 1.5–3× faster than CPU | 0.53× (slower) for every frame | **Wrong** |
| P8 | 1080p decode about 140 s/video-hour | Not tested: there is no 1080p film in the test set | **Untested** |

## Limits

- Heavily shared machine during timing (see Setup).
- One resolution: 480p, from Internet Archive copies, which is below streaming quality.
- The western test film was substituted because archive.org was down (see the test-set config).
- The quality score compares against the every-frame answer key, not against human judgement:
  it measures what sampling costs, not whether CLIP is right.
- Recall@10 is strict. Swapping one near-identical segment for another counts as a miss.
