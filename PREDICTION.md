# PREDICTION — written before any GPU was rented

Written 2026-10-06, after measuring on the laptop only. **This file is not edited after it is
committed.** Later phases compare their measurements against it, and a wrong prediction stays
here.

---

## 1. What was measured (laptop)

**Machine:** Apple M5 (10 cores, 16 GB), macOS. The machine was in normal use during the runs
(load average 4–8 on 10 cores), so expect some noise.

**Video:** *Night of the Living Dead* (1968), public domain, Internet Archive item
`Night_Of_The_Living_Dead_raw_HD_WS`, file `NightOfTheLivingDeadRemasteredHD.mp4`
(1,902,907,166 bytes; H.264, 1024×768, 29.96 fps, 95.6 min).

**Window:** 10:00–20:00, which is 17,983 frames. Each stage was run 5 times; the figures are
medians, with the range in brackets.

| Stage | Lane | Rate (frames/sec) | Seconds per hour of video |
|---|---|---|---|
| Decode, every frame, incl. YUV→RGB (PyAV/ffmpeg) | cpu-laptop | **2,045** (1,838–2,192) | **53** |
| Preprocess (resize/crop/normalise, PIL, 1 process) | cpu-laptop | 285–338 median (79–444 range, very noisy) | ~11–13 at 1 fps sampling |
| CLIP ViT-B/32 fp32, PyTorch eager, batch 32 | cpu-laptop | **107** (106–117) | 34 at 1 fps · 1,012 at all frames |
| CLIP ViT-B/32 fp32, PyTorch eager, batch 32 | gpu-apple-m5 (MPS) | **240** (206–255) | **15** at 1 fps · 449 at all frames |

Source files: `results/phase0/decode_cpu-laptop_seed0.json`,
`results/phase0/embed_cpu-laptop_seed0.json` and `results/phase0/embed_gpu-apple-m5_seed0.json`.
The model was timed on 600 frames sampled at about 1 fps from the same window.

### The ratio that matters

Decode runs **8.5×** faster than the model on the M5 GPU, and 19× faster than the model on the
CPU. But decode has to touch **every** frame (H.264 frames depend on each other), while the model
sees only the **sampled** frames.

- **Sampling at 1 fps (keeps 1 in 30): the pipeline is decode-bound.** Decode costs 53 s per hour
  of video; the model costs 15 s.
- **Keeping every frame: the pipeline is inference-bound.** The model costs 449 s; decode costs
  53 s.

**So hardware decode only matters once sampling is in place**, and once it is, decode is the
main cost.

---

## 2. One more count: keyframes

Counted without decoding (read the packet flags only): **3,717 keyframes among 171,904 frames**.
That's one every 46 frames, about **every 1.5 seconds**. Keyframes can be decoded on their own,
so a `keyframes` policy can skip about 98% of frames without decoding them. That attacks exactly
the stage measured above as the bottleneck.

---

## 3. Plan this prediction is for

- **Laptop for everything.**
- **One short rented NVIDIA session (≤ 3 hours)**, only to add the NVIDIA lane: NVDEC, TensorRT
  and a rented-GPU cost figure.

---

## 4. Predictions

Each one says what would prove it wrong.

**P1 — Uniform 1 fps sampling stays decode-bound on every machine.** ViT-B/32 is small enough
that the model mostly waits for frames. *Wrong if:* at 1 fps, any lane's model time per hour of
video exceeds its decode time.

**P2 — Keyframe-only decoding is 10–30× faster than decoding every frame. That's less than the
46× that frame counts alone suggest**, because keyframes are the largest and most expensive
frames to decode. *Wrong if:* Phase 1 measures outside 10–30×.

**P3 — With keyframes, the bottleneck moves from decode to preprocessing plus the model.**
Keyframes give about 2,300 frames per hour of video. On the M5 that's about 8 s of preprocessing
plus about 10 s of model time, against a few seconds of decode. *Wrong if:* keyframe decode is
still the slowest stage.

**P4 — Hardware decode on the laptop (VideoToolbox) is 1.5–3× faster than CPU decode for
every-frame decoding, not more.** The RGB conversion and the copy back to memory still happen
on the CPU. Low confidence. *Wrong if:* outside 1.5–3×.

**P5 — The laptop does 1,000 hours of video in about 6 hours** (about 20 s per hour of video:
keyframes, M5 GPU model, stages overlapped). That's about 4× faster than the Phase 0 path, which
works out to about 22 hours per 1,000 hours. A 100–200 hour corpus takes under an hour.
*Wrong if:* Phase 6 measures more than 40 s per hour of video on this path.

**P6 — A rented NVIDIA GPU is 1–4× faster than the laptop end to end on the keyframes path. It
is not 10× faster**, because the remaining work (preprocessing, data movement) isn't where a big
GPU helps. *Wrong if:* the measured speedup is outside 1–4×.

**P7 — The GPU session fits in 3 hours**, since it only runs scripts that were already tested on
the laptop. *Wrong if:* it needs a second session.

**P8 — Higher resolution makes decode worse roughly in proportion to pixels.** 1080p has 2.6×
the pixels of this file, so every-frame laptop decode would be about 140 s per hour of video.
*Wrong if:* outside 100–200 s per hour.

---

## 5. What this changes in the plan

1. **Phase 1 matters most for speed**, because the gain comes from decode and sampling, not the
   model. Measure, in order:
   - keyframes
   - parallel decoding across titles
   - VideoToolbox
2. **Preprocessing is the next limit after decode.** It runs as a single PIL process at about
   300 fps, and the measurements were very noisy. Parallelise it.
3. **Keyframes are not evenly spaced.** Encoders also insert them at scene cuts, which may be
   exactly where a sample is wanted. Phase 1 has to show their retrieval quality against uniform
   1 fps before relying on them.

## 6. Known weaknesses of this prediction

- One title, one 10-minute window, one resolution (1024×768 is below streaming resolution).
- The laptop was shared with other work during the runs.
- Keyframe spacing depends on how each file was encoded, and other titles may differ.
- Every NVIDIA figure is an assumption until the GPU session measures it.
