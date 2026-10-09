# Phase 2 — embed: results

**Question:** what does making inference fast cost in quality?

## Setup

**Model:** one throughout, CLIP ViT-B/32 (OpenCLIP `laion2b_s34b_b79k`), image encoder only.
Only the inference configuration changes: runtime × precision × device.

**Inputs:** the Phase 1 test windows (7 films, 1.75 h) sampled at 1 fps, which is **6,300 frames**.
They were resized and normalised once, so every configuration saw identical pixels.

**int8 calibration:** 336 frames from **7 other films** (one per genre, none from the test set),
listed in `calibration_titles` in the results files.

**Reference:** `pytorch-fp32-mps`, the configuration used in Phases 0–1.

**Accuracy measures:**
- cosine similarity of each frame's embedding to the reference (1 = identical)
- recall@10 and @50 of the 20 Phase 1 searches over 10-second segments

**Speed:** frames/sec on a fixed 1,024-frame subset, 3 timed runs after one untimed warm-up pass,
at batch sizes 1, 8, 32 and 128.

**Machine:** Apple M5 laptop. Load average 3.2–5.2 on 10 cores during timing. That's lighter than
in Phase 1, but the machine was still shared.

## Results

The best batch size is shown for each configuration. All batch sizes are in the results files.

| Configuration | Lane | Frames/sec (best batch) | Cosine to fp32 (mean / 1st pct / min) | Recall@10 | Recall@50 |
|---|---|---|---|---|---|
| pytorch-fp32-mps (**reference**) | gpu-apple-m5 | 307 (303–310), b128 | 1 / 1 / 1 | 1.000 | 1.000 |
| **pytorch-fp16-mps** | gpu-apple-m5 | **987** (886–1,001), b128 | 0.999997 / 0.99999 / 0.99999 | **1.000** | 0.997 |
| coreml-fp32-gpu | gpu-apple-m5 | 261 (260–262), b32 | 1.000000 / 1.00000 / 1.00000 | 1.000 | 1.000 |
| coreml-fp32-ane | ane-apple-m5 | 154 (154–154), b32 | 1.000000 / 1.00000 / 1.00000 | 1.000 | 1.000 |
| pytorch-fp32-cpu | cpu-laptop | 130 (125–130), b128 | 1.000000 / 1.00000 / 1.00000 | 1.000 | 1.000 |
| onnxruntime-fp32-cpu | cpu-laptop | 125 (125–127), b8 | 1.000000 / 1.00000 / 1.00000 | 1.000 | 1.000 |
| onnxruntime-int8dynamic-cpu | cpu-laptop | 130 (127–134), b8 | 0.960 / 0.901 / 0.854 | 0.720 | 0.808 |
| onnxruntime-int8static-cpu | cpu-laptop | 105 (104–107), b128 | 0.635 / 0.460 / 0.354 | **0.215** | 0.373 |

Sources: `inference_{cpu-laptop,gpu-apple-m5,ane-apple-m5}_seed0.json`. Chart: `frontier.png`.

**Where the time goes on the GPU:** copying frames from memory to the GPU takes **1–4%** of the time
at every batch size (`transfer_share`). Inference is compute-bound, not transfer-bound.

## Full corpus, with the chosen configuration

**Pipeline:** CPU decode at 1 fps in a background thread → on-device resize → CLIP fp16 on MPS,
batch 128.

| Figure | Value |
|---|---|
| Titles | 35 (all verified by size and MD5) |
| Video | 49.9 hours |
| Frames embedded | 179,545 |
| Wall-clock | **8.5 minutes** |
| End-to-end | **10.2 s per video-hour** (8.2–11.2 across titles) |
| Spot check (512 frames × 3 titles, fp16 vs fp32) | cosine mean ≥ 0.99999, min 0.99938 |
| Stored | 177 MB (`data/embeddings/`, float16) |

Source: `corpus_embed_gpu-apple-m5_seed0.json`.

## Findings

1. **fp16 is free speed: 3.2× faster than fp32 on the M5 GPU, with identical search results**
   (recall@10 = 1.000; the lowest cosine on any frame is 0.99999).
2. **int8 lost on both counts on this laptop. This is a negative result:**
   - **Static int8 (calibrated, MinMax)** wrecked quality: cosine 0.63, recall@10 0.215. About 4 in
     5 searches return different scenes. It was also *slower* than fp32 on the CPU (105 vs 125 fps),
     so a better calibration method could not make it competitive. **This is the "faster and
     quietly worse" trap, except it wasn't even faster.**
   - **Dynamic int8** was less damaging (recall@10 0.72) but no faster than fp32 on the CPU.
3. **The runtime matters less than the chip.** ONNX Runtime on the CPU ran at the same speed as
   PyTorch on the CPU (125 vs 130 fps). Core ML on the GPU was slightly *slower* than PyTorch on the
   GPU (261 vs 307).
4. **Apple's Neural Engine route (Core ML) reached 154 fps, half the GPU's speed.** Its outputs
   matched fp32 *exactly*, which is unexpected: the Neural Engine computes in fp16. Core ML decides
   for itself where each operation runs, so **we cannot confirm the Neural Engine was actually
   used**. Treat this row as "Core ML with the Neural Engine allowed".
5. **Bigger batches help, up to about 32.** On the GPU, fp16 goes from 180 fps at batch 1 to 905 at
   batch 32 and 987 at batch 128.
6. **After this phase the pipeline is decode-bound again.** The model now needs about 3.6 s per
   video-hour (3,600 frames ÷ 987 fps). 1 fps CPU decode needs about 9–12 s (Phase 1), and the
   whole corpus ran at 10.2 s per video-hour, which is about the decode cost.

## Decision

**Inference configuration: PyTorch · fp16 · MPS (M5 GPU) · batch 128.** It is the fastest by 3×,
and the only configuration with a measurable accuracy cost of any size is int8. Every later phase
uses the corpus embeddings made with it.

## Predictions checked (from `PREDICTION.md`)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P3 | With fewer frames, the bottleneck moves to preprocessing + model | It did after Phase 1. After fp16 it moved **back to decode** (finding 6). | Right for Phase 1; superseded |
| P5 | The laptop does 1,000 video-hours in ~6 h (~20 s per video-hour); wrong if > 40 s | Measured **10.2 s per video-hour**, so about **2.8 h per 1,000 video-hours** | **Wrong, in the good direction.** 2× faster than predicted. The formal check is Phase 6. |
| P6 | A rented NVIDIA GPU is 1–4× faster than the laptop end to end | — | Untested until the GPU session |

## Limits

- One model (ViT-B/32). Larger models would shift the balance back towards inference.
- Static int8 used MinMax calibration only. Percentile or entropy calibration might recover
  accuracy, but static int8 was already slower than fp32 on this CPU, so it was not pursued.
- Core ML placement is opaque (finding 4).
- The machine was shared during timing (see Setup).
- TensorRT (fp32 / fp16 / int8) is deferred to the single NVIDIA GPU session.
