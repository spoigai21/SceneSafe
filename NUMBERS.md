# NUMBERS — every figure, traced to the file and seed that produced it

A figure that isn't in this table doesn't go in the README.

**Lanes:**
- `cpu-laptop`: Apple M5 CPU
- `gpu-apple-m5`: Apple M5 GPU via PyTorch MPS
- `hwdec-apple-m5`: Apple M5 hardware video decoder (VideoToolbox)
- `ane-apple-m5`: Core ML allowed to use the M5 Neural Engine (actual placement not confirmed)
- `gpu-<model>`: rented NVIDIA GPUs

Lanes are never averaged together.

## Data counts (counted, not taken from docs)

| Item | Value | How it was obtained |
|---|---|---|
| NOTLD file size | 1,902,907,166 bytes | `ls -l` after download; matches the archive.org metadata |
| NOTLD stream | H.264, 1024×768, 29.957 fps, 5,738.6 s | `scenesafe.decode.cpu.probe` |
| Keyframes / frames, whole film | 3,717 / 171,904 (one every 46 frames, ~1.5 s) | counted from packet flags with PyAV demux, no decode |
| Corpus manifest | 35 titles, 49.9 h, 18.8 GB (5 per genre × 7 genres) | `corpus/manifest.json`, built by `scripts/build_manifest.py` |
| Corpus downloaded and verified (size + MD5) | 35 of 35 (25 on 2026-10-06, the remaining 10 on 2026-10-08) | `scripts/fetch_corpus.py` |
| Corpus frames at 1 fps | 179,545 | counted, `results/phase2/corpus_embed_gpu-apple-m5_seed0.json` |
| Phase 1 test set | 7 windows × 26,973 frames = 188,811 frames (1.75 h) | counted by decoding, `results/phase1/quality_gpu-apple-m5_seed0.json` |
| Keyframes in Phase 1 test set | 1,345 (one per 4.7 s) | counted by decoding, `results/phase1/decode_cpu-laptop_seed0.json` |
| Frames in window 600–1200 s | 17,983 | counted by decoding, `results/phase0/decode_cpu-laptop_seed0.json` |

## Phase 0

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| Decode rate, all frames, incl. RGB conversion | 2,045 fps median (1,838–2,192), n=5 | cpu-laptop | `results/phase0/decode_cpu-laptop_seed0.json` |
| CLIP ViT-B/32 fp32 eager, batch 32 | 107 fps median (106–117), n=5 | cpu-laptop | `results/phase0/embed_cpu-laptop_seed0.json` |
| CLIP ViT-B/32 fp32 eager, batch 32 | 240 fps median (206–255), n=5 | gpu-apple-m5 | `results/phase0/embed_gpu-apple-m5_seed0.json` |
| Preprocess (PIL, 1 process) | 338 / 285 fps median; range 79–444 (noisy) | cpu-laptop | both `embed_*` files, `preprocess_fps` |

## Phase 1

The machine was heavily shared during timing (load average 8.6–16 on 10 cores). Medians of 3 runs.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| Decode, every frame (incl. RGB) | 51.6 s/video-hour (45.6–52.5) | cpu-laptop | `results/phase1/decode_cpu-laptop_seed0.json` |
| Decode, uniform 1 fps | 11.6 s/video-hour (11.3–12.5); 8.9 with 2 processes | cpu-laptop | same |
| Decode, keyframes | 1.6 s/video-hour (1.6–1.8); 0.7 with 7 processes | cpu-laptop | same |
| Decode, scene change @0.05 | 57.9 s/video-hour (55.1–62.0) | cpu-laptop | same |
| Decode, every frame | 96.7 s/video-hour (95.9–98.4) | hwdec-apple-m5 | `results/phase1/decode_hwdec-apple-m5_seed0.json` |
| Decode, uniform 1 fps | 61.0 s/video-hour (60.7–61.7) | hwdec-apple-m5 | same |
| Decode, keyframes | 2.0 s/video-hour (1.9–2.1) | hwdec-apple-m5 | same |
| Recall@10 / @50, uniform 1 fps vs every frame | 0.630 / 0.736 | (not hardware-dependent) | `results/phase1/quality_gpu-apple-m5_seed0.json` |
| Recall@10 / @50, keyframes | 0.425 / 0.570 | — | same |
| Recall@10, uniform 0.25 / 0.5 / 2 fps | 0.450 / 0.535 / 0.695 | — | same |
| Recall@10, scene change @0.03 / 0.05 / 0.08 / 0.12 | 0.715 / 0.575 / 0.500 / 0.420 | — | same |
| On-device vs PIL preprocessing, embedding cosine | 0.995 mean, 0.987 min (30 frames) | gpu-apple-m5 | docstring of `ClipEmbedder.preprocess_on_device`; not in a results file. Re-measure before quoting it. |

## Phase 2

The machine was shared during timing (load average 3.2–5.2 on 10 cores). Medians of 3 runs, best batch size.
Accuracy is measured against `pytorch-fp32-mps` on 6,300 test frames (20 searches, 10-s segments).

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| pytorch-fp32-mps | 307 fps (303–310), b128 | gpu-apple-m5 | `results/phase2/inference_gpu-apple-m5_seed0.json` |
| pytorch-fp16-mps | 987 fps (886–1,001), b128; cosine min 0.99999; R@10 1.000, R@50 0.997 | gpu-apple-m5 | same |
| coreml-fp32-gpu | 261 fps (260–262), b32; R@10 1.000 | gpu-apple-m5 | same |
| coreml-fp32-ane | 154 fps (154–154), b32; R@10 1.000 | ane-apple-m5 | `results/phase2/inference_ane-apple-m5_seed0.json` |
| pytorch-fp32-cpu | 130 fps (125–130), b128 | cpu-laptop | `results/phase2/inference_cpu-laptop_seed0.json` |
| onnxruntime-fp32-cpu | 125 fps (125–127), b8; R@10 1.000 | cpu-laptop | same |
| onnxruntime-int8dynamic-cpu | 130 fps (127–134), b8; cosine mean 0.960; R@10 0.720 | cpu-laptop | same |
| onnxruntime-int8static-cpu | 105 fps (104–107), b128; cosine mean 0.635; R@10 0.215 | cpu-laptop | same |
| Host→GPU transfer share (MPS) | 1–4% of inference time | gpu-apple-m5 | `transfer_share` in the gpu file |
| Full corpus end to end (decode 1 fps + fp16 MPS) | 49.9 video-hours in 8.5 min = 10.2 s per video-hour (8.2–11.2 per title) | cpu-laptop decode + gpu-apple-m5 model, one machine | `results/phase2/corpus_embed_gpu-apple-m5_seed0.json` |
| fp16 vs fp32 spot check, 3 titles × 512 frames | cosine mean ≥ 0.99999, min 0.99938 | gpu-apple-m5 | same |

## Phase 3

CPU only, FAISS 1.15.1, load average 2.2–3.0 on 10 cores. 100 text queries (`configs/phase3_queries.yaml`),
1,000 image queries (random corpus seconds, seed 0). Latency on 1 thread, throughput on 10.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| Detected scenes, threshold 0.75 / 0.80 / 0.85 (min 3 s) | 10,241 / 16,143 / 22,738; median length 7 / 6 / 5 s | (not hardware-dependent) | `results/phase3/units_cpu-laptop_seed0.json` |
| Separate moments in top 10: every second / fixed 10 s / pooled scene@0.75 / second→scene@0.75 | 5.67 / 7.78 / 8.73 / 8.66 | — | same |
| 10 best moments found: every second / fixed 10 s / pooled scenes / second→scene@0.75 | 5.68 / 2.89 / 3.30–4.43 / 9.26 | — | same |
| second→scene@0.75 latency (exact, top 1,000 then grouped) | 4.94 ms p50, 5.22 ms p99 | cpu-laptop | same |
| Exact (flat) search, 179,545 vectors | 368 MB; 4.72 ms p50, 5.09 ms p99; 7,451 queries/s batched | cpu-laptop | `results/phase3/index_cpu-laptop_seed0.json` |
| IVF2048, nprobe 8 / 32 / 64 / 128: recall@10 text | 0.683 / 0.856 / 0.915 / 0.961 (image queries: 0.970 / 0.995 / 0.999 / 1.000) | cpu-laptop | same |
| IVF2048, nprobe 64: latency, size | 0.251 ms p50; 373 MB | cpu-laptop | same |
| IVF2048-PQ64 / PQ32: size, best text recall@10 (nprobe 128) | 17.7 MB, 0.453 / 11.9 MB, 0.320 | cpu-laptop | same |
| Exact search latency vs size (1 thread, p50) | 17,954: 0.44 ms; 44,886: 1.15; 89,772: 2.33; 179,545: 4.72 | cpu-laptop | same, `flat_scaling` |

## Phase 4

Rules: `EVAL-PROTOCOL.md` (commit `553418d`). 14 held-out films (71,091 s, 3,911 scenes), 20 simulated ads,
589 pooled (ad, scene) pairs. Primary labels by an AI assistant (Claude) blind to method; 100 re-labelled by
the project author. Means over 20 ads, 95% bootstrap intervals over ads.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| precision@10: CLIP / Genre / Random | 0.440 [0.315, 0.575] / 0.075 [0.020, 0.145] / 0.020 [0.005, 0.040] | (not hardware-dependent) | `results/phase4/match_cpu-laptop_seed0.json` |
| nDCG@10: CLIP / Genre / Random | 0.702 / 0.102 / 0.027 | — | same |
| CLIP − Genre, precision@10 | +0.365 [+0.255, +0.485]; 18 ads won, 2 tied, 0 lost | — | same |
| Label agreement (100 pairs) | 91%, Cohen's kappa 0.68; fit rate 13% (primary) vs 20% (author) | — | same, `agreement` |
| Sensitivity (author labels where present) | CLIP 0.46, Genre 0.09; CLIP − Genre +0.37 [+0.26, +0.49] | — | computed from `labels_*.json`; in `results/phase4/SUMMARY.md` |
| Near-black scenes in the labelling pool | 29 of 466 scenes (35 of 589 pairs) | — | mean brightness < 20 of the 4-frame strips |

## Phase 4b

Rules: Phase 4b amendment in `EVAL-PROTOCOL.md`, frozen in commit `eebe7a5`. Same 14 test films and 20 ads as
Phase 4. Cut-off tau = 0.146 from 300 labelled practice-film pairs (`results/phase4b/calibration.json`).

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| precision@10, v1 → v2 | 0.440 → 0.450; v2 − v1 +0.010 [−0.055, +0.085]; 4 ads better, 10 same, 6 worse | (not hardware-dependent) | `results/phase4b/match_v2_cpu-laptop_seed0.json` |
| nDCG@10, v1 → v2 | 0.644 → 0.712; +0.068 [−0.018, +0.164] | — | same |
| Precision of returned scenes, v1 → v2 | 0.44 (88/200) → 0.52 (90/174) | — | same |
| Ads with nothing returned (v2) | 1 (beach), correct | — | same |
| Label agreement, 50 new pairs | 84%, κ 0.69 (47 undiscussed pairs: 83%, κ 0.67) | — | same; `results/phase4b/SUMMARY.md` |
| Junk seconds (pixel rule), practice films | 3,030 of 108,454 (2.8%) | — | `scripts/run_phase4b.py practice-pool` output |

## Phase 4c

Rules: Phase 4c amendment in `EVAL-PROTOCOL.md`, frozen in commit `ec47455`. Same 14 test films, 20 ads, Phase 3
scenes and v1 method; only the model changes (CLIP ViT-B/32 → SigLIP2 ViT-B/16, OpenCLIP `webli`).

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| precision@10, ViT-B/32 → SigLIP2 | 0.440 → 0.465; difference +0.025 [−0.045, +0.095]; 8 ads better, 6 same, 6 worse | (not hardware-dependent) | `results/phase4c/match_model_cpu-laptop_seed0.json` |
| nDCG@10, ViT-B/32 → SigLIP2 | 0.572 → 0.635; +0.063 [−0.054, +0.189] | — | same |
| Sensitivity (author's labels where present), precision@10 | 0.460 → 0.510; +0.050 [−0.040, +0.145] | — | same |
| Label agreement, 50 new pairs | 82%, κ 0.61 (49 undiscussed pairs: 82%, κ 0.60) | — | same; `results/phase4c/SUMMARY.md` |
| Picks that are mostly near-black (> half their seconds junk by the 4b pixel rule) | ViT-B/32 18 of 200; SigLIP2 5 of 200 | — | `results/phase4c/match_model_cpu-laptop_seed0.json` |

The derived per-hour figures and the predictions are in `PREDICTION.md`. They are predictions,
not measurements. The 1,000-hour search projection (about 95 ms per query, 7.4 GB) in
`results/phase3/SUMMARY.md` is a linear extrapolation, not a measurement.
