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

## Phase 4d

Rules: Phase 4d amendment in `EVAL-PROTOCOL.md`, frozen in commit `5d0ae7f`. 14 fresh films (`corpus/manifest_4d.json`;
67,121 seconds, 3,116 scenes), 60 ads (`configs/ads.yaml` + `configs/ads_extra.yaml`), label guide
(`configs/label_guide.yaml`). Same v1 method for both models.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| precision@10, ViT-B/32 → SigLIP2 | 0.268 → 0.283; difference +0.015 [−0.013, +0.045]; 18 ads better, 28 same, 14 worse | (not hardware-dependent) | `results/phase4d/match_model_cpu-laptop_seed0.json` |
| nDCG@10, ViT-B/32 → SigLIP2 | 0.393 → 0.430; +0.037 [−0.031, +0.105] | — | same |
| Ads with no fitting scene in either model's picks | 20 of 60 | — | same |
| Pairs labelled (primary) | 960, of which 236 fit | — | `results/phase4d/labels_primary.json` |
| Label agreement, 100 pairs | 88%, κ 0.68 (97 undiscussed pairs: 89%, κ 0.70); fit rate 25% for both labellers | — | `results/phase4d/match_model_cpu-laptop_seed0.json`; `results/phase4d/SUMMARY.md` |
| Sensitivity (author's labels where present), precision@10 | +0.015 [−0.013, +0.043] | — | same |
| Picks that are mostly near-black | ViT-B/32 42 of 600; SigLIP2 32 of 600 | — | same |

## Phase 4e

Rules: Phase 4e amendment in `EVAL-PROTOCOL.md`, frozen in commit `6da2abb`. The 14 Phase 4d films, 60 ads, theme
lines (`configs/themes.yaml`). Three-level labels (2 literal, 1 thematic, 0 neither). Primary labels only.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| thematic precision@10, literal search → theme search | 0.525 → 0.543; +0.018 [−0.040, +0.075]; 24 ads better, 21 same, 15 worse | (not hardware-dependent) | `results/phase4e/match_theme_cpu-laptop_seed0.json` |
| literal precision@10, literal search → theme search | 0.287 → 0.218; −0.068 [−0.113, −0.028] | — | same |
| graded nDCG@10, literal search → theme search | 0.547 → 0.537; −0.011 [−0.072, +0.049] | — | same |
| Pairs labelled (primary) | 1,013: 612 × 0, 357 × 1, 44 × 2 | — | `results/phase4e/labels_primary.json` |

## Phase 4f

Rules: Phase 4f amendment in `EVAL-PROTOCOL.md`, frozen in commit `aaeea3a`. The 14 Phase 4d films (3,116 scenes,
1,639 with speech), 60 ads. Whisper large-v3-turbo transcripts. Primary labels only.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| thematic precision@10: frames / dialogue / both | 0.443 / 0.337 / 0.427 | (not hardware-dependent) | `results/phase4f/match_dialogue_cpu-laptop_seed0.json` |
| both − frames, thematic precision@10 | −0.017 [−0.087, +0.052]; 23 ads better, 9 same, 28 worse | — | same |
| literal precision@10: frames / dialogue / both; both − frames | 0.285 / 0.057 / 0.155; −0.130 [−0.180, −0.083] | — | same |
| graded nDCG@10: frames / dialogue / both; both − frames | 0.517 / 0.323 / 0.423; −0.094 [−0.184, −0.005] | — | same |
| Picks with speech: frames / dialogue / both | 357 / 600 / 593 of 600 | — | same |
| Pairs labelled (primary) | 1,425: 881 × 0, 317 × 1, 227 × 2 | — | `results/phase4f/labels_primary.json` |

## Phase 4g

Rules: Phase 4g amendment in `EVAL-PROTOCOL.md`, frozen by hash before the run (`results/phase4g/rules_hash.txt`).
The 12 Phase 4d films not used to develop the cards (2,635 scenes), 60 ads. Scene cards from Qwen3-VL-4B (4-bit,
mlx-vlm).

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| thematic precision@10: frames / cards / frames+cards | 0.482 / 0.670 / 0.692 | (not hardware-dependent) | `results/phase4g/match_cards_cpu-laptop_seed0.json` |
| frames+cards − frames, thematic precision@10 | +0.210 [+0.155, +0.268]; 42 ads better, 14 same, 4 worse | — | same |
| cards − frames, thematic precision@10 | +0.188 [+0.123, +0.255] | — | same |
| literal precision@10: frames / cards / frames+cards; frames+cards − frames | 0.275 / 0.252 / 0.340; +0.065 [+0.035, +0.097] | — | same |
| graded nDCG@10: frames / cards / frames+cards; frames+cards − frames | 0.502 / 0.659 / 0.736; +0.234 [+0.169, +0.297] | — | same |
| Pairs labelled (primary) | 1,249: 554 × 0, 403 × 1, 292 × 2 | — | `results/phase4g/labels_primary.json` |
| Label agreement, 51 pairs (pre-registered labels) | exact 63%; thematic-or-better κ 0.57; literal κ 0.30 | — | `results/phase4g/match_cards_cpu-laptop_seed0.json` |
| Sensitivity (author's labels where present), frames+cards − frames | +0.213 [+0.160, +0.268] | — | same |
| Relabel under the author's rules (written after the result): thematic precision@10 frames / cards / frames+cards | 0.553 / 0.753 / 0.750; frames+cards − frames +0.197 [+0.142, +0.255]; 39 better, 18 same, 3 worse | — | `results/phase4g/match_cards_relabel_cpu-laptop_seed0.json` |
| Relabel: label agreement, 51 pairs | exact 82%; thematic-or-better κ 0.72; literal κ 0.78 | — | same |
| Scene card speed | about 1.5 s per scene (Qwen3-VL-4B 4-bit) | gpu-apple-m5 | progress log of the card run; not in a results file. Re-measure before quoting it. |

## Phase 4h

Rules: Phase 4h amendment in `EVAL-PROTOCOL.md`, frozen by hash before the films were selected
(`results/phase4h/rules_hash.txt`). 7 fresh films (`corpus/manifest_4h.json`; 2,287 scenes), 60 ads. Labels under the
author's rules (Phase 4g addendum). Primary labels are the headline; the author labelled a 100-pair sample.

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| thematic precision@10: frames / frames+cards / rerank | 0.498 / 0.695 / 0.730 | (not hardware-dependent) | `results/phase4h/match_confirm_cpu-laptop_seed0.json` |
| frames+cards − frames, thematic precision@10 | +0.197 [+0.142, +0.253]; 45 ads better, 8 same, 7 worse | — | same |
| rerank − frames+cards, thematic precision@10 | +0.035 [+0.008, +0.062]; 23 better, 29 same, 8 worse | — | same |
| rerank − frames, thematic precision@10 | +0.232 [+0.180, +0.287]; 47 better, 9 same, 4 worse | — | same |
| literal precision@10: frames / frames+cards / rerank | 0.233 / 0.288 / 0.330; rerank − frames +0.097 [+0.055, +0.142] | — | same |
| graded nDCG@10: frames / frames+cards / rerank | 0.526 / 0.738 / 0.779 | — | same |
| Judge answers on 1,800 shortlist checks | 806 × 0, 312 × 1, 682 × 2, 0 unreadable | — | same |
| Judge vs labels, 893 pairs | exact 53%; fits vs doesn't fit 74% | — | computed from `data/phase4h/judge.json` and `results/phase4h/labels_primary.json`; not in a results file. Re-compute before quoting it. |
| Pairs labelled (primary) | 1,086: 461 × 0, 386 × 1, 239 × 2 | — | `results/phase4h/labels_primary.json` |
| Author vs primary labels, 100 pairs | exact 64%; thematic-or-better κ 0.50; literal κ 0.59 | — | `results/phase4h/match_confirm_cpu-laptop_seed0.json` (agreement) |
| Sensitivity (author's labels where present): frames+cards − frames | +0.202 [+0.147, +0.258]; 44 better, 11 same, 5 worse | — | same (sensitivity_author_labels_where_present) |
| Sensitivity (author's labels where present): rerank − frames+cards | +0.022 [−0.003, +0.047]; 20 better, 31 same, 9 worse (interval includes 0) | — | same |
| Relabel under the Phase 4h addendum (written after the result): thematic precision@10 frames / frames+cards / rerank | 0.473 / 0.662 / 0.697 | — | `results/phase4h/match_confirm_relabel_cpu-laptop_seed0.json` |
| Relabel: frames+cards − frames; rerank − frames+cards | +0.188 [+0.138, +0.240], 45 / 7 / 8; +0.035 [+0.007, +0.065], 22 / 27 / 11 | — | same |
| Relabel with the author's labels where present: frames+cards − frames; rerank − frames+cards | +0.192 [+0.143, +0.243]; +0.027 [−0.000, +0.055] | — | same (sensitivity_author_labels_where_present) |
| Relabel vs author, 100 pairs (not independent: rules written from these labels) | exact 75%; thematic-or-better κ 0.65; literal κ 0.71 | — | same (agreement) |
| Repeat check, greedy decoding | cards 30/30 identical, judge 60/60 identical | gpu-apple-m5 | `results/phase4h/repeat.json` |
| Judge speed | about 0.3 s per shortlist check (Qwen3-VL-4B 4-bit) | gpu-apple-m5 | progress log of the test step; not in a results file. Re-measure before quoting it. |

The derived per-hour figures and the predictions are in `PREDICTION.md`. They are predictions,
not measurements. The 1,000-hour search projection (about 95 ms per query, 7.4 GB) in
`results/phase3/SUMMARY.md` is a linear extrapolation, not a measurement.
