# NUMBERS — every figure, traced to the file and seed that produced it

A figure that isn't in this table doesn't go in the README.

**Lanes:**
- `cpu-laptop`: Apple M5 CPU
- `gpu-apple-m5`: Apple M5 GPU via PyTorch MPS
- `hwdec-apple-m5`: Apple M5 hardware video decoder (VideoToolbox)
- `gpu-<model>`: rented NVIDIA GPUs

Lanes are never averaged together.

## Data counts (counted, not taken from docs)

| Item | Value | How it was obtained |
|---|---|---|
| NOTLD file size | 1,902,907,166 bytes | `ls -l` after download; matches the archive.org metadata |
| NOTLD stream | H.264, 1024×768, 29.957 fps, 5,738.6 s | `scenesafe.decode.cpu.probe` |
| Keyframes / frames, whole film | 3,717 / 171,904 (one every 46 frames, ~1.5 s) | counted from packet flags with PyAV demux, no decode |
| Corpus manifest | 35 titles, 49.9 h, 18.8 GB (5 per genre × 7 genres) | `corpus/manifest.json`, built by `scripts/build_manifest.py` |
| Corpus downloaded and verified (size + MD5) as of 2026-10-06 | 25 of 35 | `scripts/fetch_corpus.py`; the other 10 failed while archive.org returned HTTP 503 |
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

The derived per-hour figures and the predictions are in `PREDICTION.md`. They are predictions,
not measurements.
