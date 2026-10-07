# NUMBERS — every figure, traced to the file and seed that produced it

A figure that isn't in this table doesn't go in the README.

**Lanes:**
- `cpu-laptop`: Apple M5 CPU
- `gpu-apple-m5`: Apple M5 GPU via PyTorch MPS
- `gpu-<model>`: rented NVIDIA GPUs

Lanes are never averaged together.

## Data counts (counted, not taken from docs)

| Item | Value | How it was obtained |
|---|---|---|
| NOTLD file size | 1,902,907,166 bytes | `ls -l` after download; matches the archive.org metadata |
| NOTLD stream | H.264, 1024×768, 29.957 fps, 5,738.6 s | `scenesafe.decode.cpu.probe` |
| Keyframes / frames, whole film | 3,717 / 171,904 (one every 46 frames, ~1.5 s) | counted from packet flags with PyAV demux, no decode |
| Frames in window 600–1200 s | 17,983 | counted by decoding, `results/phase0/decode_cpu-laptop_seed0.json` |

## Phase 0

| Figure | Value | Lane | Source (seed 0) |
|---|---|---|---|
| Decode rate, all frames, incl. RGB conversion | 2,045 fps median (1,838–2,192), n=5 | cpu-laptop | `results/phase0/decode_cpu-laptop_seed0.json` |
| CLIP ViT-B/32 fp32 eager, batch 32 | 107 fps median (106–117), n=5 | cpu-laptop | `results/phase0/embed_cpu-laptop_seed0.json` |
| CLIP ViT-B/32 fp32 eager, batch 32 | 240 fps median (206–255), n=5 | gpu-apple-m5 | `results/phase0/embed_gpu-apple-m5_seed0.json` |
| Preprocess (PIL, 1 process) | 338 / 285 fps median; range 79–444 (noisy) | cpu-laptop | both `embed_*` files, `preprocess_fps` |

The derived per-hour figures and the predictions are in `PREDICTION.md`. They are predictions,
not measurements.
