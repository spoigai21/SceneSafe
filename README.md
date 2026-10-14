# SceneSafe

Finds the film scenes that suit an ad, and keeps brands away from the scenes that don't. It was built and measured on
one laptop, with every number traced to a results file and every test's rules fixed before it ran.

**Results site: https://spoigai21.github.io/SceneSafe/** (the same page is in `site/`).

## Honest limits (read these first)

1. **The video is public-domain film** from the Internet Archive (35 titles, 49.9 hours), not any streaming service's
   catalogue. Nothing here is a claim about how any company's ads perform.
2. **The ads and the ad demand are simulated.** There is no public corpus of video ad creatives with outcomes. The 60
   ads are written briefs, and the Phase 7 advertisers' values and budgets are assumptions. Every result that depends on
   them says so.
3. **Every number names its hardware.** The lanes are:
   - laptop CPU (`cpu-laptop`)
   - laptop GPU (`gpu-apple-m5`)
   - rented NVIDIA

   They are never blended into one figure. No NVIDIA lane has been measured yet.
4. **Speed quality is measured against exact fp32 search** on the same embeddings, not an external leaderboard.
5. **Match and safety quality come from labels by an AI assistant**, blind to which method produced each result, with
   samples checked by the author. They are not large human panels.

## What it found

All figures are from `NUMBERS.md`. Losses are listed with the wins.

| Question | Answer |
|---|---|
| Does adding scene descriptions improve matching? | **Yes.** 48% → 69% of an ad's top 10 scenes suit it (+21 points, 95% interval +16 to +27; 42 of 60 ads better). The gain held on 7 fresh films (50% → 70%). |
| Did other ideas help? | Searching seconds and grouping them into scenes, yes. Re-ranking a shortlist, a small gain. A newer image model (SigLIP2): no clear change. Dialogue search **hurt** literal matches (−13 points), and searching by an ad's theme **hurt** them too (−7). |
| Can it keep brands safe? | Partly. At standard settings it blocks 6% of scenes and catches 71% of risky ones, but 64% of what it blocks was actually fine. Weapons: caught 8 of 9 (from 0 of 9). **Violence: still missed** (2 of 11) with every local model tried. |
| What does safety cost in ad space? | Under standard rules a busy ad schedule stays 76% full, against 86% with no rules (measured with the safety version before the final weapons check). |
| How fast is it? | The 49.9-hour library was indexed in 8.5 minutes (CPU decode + M5 GPU). fp16 runs CLIP at 987 frames per second with no loss in search accuracy. The Mac hardware video decoder was **slower** than CPU decoding at 1 fps. |
| What does it cost per 1,000 hours of video? | **185 machine-hours, about $227**, on an M5-class Mac at a cloud Mac's list price ($1.23/hour, 13 Oct 2026; assumes an M4 runs as fast as the M5). The AI models are 93% of that. Frame search alone costs about $3. The whole pipeline costs **1.65×** the naive every-frame approach. |
| Does knowing the scene make ad space worth more? | **Only sometimes** (simulated market). Selling by safety × topic earned 1.13× a flat pool when demand was light (10 of 10 simulations). Under heavy demand it sold 25% fewer breaks and won only 7 of 10, which fails the pre-registered bar. 7% of breaks sit next to scenes no advertiser here accepts. |

## How it works

1. **Frames:** decode 1 frame per second (PyAV) and embed it with CLIP ViT-B/32 in fp16 on the laptop GPU. Group the
   seconds into scenes, and search them with FAISS.
2. **Scene descriptions:** a small vision-language model (Qwen3-VL-4B, 4-bit, mlx-vlm) writes a short card for each
   scene from 4 frames and the scene's dialogue (Whisper large-v3-turbo). Ads match against frames + cards.
3. **Brand safety:** the same model tags 9 content types per scene. A per-tag yes/no check confirms each tag, and a
   sharper 2 × 2 frame check handles weapons. Three rule sets (lenient, standard, strict), plus adjacency norms by ad
   category, are applied to both scenes around each ad break.
4. **Inventory:** ad breaks are grouped by safety class × topic and sold to simulated advertisers.

## Repository

| Path | What |
|---|---|
| `SPEC.md` | the plan: phases, questions, ground rules |
| `EVAL-PROTOCOL.md` | every test's rules, written and hash-frozen before it ran (`results/*/rules_hash.txt`) |
| `PREDICTION.md` | predictions written before the measurements, never edited |
| `NUMBERS.md` | every figure, with the results file and seed behind it |
| `BUGS.md` | what broke, what caught it, and the fix |
| `results/phase*/` | results JSON per run, and a `SUMMARY.md` per phase |
| `scenesafe/` | the package: decode, embed, index, match, safety |
| `scripts/run_phase*.py` | one script per phase; each docstring lists its steps |
| `configs/` | ads, safety rule sets, prices, Phase 7 parameters |
| `site/` | the results website (static) |

## Re-running

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/fetch_corpus.py          # 35 public-domain films, about 18.8 GB, MD5-checked
PYTHONPATH=. .venv/bin/python scripts/run_phase<N>.py <step>   # see each script's docstring
.venv/bin/pytest
```

The scene-description, safety and Whisper steps need Apple silicon (mlx). Large data (frames, embeddings, transcripts,
cards) is re-derived and kept out of git.
