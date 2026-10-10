# Phase 4 — contextual matching: results

**Question:** is the signal real, or is this an expensive way to do nothing?

**Rules:** `EVAL-PROTOCOL.md`, committed before any result (commit `553418d`).

## Setup

**Test films:** 14 held-out films, 2 per genre (`configs/phase4_split.yaml`): 71,091 seconds and
3,911 detected scenes.

**Ads:** 20 simulated ads (`configs/ads.yaml`).

**Methods:** each returns 10 scenes per ad.
- **CLIP:** the brief's text embedding; exact search over every second, grouped into scenes (the
  Phase 3 decision).
- **Genre:** random moments from films of the ad's chosen genres.
- **Random:** random moments from any test film.

**Labelling:**
- **Pool:** 589 unique (ad, scene) pairs, shuffled, shown as 4 frames per scene, with no indication of
  the method.
- **Primary labeller:** an AI assistant (Claude), which also wrote the pipeline. It labelled all 589,
  blind to the method.
- **Second labeller:** the project author labelled a random 100 without seeing the primary labels.

## Results

All values are means over the 20 ads, with 95% bootstrap intervals over ads (10,000 resamples).

| Method | precision@10 | nDCG@10 |
|---|---|---|
| **CLIP** | **0.440** [0.315, 0.575] | **0.702** [0.565, 0.822] |
| Genre only | 0.075 [0.020, 0.145] | 0.102 [0.027, 0.197] |
| Random | 0.020 [0.005, 0.040] | 0.027 [0.005, 0.054] |

| Paired difference | precision@10 | Ads won / tied / lost | Decision rule |
|---|---|---|---|
| CLIP − Genre | **+0.365** [+0.255, +0.485] | 18 / 2 / 0 | **CLIP beats genre** (interval above 0) |
| CLIP − Random | +0.420 [+0.295, +0.550] | 18 / 2 / 0 | CLIP beats random |

Source: `match_cpu-laptop_seed0.json` (seed 0). Chart: `match.png`.

**Per ad, precision@10:**

| Ad | Fitting scenes found by any method | CLIP | Genre | Random |
|---|---|---|---|---|
| phone | 10 | 1.0 | 0.0 | 0.0 |
| car | 9 | 0.9 | 0.0 | 0.0 |
| western_wear | 13 | 0.9 | 0.4 | 0.0 |
| sci_fi_game | 15 | 0.9 | 0.5 | 0.1 |
| restaurant | 7 | 0.7 | 0.0 | 0.0 |
| coffee | 6 | 0.5 | 0.0 | 0.1 |
| fashion | 6 | 0.5 | 0.3 | 0.1 |
| pharmacy | 5 | 0.5 | 0.0 | 0.0 |
| beer | 4 | 0.4 | 0.0 | 0.0 |
| wine | 4 | 0.4 | 0.0 | 0.0 |
| jewelry | 4 | 0.4 | 0.0 | 0.0 |
| outdoor_gear | 4 | 0.4 | 0.0 | 0.0 |
| cruise | 3 | 0.3 | 0.0 | 0.0 |
| music | 4 | 0.3 | 0.2 | 0.1 |
| furniture | 4 | 0.3 | 0.1 | 0.0 |
| airline | 2 | 0.2 | 0.0 | 0.0 |
| toys | 1 | 0.1 | 0.0 | 0.0 |
| bank | 1 | 0.1 | 0.0 | 0.0 |
| beach_resort | 0 | 0.0 | 0.0 | 0.0 |
| pet_food | 0 | 0.0 | 0.0 | 0.0 |

## Label agreement

On the 100 pairs both labellers judged:

- **agreement: 91%; Cohen's kappa: 0.68**
- **fit rate:** primary 13%, second 20%. The second labeller was more lenient: in 8 of the 9
  disagreements, the second labeller said "fits" and the primary said "doesn't fit". The disagreements
  are listed in the results file.

**Sensitivity check** (not the protocol's primary metric): using the second labeller's labels wherever
they exist gives CLIP 0.46, Genre 0.09, Random 0.02. CLIP − Genre is +0.37 [+0.26, +0.49]. **The
conclusion doesn't change.**

## Findings

1. **Scene understanding beats genre metadata clearly.**
   - CLIP's picks fit **44%** of the time, against 7.5% for genre-only and 2% for random.
   - It won on 18 of 20 ads and lost on none.
2. **But fewer than half of CLIP's picks fit.** This is a usable signal, not a solved problem.
   - **Strong ads have a distinctive, visible subject:** phone, car, horses, spaceships (0.9–1.0).
   - **Weak ads:**
     - airline (0.2): CLIP confuses spaceship cockpits with airplanes
     - toys and bank (0.1)
3. **CLIP has no "nothing here" signal.**
   - No labeller found a single fitting scene for the beach or dog ads in these 14 films. CLIP still
     returned 10 scenes for each, all wrong.
   - A search always returns its top 10, however bad they are. A real system needs a minimum score, or
     it sells bad placements. That's for Phases 5 and 7.
4. **Genre helps only when the ad *is* the genre:** western wear in westerns (0.4), the sci-fi game in
   sci-fi (0.5). For everyday subjects (phones, cars, meals), a random scene from the "right" genre
   almost never shows the thing.
5. **The scene detector turns fades into "scenes".**
   - 29 of the 466 scenes in the pool were almost black: fades, night shots, end cards.
   - They aren't usable ad slots and should be dropped or merged into a neighbour before inventory is
     built (Phase 7).
   - This was raised by the second labeller.

## Deviations from the protocol

None in method or metrics.

One display bug (BUGS.md #8): for scenes that run to the end of a film, the last of the 4 frames could
fall past the final frame and show blank (1–2 of 4 frames, on a handful of pairs). Every method's scenes
were shown the same way. The bug is fixed for future runs, and the labelled strips weren't regenerated.

## Limits

- **The primary labeller built the system.** It was blind to the method, and a human second labeller
  agreed 91% (κ 0.68), but this is not an independent panel.
- **Judged from 4 frames, without context.** Placement in a real film depends on what comes before and
  after (raised by the second labeller). Phase 5's adjacency rules and Phase 7's breaks address that;
  this phase only measures "does this scene match".
- **The genre baseline is one reasonable definition** (random scenes from the ad's genres). A smarter
  genre rule might do better, but it would still know nothing about individual scenes.
- **Small and simulated:** 20 simulated ads, 14 public-domain films and 1 seed. The intervals are wide
  (±0.12).
- **nDCG's ideal ranking only knows the fitting scenes some method found** (pooled judging), so nDCG is
  relative to the pool, not to the whole catalogue.
