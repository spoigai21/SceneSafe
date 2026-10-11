# Phase 4d — the model comparison, re-run bigger on fresh films: results

**Question:** with the method held fixed, does SigLIP2 ViT-B/16 pick better scenes than CLIP ViT-B/32?
- **Why again:** Phase 4c asked this with 20 ads on reused films and couldn't tell.
- **What changed:** this time with 60 ads, 14 never-used films and a written label guide.

**Rules:** the Phase 4d amendment in `EVAL-PROTOCOL.md`, committed in `5d0ae7f` before any new film was embedded
or viewed.

## Setup

**Films:** 14 fresh films (`corpus/manifest_4d.json`), 2 per genre, chosen by the corpus rules.
- 67,121 seconds
- 3,116 scenes (Phase 3 detector)

**Ads:** 60.
- **The original 20** (`configs/ads.yaml`).
- **40 new** (`configs/ads_extra.yaml`), written before any frame was seen.
- **Label guide** (`configs/label_guide.yaml`): a "fits" and a "doesn't fit" line for every ad, shown to both
  labellers on every card.

**Method:** each model ran the v1 method once:
- the brief as written
- exact search over every second
- each scene once, at its best second
- top 10 per ad

**Labels:**
- **The pool:** 1,200 picks gave 960 unique (ad, scene) pairs. The two models chose the same scene for an ad 240
  times.
- **Primary labeller:** labelled all 960, blind to the model. 236 fit.
- **Second labeller:** the project author labelled a random 100.

## Results

Means over 60 ads, with 95% bootstrap intervals over ads:

| | CLIP ViT-B/32 | SigLIP2 B/16 | Difference | Ads better / same / worse |
|---|---|---|---|---|
| **precision@10** (decision metric) | 0.268 [0.193, 0.347] | 0.283 [0.210, 0.360] | **+0.015 [−0.013, +0.045]** | 18 / 28 / 14 |
| nDCG@10 | 0.393 | 0.430 | +0.037 [−0.031, +0.105] | 19 / 22 / 19 |
| picks that are mostly near-black | 42 of 600 | 32 of 600 | — | — |

**Decision (pre-registered rule): no measurable difference.**
- **What's ruled out:** the interval excludes both a SigLIP2 gain larger than 4.5 points and a SigLIP2 loss larger
  than 1.3 points.
- **Narrower than 4c:** ±0.03 here, against ±0.07 in 4c.

Source: `match_model_cpu-laptop_seed0.json` (seed 0). Chart: `match_model.png`.

**Per ad, precision@10 (ViT-B/32 → SigLIP2), where they differ:**

| SigLIP2 better | SigLIP2 worse |
|---|---|
| necklace 0.3 → 0.6, rail_travel 0.3 → 0.6, furniture 0.2 → 0.5 | menswear 0.9 → 0.7, home_insurance 0.6 → 0.4 |
| car 0.7 → 0.9, wine 0.1 → 0.3, bookstore 0.2 → 0.4, sci_fi_game 0.0 → 0.2 | fashion 0.4 → 0.2, music 0.5 → 0.3, bank 0.2 → 0.0 |
| airline, beer, outdoor_gear, pharmacy, kitchenware, party_supplies, piano_lessons, swimwear, watches, grocery, bird_feed (+0.1 each) | coffee, restaurant, pet_food, phone, city_tourism, bedding, news_media, office_supplies, department_store (−0.1 each) |

**20 of the 60 ads had no fitting scene in either model's picks:**
- beach_resort, toys
- bicycle, motorcycle, hotel
- fishing, camping, ski_resort, golf, horse_racing
- cleaning, baby_care, school_supplies, theatre_tickets
- cosmetics, hair_salon, bakery, dairy, cat_food, radio

Both models score 0 on all 20, so they count as "same".

**By ad set:**
- **The original 20:** ViT-B/32 0.460, SigLIP2 0.475 (the same +0.015 as overall).
- **The 40 new:** 0.173 and 0.188. They are much harder, because many name things these films rarely show.

## Label agreement (100 pairs)

- **agreement: 88%; Cohen's kappa: 0.68.**
- **Without the 3 cards discussed during labelling,** 97 pairs remain: 89%, κ 0.70. The 3 cards:
  - D0002: a man at a desk, office ad
  - D0005: adults in a classroom
  - D0694: a dark night scene with a cooking tripod but no visible flame, campfire ad
- **The guide closed the leniency gap:** both labellers marked 25% of the 100 pairs as fitting.
  - Phase 4c: 42% vs 28%.
  - Phase 4b: all 8 disagreements went the same way.
- **The 12 disagreements split evenly:**
  - 6 where only the author said "fits": office, bedding, baby care, party, wine, bridal.
  - 6 where only the primary said "fits": department store, shipping, bookstore, rail travel, city street, grocery.

**Sensitivity:** using the author's labels where present gives the same result to three decimals: +0.015
[−0.013, +0.043]. The disagreements cancel out.

## Findings

1. **SigLIP2 and CLIP ViT-B/32 match equally well on this task.**
   - **This time it's a confident "no difference"**, not just "we couldn't tell":
     - **The test:** fresh films, 60 ads and a tighter label rule.
     - **The interval:** about ±3 points, and it rules out any meaningful gain.
   - **It agrees with 4c** (+0.025 on reused films).
2. **Swapping the model changes *which* ads work, not how many.**
   - SigLIP2 is better on objects and settings: necklace, train stations, living rooms, wine.
   - ViT-B/32 is better on menswear, houses and fashion.
   - Overall, 18 ads are better, 14 worse and 28 the same.
3. **Precision depends mostly on whether the subject appears in the films at all.**
   - On a third of the ads, neither model found a single fit. A search always returns 10 scenes, even when the right
     answer is "nothing here" (Phase 4's finding 3, now at scale).
   - On the 40 ads that had at least one fit, precision is about 0.40–0.43.
4. **The label guide worked.** Agreement rose to 88% (κ 0.68–0.70), and the two labellers' fit rates are now
   identical. Writing down what fits for each ad removed the one-directional disagreement seen in 4b and 4c.
5. **SigLIP2 picks fewer near-black scenes again:** 32 against 42 of 600, the same direction as 4c.

## What this means for later phases

- **ViT-B/32 stays the pipeline's model.**
  - **Measured:** SigLIP2 is no better on a test that could have shown a gain of 4–5 points.
  - **Cheaper:** ViT-B/32 is about 3× faster on this laptop, and it already embeds the whole corpus.
- **The model is no longer the bottleneck.** Three experiments (4b scoring rules, 4c and 4d models) each moved
  precision@10 by 1–3 points, all within noise. What limits matching now:
  1. **The subject isn't there.** 20 of 60 ads have no fitting scene, so a system needs a "nothing here" signal (4b's
     cut-off tried this, with limited success).
  2. **What frames can't show.** Context, dialogue and what a scene is about.
- **Transcripts are the next lever.** Testing them (frames vs dialogue vs both) can reuse these 960 labels on the same
  films, so the next comparison costs far less labelling.

## Deviations from the protocol

- **Discussed cards:** 3 of the 100 cards were discussed during labelling. They are reported with and without.
- **One primary label corrected for consistency during labelling, before any scoring:**
  - **Which card:** D0111, from "fits" to "doesn't fit".
  - **Why:** it was an adults-in-a-classroom scene, which the guide excludes. The question about D0005 raised the same
    case.

## Limits

- **The primary labeller built the system** and wrote the guide. The second labeller agreed 88% (κ 0.68).
- **Each film was judged from 4 frames, without context.**
- **The 40 new ads include many subjects these films rarely contain.** That pulls the averages down for both models
  equally. It doesn't bias the comparison.
- **One seed, one run, 14 films.**
