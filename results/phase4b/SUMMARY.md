# Phase 4b — general improvements to CLIP matching: results

**Question:** do general changes, applied the same way to every ad, make CLIP's picks fit more often than
v1 (Phase 4)?

**Rules:** the Phase 4b amendment in `EVAL-PROTOCOL.md`. The code, the rules and the cut-off were frozen
in commit `eebe7a5` before the single test run.

## What v2 changes

The model is the same (CLIP ViT-B/32, the same embeddings). There's no re-ranking with a second model and
no per-ad tuning.

1. **5 fixed templates per brief**, averaged.
2. **Scored relative to ordinary footage:** similarity to the ad minus the mean similarity to 5 neutral
   descriptions.
3. **Junk filter:** nearly black or blank frames are never returned (pixel brightness and contrast).
4. **One cut-off for all ads,** τ = 0.146, set on the practice films only (best F1 over 300 labelled
   practice pairs). Below it, a scene isn't returned, so an ad can get fewer than 10 scenes, or none.

## Results (14 test films, 20 ads, one run)

**Labels:**
- v2 returned 174 scenes. 102 were already labelled in Phase 4; the other 72 were labelled blind by the
  primary labeller.
- The project author re-labelled 50 of the 72.

Means over 20 ads, with 95% bootstrap intervals over ads:

| | v1 | v2 | v2 − v1 | Ads better / same / worse |
|---|---|---|---|---|
| **precision@10** (pre-registered decision metric; scenes not returned count as misses) | 0.440 | 0.450 | **+0.010 [−0.055, +0.085]** | 4 / 10 / 6 |
| nDCG@10 | 0.644 | 0.712 | +0.068 [−0.018, +0.164] | 9 / 4 / 7 |
| precision of returned scenes (all ads pooled) | 0.44 (88 of 200) | **0.52** (90 of 174) | — | — |
| ads with nothing returned | 0 | 1 (beach), correctly | — | — |

**Decision (pre-registered rule): no measurable improvement.** The interval for precision@10 includes
zero.

v1's nDCG is lower here than in Phase 4 (0.644 vs 0.702) because its ideal ranking now also counts the
fitting scenes that v2 found.

Source: `match_v2_cpu-laptop_seed0.json` (seed 0). Chart: `match_v2.png`.

**Per ad, precision@10 (v1 → v2, scenes returned by v2):**

| Better | Same | Worse |
|---|---|---|
| cruise 0.3 → **0.7** (10) | car 0.9 (10) | fashion 0.5 → 0.3 (10) |
| music 0.3 → **0.7** (10) | airline 0.2 (10) | pharmacy 0.5 → 0.3 (10) |
| jewelry 0.4 → 0.6 (10) | beer 0.4 (10) | wine 0.4 → 0.2 (**2**) |
| restaurant 0.7 → 0.8 (10) | phone 1.0 (10) | furniture 0.3 → 0.2 (**2**) |
| | western_wear 0.9 (10) | coffee 0.5 → 0.4 (10) |
| | outdoor_gear 0.4 (10) | sci_fi_game 0.9 → 0.8 (10) |
| | toys 0.1, bank 0.1 (10) | |
| | beach_resort 0.0 (**0**, correct) | |
| | pet_food 0.0 (10) | |

## Label agreement (50 new pairs)

- **agreement: 84%; Cohen's kappa: 0.69.**
- **Without the 3 cards discussed before labelling,** 47 pairs remain: 83%, κ 0.67. (During labelling, the
  primary labeller gave the second labeller its answer for one jungle scene, shown twice, and for the
  "lion as a dog" question.)
- **All 8 disagreements go the same way:** the second labeller said "fits", the primary said "doesn't
  fit". 4 of the 8 are the fashion ad ("an elegant woman in a beautiful dress"), the most subjective
  brief.

**Sensitivity:** using the author's labels wherever they exist (Phase 4 and 4b) gives v1 0.460 and
v2 0.495. v2 − v1 is +0.035 [−0.025, +0.105]. **Same conclusion.**

## Findings

1. **The changes didn't reliably raise precision@10.** The average moved +0.01, with 4 ads better and 6
   worse.
2. **They did make the returned scenes cleaner: 52% fit, against 44%.** The cut-off removed weak picks.
   But because precision@10 counts each missing scene as a miss, abstaining costs as much as a wrong
   answer. That made wine and furniture look worse: they got only 2 scenes each.
3. **The cut-off said "nothing here" correctly once (beach) and missed the other obvious case (dog).** For
   pet food, CLIP is confidently wrong: lions and other animals score above the cut-off. **A threshold
   can't catch confident look-alikes.**
4. **The templates helped some ads a lot** (cruise +0.4, music +0.4) **and hurt others** (fashion and
   pharmacy −0.2). The net is about zero. Averaging fixed phrasings is not a reliable improvement for
   this model on this footage.
5. **Black and blank scenes no longer appear** in v2's results. That's a quality fix that precision@10
   doesn't measure.
6. **The cut-off chosen on practice films was too gentle to matter much.** It removed only 26 of 200
   picks. The practice films had more fitting scenes per ad than the test films, so the cut-off that was
   best there was low.

## What this means for later phases

- **v1 and v2 match about equally well.** v2 is the better input for building inventory (Phase 7),
  because it drops junk and some weak placements, and it can say "nothing here". It does that rarely and
  imperfectly.
- **The real limits are now the model** (look-alikes such as lion → dog and spaceship → airplane) **and
  the subjectivity of briefs** (fashion). Wording tricks don't fix either. Testing a different model is
  the next experiment, measured with this same protocol.

## Deviations from the protocol

- **The junk rule changed before any labelling.** The first version used CLIP text descriptions and
  flagged 44% of practice seconds. It was replaced by a pixel brightness and contrast rule, and the
  amendment was updated before the freeze commit.
- **Discussed cards:** 3 of the second labeller's 50 cards were discussed before they were labelled (see
  agreement). They are reported both ways.

## Limits

- **The test films were reused from Phase 4,** and v1's results on them were seen before v2 was designed.
  The changes are general rules, not per-ad fixes, but this is still weaker than a fresh test set.
- **Same caveats as Phase 4:**
  - the primary labeller built the system
  - judgements come from 4 frames, without context
  - 20 simulated ads
  - one seed
  - wide intervals (±0.07)
