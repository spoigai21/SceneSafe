# Phase 4c — a different model, same method: results

**Question:** with the matching method held fixed, does a stronger image–text model pick better scenes than CLIP
ViT-B/32?

**Rules:** the Phase 4c amendment in `EVAL-PROTOCOL.md`, committed in `ec47455` before the single test run.

## Setup

**Models:**
- **v1 (Phase 4):** CLIP ViT-B/32 (`laion2b_s34b_b79k`). Its picks and labels are reused.
- **4c:** SigLIP2 ViT-B/16 (OpenCLIP `webli`), fp16 on the laptop GPU. It embedded the 14 test films at the same
  71,091 seconds.

**What's held fixed:**
- the method: each brief as written, exact search over every second, each scene once at its best second, top 10
- the scenes: the Phase 3 scenes, so both models choose from the same 3,911
- no cut-off and no junk filter

**How the model was chosen:** on practice films only. Three candidates re-ranked the 300 labelled Phase 4b practice
pairs. All three ranked fitting pairs above non-fitting ones better than ViT-B/32 (AUC about 0.65 vs 0.57,
intervals overlapping zero). SigLIP2 B/16 was the fastest of the three. This check was exploratory and isn't a
reported result.

**Labels:**
- SigLIP2's 200 picks contained 104 pairs not labelled in Phase 4 or 4b. The primary labeller labelled them blind.
- The project author re-labelled 50 of the 104.

## Results (14 test films, 20 ads, one run)

Means over 20 ads, with 95% bootstrap intervals over ads:

| | ViT-B/32 (v1) | SigLIP2 B/16 | Difference | Ads better / same / worse |
|---|---|---|---|---|
| **precision@10** (decision metric) | 0.440 | 0.465 | **+0.025 [−0.045, +0.095]** | 8 / 6 / 6 |
| nDCG@10 | 0.572 | 0.635 | +0.063 [−0.054, +0.189] | 12 / 2 / 6 |
| picks that are mostly near-black | 18 of 200 | **5 of 200** | — | — |

**Decision (pre-registered rule): no measurable improvement.** The interval for precision@10 includes zero.

v1's nDCG is lower than in Phase 4 (0.572 vs 0.702) because its ideal ranking now also counts the fitting scenes
that 4b and 4c found. The near-black count is descriptive: a pick counts if more than half its seconds fail the
Phase 4b pixel rule.

Source: `match_model_cpu-laptop_seed0.json` (seed 0). Chart: `match_model.png`.

**Per ad, precision@10 (ViT-B/32 → SigLIP2):**

| Better | Same | Worse |
|---|---|---|
| jewelry 0.4 → **0.7** | western_wear 0.9 | fashion 0.5 → 0.3 |
| cruise 0.3 → **0.6** | sci_fi_game 0.9 | pharmacy 0.5 → 0.3 |
| furniture 0.3 → **0.6** | airline 0.2 | coffee 0.5 → 0.3 |
| beer 0.4 → 0.6 | toys 0.1 | music 0.3 → 0.1 |
| car 0.9 → 1.0 | beach_resort 0.0 | outdoor_gear 0.4 → 0.3 |
| wine 0.4 → 0.5 | pet_food 0.0 | phone 1.0 → 0.9 |
| restaurant 0.7 → 0.8 | | |
| bank 0.1 → 0.2 | | |

## Label agreement (50 new pairs)

- **agreement: 82%; Cohen's kappa: 0.61**, lower than Phase 4 (0.68) and 4b (0.69).
- **Without the one card discussed during labelling** (S087, a man holding up a banknote, for the "money being
  counted" brief), 49 pairs remain: 82%, κ 0.60.
- **The second labeller was more lenient:** fit rate 42%, against 28% for the primary.
  - In 8 of the 9 disagreements the second labeller said "fits", spread across furniture (2), jewelry (2), cruise,
    beer, music and fashion.
  - In the ninth, the primary said "fits" (outdoor gear).

**Sensitivity:** using the author's labels wherever they exist (Phases 4, 4b and 4c) gives ViT-B/32 0.460 and
SigLIP2 0.510. The difference is +0.050 [−0.040, +0.145]. **Same conclusion.**

## Findings

1. **A stronger model alone didn't measurably improve precision@10.** The average rose +0.025, with 8 ads better
   and 6 worse. The practice-film check (AUC +0.08) pointed the same way but was also inconclusive, and the test run
   didn't confirm it.
2. **It changed *which* ads work:**
   - Better on ads that depend on a specific object or setting: jewelry (couples), cruise (ships at sea), furniture
     (rooms), beer (bars).
   - Worse on fashion, pharmacy, coffee and music.
   - With 20 ads and 10 scenes each, a one-scene swing is ±0.1, so most of these per-ad changes are within noise.
3. **The look-alike errors remain.**
   - The dog ad still got lions (0.0).
   - The airline ad still got rockets and spaceships (0.2).
   - Neither model can tell these apart from frames alone.
4. **SigLIP2 picks far fewer near-black scenes: 5 against 18.** It's less drawn to dark, empty frames, which is
   part of what the Phase 4b junk filter was added to fix.
5. **The second labeller and the primary labeller disagree more on 4c's picks (κ 0.61).** The new model finds more
   borderline scenes (a room that may or may not be a living room, a glass that may or may not be wine), where
   "fits" is a judgement call.

## What this means for later phases

- **ViT-B/32 stays the pipeline's model.** SigLIP2 isn't measurably better, and ViT-B/32 is faster and already
  embeds the whole corpus. A switch would need a measured gain.
- **Three changes (4b's scoring rules, 4c's model) have each moved precision@10 by +0.01 to +0.03, all within
  noise.** The ceiling here doesn't seem to come from the image model or its wording. It comes from what a few
  frames can show:
  - look-alikes (a lion is not a dog)
  - objects that are visible but not the point of the scene
  - briefs that need context to judge
- **Dialogue is the missing signal (raised by the project author during labelling).** A person watching knows what a
  scene is about from what's said, not only from what's shown. Transcripts (subtitles, or speech-to-text for films
  without them) could:
  - resolve cases frames can't, such as what the money is for, or whether a scene is a dinner or an argument
  - catch unsafe content that is spoken rather than shown, which matters for Phase 5
  - possibly carry most of the matching on their own, with frames still needed for what is *happening* on screen
    (driving, dancing, riding)
- **That is a hypothesis, not a result.** It would be tested like everything else: frames only vs transcript only vs
  both, under a pre-registered protocol.

## Deviations from the protocol

- **One discussed card (S087).** The second labeller asked what the object was. It was described as a banknote held
  up, with no fit verdict given. Agreement is reported with and without it.
- **The near-black count** was added to the scoring script after the rules were frozen. It's descriptive, and the
  decision doesn't depend on it.

## Limits

- **The test films were used a third time,** and v1 and v2 results on them were seen before this amendment. The
  model was chosen on practice films only and the method is unchanged, but this is still weaker than a fresh test
  set.
- **The primary labeller built the system,** and agreement with the second labeller is the lowest of the three
  rounds (κ 0.61).
- **Same caveats as Phase 4:** judged from 4 frames without context, 20 simulated ads, one seed, and wide intervals
  (±0.07).
- **Only one alternative model was tested.** ViT-L/14 and SigLIP2 SO400M weren't run on the test films, by design.
