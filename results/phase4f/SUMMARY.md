# Phase 4f — adding what is said (Whisper transcripts) to what is shown: results

**Question:** does adding a scene's dialogue to its frames find scenes that suit the ad more often than frames alone?

**Rules:** the Phase 4f amendment in `EVAL-PROTOCOL.md`, committed in `aaeea3a` before any Phase 4f pick was made.

## Setup

- **Films, scenes and ads:** the 14 Phase 4d films (3,116 scenes, 1,639 with speech) and the 60 ads.
- **Transcripts:** Whisper large-v3-turbo (mlx-whisper, laptop GPU) on every film. Before the rules were committed
  the transcripts were read for problems; two kinds of junk were found and filtered by general rules (long, nearly
  empty segments such as "The End" over music; repetition loops).
- **Systems (top 10 per ad):**
  - **frames:** the Phase 4d ViT-B/32 picks
  - **dialogue:** the ad's brief and theme vs the scene's dialogue (all-mpnet-base-v2, best 50-word window)
  - **both:** per-ad z-scores of the two, added with equal weight
- **Labels:** 1,425 unique pairs, labelled blind with frames and dialogue shown, three levels as in 4e. The primary
  labelling was split across 4 parallel instances of the same model with the same rules.

## Results

Means over 60 ads, with 95% bootstrap intervals over ads:

| | Frames | Dialogue | Both | Both − frames | Ads better / same / worse |
|---|---|---|---|---|---|
| **thematic precision@10** (decision metric) | 0.443 | 0.337 | 0.427 | **−0.017 [−0.087, +0.052]** | 23 / 9 / 28 |
| literal precision@10 | 0.285 | 0.057 | 0.155 | −0.130 [−0.180, −0.083] | 4 / 23 / 33 |
| graded nDCG@10 | 0.517 | 0.323 | 0.423 | −0.094 [−0.184, −0.005] | 19 / 3 / 38 |

**Decision (pre-registered rule): no measurable difference** on thematic precision. Adding dialogue this way makes
literal precision and ranking worse.

Source: `match_dialogue_cpu-laptop_seed0.json` (seed 0). Chart: `match_dialogue.png`.

**Dialogue helps where the picture is weak and hurts where it is strong** (thematic precision, frames → both):

| Helped | Hurt |
|---|---|
| sci_fi_game 0.1 → 0.7, necklace 0.3 → 0.8, fashion 0.4 → 0.9, bakery 0.1 → 0.6 | home_insurance 0.9 → 0.2, city_tourism 0.7 → 0.1, car 0.8 → 0.4, pet_food 1.0 → 0.5 |

## Findings

1. **Dialogue alone is a weak matcher** (0.337 vs 0.443 for frames). People talk about money, weddings and the
   future; they rarely name the car or the dog on screen.
2. **An equal-weight sum cancels out.** Gains on abstract ads and losses on visual ads roughly balance.
3. **The sum favours talking scenes.** 593 of 600 "both" picks have speech, against 357 of 600 frame picks. A silent
   scene gets 0 for dialogue, so silent chases, landscapes and animals are pushed down.

## Practice round after the result (exploratory, not reported as results)

Other ways of combining were tried on the same films, with 278 extra practice labels:

- strongest of the two
- average of the available signals
- frames only when they are confident
- 7 frame picks + 3 dialogue picks
- frames' top 20 re-ranked with dialogue
- the ContextIQ rule (weight + threshold + max)

All were within ±0.02 of frames. The way of combining was not the problem; the dialogue signal itself was weak. This
led to Phase 4g, where dialogue is context for a model that describes the scene.

## Limits

- **Fourth use of these films.**
- **Whisper's accuracy is not measured** against human subtitles.
- **Primary labels only.** The author did not label a 4f sample; the result was clear enough to move on.
- **Not comparable with 4d or 4e:** labels now include dialogue.
- **One seed, one run.**
