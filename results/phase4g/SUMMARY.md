# Phase 4g — scene cards (a written description of each scene): results

**Question:** does matching ads against a short written description of each scene, made by a vision-language model
that looks at the frames and reads the dialogue, find scenes that suit the ad more often than frames alone?

**Rules:** the Phase 4g amendment in `EVAL-PROTOCOL.md`. They were frozen before any test card was written, but
committed after the run because the author asked to run straight away. The SHA-256 of the protocol and code at the
freeze is in `rules_hash.txt`, with two later memory-only fixes logged there (see "Deviations").

## Setup

- **Films:** the 12 Phase 4d films the card method was never tried on (2,635 scenes, 1,377 with speech). The 2 films
  used to develop the cards (`calvary`, `Hitchcock_Secret_Agent`) are left out.
- **Ads:** the 60 Phase 4d ads, with their guide and theme lines.
- **Scene cards:** Qwen3-VL-4B (4-bit, mlx-vlm, laptop GPU) looks at the scene's 4-frame strip and reads its
  dialogue, and writes five lines: place, visible things, activity, what people talk about, and mood. For example:
  > place: nightclub or lounge · visible: men in tuxedos, cocktail glasses… · talk: money lending, dry martini order
- **Systems (top 10 per ad):**
  - **frames:** the Phase 4d ViT-B/32 scene scores
  - **cards:** the card text (place, visible, activity, talk) vs the ad's brief and theme (all-mpnet-base-v2)
  - **frames+cards:** per-ad z-scores of the two, added with equal weight
- **Labels:** 1,249 unique pairs, labelled blind with frames and dialogue shown. The labeller never saw the cards.
  Three levels as in 4e. The primary labelling was split across 4 parallel instances of the same model.

## Results

Means over 60 ads, with 95% bootstrap intervals over ads (pre-registered labels):

| | Frames | Cards | Frames+cards | Frames+cards − frames | Ads better / same / worse |
|---|---|---|---|---|---|
| **thematic precision@10** (decision metric) | 0.482 | 0.670 | **0.692** | **+0.210 [+0.155, +0.268]** | 42 / 14 / 4 |
| literal precision@10 | 0.275 | 0.252 | 0.340 | +0.065 [+0.035, +0.097] | 23 / 33 / 4 |
| graded nDCG@10 | 0.502 | 0.659 | 0.736 | +0.234 [+0.169, +0.297] | 44 / 4 / 12 |

**Decision (pre-registered rule): scene cards improve matching.** The whole interval is above 0.

- **Cards alone vs frames:** +0.188 [+0.123, +0.255] thematic.
- **Frames+cards vs cards alone:** +0.022 [−0.012, +0.058] thematic, but +0.088 [+0.053, +0.125] literal. Keeping
  the frame score is what protects the literal fits.

Source: `match_cards_cpu-laptop_seed0.json` (seed 0). Chart: `match_cards.png`.

**Largest changes per ad** (thematic precision, frames → frames+cards):

| Better | Worse |
|---|---|
| cosmetics 0.1 → 1.0, bakery 0.1 → 0.9, beach_resort 0.1 → 0.8, kitchenware 0.4 → 0.9, furniture 0.5 → 1.0, bird_feed 0.0 → 0.5 | phone 0.9 → 0.7, coffee 0.9 → 0.8, golf 0.2 → 0.1, baby_care 0.1 → 0.0 |

Toys, bicycle and baby care still have no suitable pick.

## Label agreement and the relabel

**Second labeller:** the project author labelled 51 of the planned 100 pairs.

| | Pre-registered labels | Relabel under the author's rules |
|---|---|---|
| exact agreement (0/1/2) | 63% | 82% |
| thematic-or-better, Cohen's κ | 0.57 | 0.72 |
| literal, Cohen's κ | 0.30 | 0.78 |

- **The disagreements were reviewed with the author.** The primary labeller was stricter on themes (for example, it
  gave 0 when a captain talks about the voyage, which the author counts as a sea theme) and looser on literal fits.
  The author's rules were written into the protocol ("Phase 4g addendum") and the whole pool was relabelled from
  scratch, blind to the system.
- **Relabel result:** frames 0.553, cards 0.753, frames+cards 0.750; **frames+cards − frames +0.197 [+0.142,
  +0.255]**, 39 ads better, 18 same, 3 worse. Literal +0.063 [+0.035, +0.093]. Source:
  `match_cards_relabel_cpu-laptop_seed0.json`.
- **Sensitivity (author's labels where present, pre-registered labels elsewhere):** +0.213 [+0.160, +0.268].

The gain is the same under both labelling standards. The more generous rules raise every system by about 0.06–0.08.

## Findings

1. **Describing the scene in words is the first change since Phase 4 that clearly improves matching.** About 7 in 10
   picks now suit the ad, against about 5 in 10 for frames. Earlier changes (scoring rules, SigLIP2, theme search,
   raw dialogue) all moved precision by under 0.03.
2. **The card says what a scene is, not just how it looks.** "Dressing room, woman, mirror" matches a cosmetics ad;
   a frame-only search for "a woman putting on makeup" found staircases and cabins.
3. **Dialogue helps as context, not as a score.** In 4f, dialogue added to the frame score gave nothing. Here it is
   one input to the describer. In the practice round, removing the talk line lowered card precision (0.443 →
   at least 0.370, on 2 films).
4. **Keeping the frame score protects literal fits.** Cards alone lose a little literal precision. Frames+cards
   gains it.

## What this means for later phases

- **Frames+cards becomes the pipeline's matcher.**
- **Cost:** a card takes about 1.5 s per scene on this laptop, about an hour for 12 films, against seconds for frame
  search. Cards are written once per scene and cached.
- **The same cards and transcripts can feed Phase 5 (brand safety):** what is said and done in a scene, in words.

## Deviations from the protocol

- **Rules committed after the run.** The author asked to run before committing; the freeze is recorded by hash in
  `rules_hash.txt`.
- **Two memory-only code changes during the card run,** both logged with new hashes in `rules_hash.txt`:
  - the model's GPU cache is cleared after each card
  - the model is loaded once per run instead of once per film

  The laptop was swapping (about 15 s per scene). Decoding is greedy, so the cards are unchanged.
- **Author sample:** 51 of 100 labelled.
- **Author corrections during review:** 3 of the author's labels changed while the author reviewed cards for the
  discussion (2 from 1 to 0, 1 from 1 to 2), recorded in `labels_author.json`.
- **The relabel is a sensitivity analysis written after the result.** The pre-registered labels stay the headline.

## Limits

- **Fifth use of these 12 films** (Phases 4d, 4e, 4f and the fusion practice round). The card method was never run on
  them before the test. A confirmation on fresh films is needed before treating the result as general.
- **Old black-and-white films and 4 frames per scene** make many cards hard to judge, for both labellers.
- **The card model can be wrong.** Cards were not checked one by one; a few sampled cards contained small errors (for
  example, an object that isn't there).
- **The primary labeller built the system** and wrote the theme lines. The labeller never saw the cards.
- **One seed, one run.**
