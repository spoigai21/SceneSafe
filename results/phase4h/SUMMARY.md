# Phase 4h — confirmation on fresh films, and a re-rank by the card model: results

**Questions:**
1. Do scene cards still beat frames on films that no part of the method has been tuned or tested on?
2. Does letting the card model judge a shortlist (re-rank) improve on frames+cards?

**Rules:** the Phase 4h amendment in `EVAL-PROTOCOL.md`. They were frozen before the films were selected, by hash in
`rules_hash.txt`, and committed afterwards.

## Setup

- **Films:** 7 fresh public-domain films, 1 per genre, chosen by the corpus rules and skipping every earlier title
  (`corpus/manifest_4h.json`): Teenagers from Outer Space, Mr. Robinson Crusoe, Inner Sanctum, The Snows of
  Kilimanjaro, Gone with the West, The Phantom Planet, Minesweeper. 2,287 scenes, 1,259 with speech.
- **Ads:** the 60 Phase 4d ads, with their guide and theme lines.
- **Systems (top 10 per ad):**
  - **frames:** the Phase 4d ViT-B/32 scene scores
  - **frames+cards:** exactly as in Phase 4g
  - **rerank:** frames+cards' top 30, judged 0/1/2 by the card model (Qwen3-VL-4B) from the frames, dialogue and the
    ad's brief, guide and theme lines; re-ordered by the judgement, ties by the frames+cards score
- **Labels:** 1,086 unique pairs, labelled blind under the author's rules (Phase 4g addendum). The labeller never saw
  the cards or the judge's answers. Primary labelling split across 4 parallel instances of the same model.

## Results

Means over 60 ads, with 95% bootstrap intervals over ads:

| | Frames | Frames+cards | Rerank |
|---|---|---|---|
| **thematic precision@10** | 0.498 | 0.695 | 0.730 |
| literal precision@10 | 0.233 | 0.288 | 0.330 |
| graded nDCG@10 | 0.526 | 0.738 | 0.779 |

| Paired difference | Thematic precision@10 | Ads better / same / worse | Literal precision@10 | graded nDCG@10 |
|---|---|---|---|---|
| **frames+cards − frames** (decision 1) | **+0.197 [+0.142, +0.253]** | 45 / 8 / 7 | +0.055 [+0.022, +0.092] | +0.212 [+0.141, +0.285] |
| **rerank − frames+cards** (decision 2) | **+0.035 [+0.008, +0.062]** | 23 / 29 / 8 | +0.042 [+0.022, +0.063] | +0.041 [+0.009, +0.071] |
| rerank − frames | +0.232 [+0.180, +0.287] | 47 / 9 / 4 | +0.097 [+0.055, +0.142] | +0.253 [+0.187, +0.321] |

**Decisions (pre-registered rules):**
1. **Cards confirmed.** The interval is entirely above 0, and the gain matches Phase 4g (+0.210) on films nothing
   was tuned on.
2. **The re-rank helps, but the gain is small and fragile.** The interval is entirely above 0 with the pre-registered
   labels. With the author's labels swapped in, it shrinks to +0.022 and its interval includes 0 (see "Label
   agreement").

Source: `match_confirm_cpu-laptop_seed0.json` (seed 0). Chart: `match_confirm.png`.

**Largest changes per ad** (thematic precision, frames → rerank): boating 0.1 → 1.0, bookstore 0.2 → 0.9,
beach_resort 0.2 → 0.9, music 0.3 → 0.9, golf 0.0 → 0.5, news_media 0.3 → 0.8, jewelry 0.5 → 1.0. Worse:
bicycle 0.1 → 0.0, outdoor_gear, phone and home_insurance 1.0 → 0.9.

## The judge

- **Answers on 1,800 shortlist checks:** 806 × 0, 312 × 1, 682 × 2; none unreadable. About 0.3 s per check.
- **Against the labels** (893 pairs judged and labelled): exact agreement 53%; fits vs doesn't fit 74%. Good enough
  to re-order a shortlist, not to replace a labeller.
- **"Nothing here":** the judge gave 0 to all 30 candidates only for bicycle, which also has no suitable pick in any
  system.

## Label agreement

**Second labeller:** the project author labelled all 100 pairs of the planned sample (rng 7), under the same rules.

| | Value |
|---|---|
| exact agreement (0/1/2) | 64% |
| thematic-or-better: agreement, Cohen's κ | 76%, κ 0.50 |
| literal: agreement, Cohen's κ | 86%, κ 0.59 |

- **Where they differ:** the author was looser on themes (14 pairs the primary labelled 0, the author 1) and stricter
  on literal fits (7 pairs labelled 2 by the primary, 1 by the author). The author gives a literal fit only when the
  ad's picture is the main thing on screen (for example, a busy street with crowds, not a quiet street with a few
  people).
- **Agreement is lower than after the 4g relabel** (κ 0.72 there, 0.50 here), so the written rules still leave room
  for judgement.

**Sensitivity (author's labels where present, primary labels elsewhere), thematic precision@10:**

| Paired difference | Pre-registered labels | With the author's labels |
|---|---|---|
| frames+cards − frames | +0.197 [+0.142, +0.253] | +0.202 [+0.147, +0.258]; 44 / 11 / 5 |
| rerank − frames+cards | +0.035 [+0.008, +0.062] | **+0.022 [−0.003, +0.047]**; 20 / 31 / 9 |
| rerank − frames | +0.232 [+0.180, +0.287] | +0.223 [+0.172, +0.277]; 47 / 8 / 5 |

The cards' gain holds. The re-rank's gain is no longer clearly above 0.

## Follow-up: sharper rules, a relabel and a repeat check (written after the result)

Rules: the Phase 4h addendum in `EVAL-PROTOCOL.md`, frozen by hash before the relabel (`rules_hash.txt`).

- **Sharper rules.** The 36 disagreements and the author's questions while labelling were turned into five more
  written rules (6–10): 2 only when the ad's picture is the main subject; size words such as "crowds" count; the
  literal case is part of the theme; the guide's exclusions are not 2; a place or activity named in the theme line
  fits the theme by itself.
- **Relabel.** All 1,086 pairs relabelled from scratch by 4 fresh instances given only the written rules and the
  sheets: blind to the system, the cards, the judge and the author's labels.

| | Pre-registered labels | Relabel |
|---|---|---|
| agreement with the author (100 pairs), exact | 64% | 75% |
| thematic-or-better: agreement, κ | 76%, κ 0.50 | 83%, κ 0.65 |
| literal: agreement, κ | 86%, κ 0.59 | 91%, κ 0.71 |

The new rules were written from the author's 100 labels, so the relabel's agreement on those 100 is not an
independent measure.

**Results under the relabel** (thematic precision@10; ads better / same / worse):

| Paired difference | Relabel | Relabel, author's labels where present |
|---|---|---|
| frames+cards − frames | **+0.188 [+0.138, +0.240]**; 45 / 7 / 8 | +0.192 [+0.143, +0.243]; 44 / 11 / 5 |
| rerank − frames+cards | **+0.035 [+0.007, +0.065]**; 22 / 27 / 11 | +0.027 [−0.000, +0.055]; 22 / 27 / 11 |
| rerank − frames | +0.223 [+0.173, +0.275]; 46 / 10 / 4 | +0.218 [+0.170, +0.270]; 48 / 7 / 5 |

Means under the relabel: frames 0.473, frames+cards 0.662, rerank 0.697. Literal precision: 0.165 / 0.215 / 0.248.
Source: `match_confirm_relabel_cpu-laptop_seed0.json`.

- **The cards' gain is the same under all four sets of labels** (+0.188 to +0.202).
- **The re-rank's gain stays small (+0.027 to +0.035) and sits right at the edge.** Its lower bound is above 0 with
  the primary labels and with the relabel, and at 0 whenever the author's own labels are swapped in.

**Repeat check.** The card writer and the judge were re-run from scratch on a random sample: 30 of 30 cards and 60
of 60 judgements were identical to the stored ones (`repeat.json`). Decoding is greedy, so re-running the pipeline
gives the same picks; "one run" is not a source of noise here. The remaining randomness is in which films and ads
were chosen, which the bootstrap over ads covers only in part.

**Decision (made before the relabel, Phase 4h addendum): frames+cards is the project's matcher.** The re-rank is an
optional extra with a small, uncertain gain. It is not tuned further on these films.

## Findings

1. **Scene cards generalise.** On 7 films no part of the method had seen, frames+cards beats frames by +0.197, the
   same size as on the development films (+0.210), and between +0.188 and +0.202 under every set of labels.
2. **A second look by the vision model probably helps a little.** Re-ranking the top 30 adds +0.035 thematic and
   +0.042 literal precision with the pre-registered labels, and +0.035 under the relabel. With the author's labels
   swapped in, the thematic gain is +0.022 to +0.027 and could be 0.
3. **The matcher (frames+cards) finds about 7 suitable scenes in 10, against 5 for frames alone** (0.695 vs 0.498;
   0.730 with the re-rank).

## What this means

- **The matcher is frames + scene cards.** The card model's re-check of the top 30 is an optional extra: cheap, but
  its gain is small and not robust to the author's labels.
- **Cost on this laptop:** about 1.5 s per scene to write a card (once per scene, cached), and about 0.3 s per
  shortlist check (30 per ad).

## Limits

- **The headline uses the primary labels.** The author's 100-pair sample is a check, not the headline. It agreed
  moderately with the primary labels (κ 0.50) and better with the relabel (κ 0.65, not independent: the relabel's
  rules were written from the author's labels).
- **7 films**, 1 per genre. The pipeline is deterministic (repeat check), but one draw of films and ads.
- **The judge sees the same guide and theme lines as the labeller,** which makes its task close to the labelling
  task. An advertiser would supply such a description.
- **Less usable dialogue in two films:** in Gone with the West (1975, much singing) and Mr. Robinson Crusoe (1932,
  mostly music), the speech rules dropped about half the transcript segments as junk (repeat loops, low-confidence
  lines, "The End" over music).
- **Old black-and-white films, 4 frames per scene.**
- **One human labeller.** The author labelled 100 of 1,086 pairs; every other label is by the same AI model that
  built the system, blind to which system picked each scene.

## Known issues (everything that went wrong or is still open, Phases 4–4h)

**Fixed along the way**
- **Whisper junk** (4f): repetition loops and long near-empty segments ("The End" over music). Its no-speech score
  was always 0, so it could not be used. Fixed by two general rules, disclosed in the protocol; in two 4h films
  about half the segments were dropped.
- **Card model repeating itself** (4g development): lists of the same item over and over. Fixed with a repetition
  penalty and a "no repeats" line in the prompt, before the test.
- **Laptop out of memory during the card run** (4g): about 15 s per scene from swapping. Fixed by clearing the GPU
  cache after each card and loading the model once. Memory-only changes, logged by hash; the repeat check confirms
  outputs are deterministic.
- **Display bug** (Phase 4): the last frame of scenes at the end of a film could be blank. Fixed for later runs.
- **A chart saved under the wrong name** (4g): a variable clash, fixed and the stray file deleted.

**Process deviations, disclosed**
- **Rules frozen by hash, committed later** (4g, 4h, the 4h addendum): the author asked to run before committing.
  The SHA-256 at each freeze is in `rules_hash.txt`.
- **Labelling rules changed twice after results** (the 4g and 4h addenda). Each time the pre-registered labels stayed
  the headline and the new labels were reported beside them.
- **Labels changed during review:** a few author and primary labels were corrected during labelling or discussion;
  each is recorded (4b–4g summaries, `corrections` in `labels_author.json`).
- **Films reused** (4b–4g used the same 14 films up to five times); 4h is the fresh-film check.
- **Some phases have no human check** (4e, 4f).

**Still open (needs new films, time or money)**
- **Old black-and-white public-domain films at 480p.** Modern colour footage may behave differently.
- **4 frames per scene.** More frames, or short clips, would show more; not tested.
- **Labels by the same AI model that built the system,** checked by one human on 100 pairs per phase at most. An
  independent labelling panel would be stronger.
- **Simulated ads** written by the project, with guide and theme lines the judge also sees.
- **The re-rank's gain is unresolved.** A fresh set of films would be needed to settle it without tuning on the test.
- **Whisper's accuracy is not measured** against human subtitles; mishearings ("repril units") reach the cards.

