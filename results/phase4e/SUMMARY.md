# Phase 4e — searching by theme instead of literal description: results

**Question:** does searching with an ad's theme ("beauty and glamour: a beautiful woman, getting ready to go out…")
find scenes that suit the ad more often than searching with its literal brief ("a woman putting on makeup at a
mirror")?

**Rules:** the Phase 4e amendment in `EVAL-PROTOCOL.md`, committed in `6da2abb` before any Phase 4e pick was made.

## Setup

- **Films, scenes and ads:** the 14 Phase 4d films (3,116 scenes) and the 60 ads.
- **Theme lines** (`configs/themes.yaml`): one theme and three short search phrases per ad, written before any 4e
  pick was seen.
- **Systems:** the literal system (Phase 4d ViT-B/32 picks) and the theme system (the three phrases, ViT-B/32), top
  10 per ad.
- **Labels:** three levels: 2 = literal fit, 1 = thematic fit, 0 = neither. 1,013 unique pairs, labelled blind by
  the primary labeller.

## Results

Means over 60 ads, with 95% bootstrap intervals over ads:

| | Literal system | Theme system | Difference (theme − literal) | Ads better / same / worse |
|---|---|---|---|---|
| **thematic precision@10** (decision metric) | 0.525 | 0.543 | **+0.018 [−0.040, +0.075]** | 24 / 21 / 15 |
| literal precision@10 | 0.287 | 0.218 | −0.068 [−0.113, −0.028] | 8 / 29 / 23 |
| graded nDCG@10 | 0.547 | 0.537 | −0.011 [−0.072, +0.049] | 32 / 4 / 24 |

**Decision (pre-registered rule): no measurable difference** on thematic precision. The theme system finds fewer
literal fits.

Source: `match_theme_cpu-laptop_seed0.json` (seed 0). Chart: `match_theme.png`.

## Findings

1. **Searching by theme does not help.** It finds as many thematic fits as the literal brief, and fewer literal ones.
2. **Judging by theme changes the picture far more than searching by theme.** The same literal picks that scored
   0.268 under the strict Phase 4d rule score 0.525 when a thematic fit counts. About half the "misses" in 4d suit the
   ad in a looser sense.
3. **Few ads have nothing at all.** Only 4 ads (toys, bicycle, baby care, radio) had no suitable scene in either list.

## What this means for later phases

- **Keep the literal search; judge thematically.** Later phases report thematic precision as the main metric.
- **The search is not the limit; what the system knows about a scene is.** This led to Phase 4f (dialogue) and 4g
  (written scene descriptions).

## Deviations and limits

- **Primary labels only.** The author did not label a 4e sample, so there is no agreement figure for this phase.
- **13 pairs labelled "doesn't fit" in 4d were labelled literal fits (2) here,** under the three-level rule. Reported
  in the result file (`4d_non_fits_now_labelled_literal`).
- **One label changed during labelling for consistency:** E0066, 0 → 1, before scoring.
- **The primary labeller wrote the theme lines** and labelled the pool.
- **Third use of these films.** One seed, one run.
