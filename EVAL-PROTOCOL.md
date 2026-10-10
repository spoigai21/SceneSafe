# Phase 4 evaluation protocol

Written and committed **before** any Phase 4 result was produced. Any later change is listed under
"Deviations" in `results/phase4/SUMMARY.md`, with the reason.

## Question

Does matching ads to scenes with CLIP find better ad placements than (a) random scenes and (b) using
only the film's genre?

## Data

- **Test films:** the 14 films in `configs/phase4_split.yaml` (2 per genre, seeded draw). The other 21
  films are practice films. They are used only to check that the pipeline runs, never to tune the
  ads or settings.
- **Scenes:** the Phase 3 detected scenes (threshold 0.75, minimum 3 s), restricted to the test films.
  The scene is the unit that gets judged.
- **Embeddings:** the Phase 2 corpus embeddings (CLIP ViT-B/32, OpenCLIP `laion2b_s34b_b79k`, 1 fps).
- **Ads:** the 20 simulated ads in `configs/ads.yaml`, frozen at this commit. The briefs were written
  once and not tuned.

## Methods

Each method returns **10 different scenes from the test films for each ad**.

1. **CLIP:** embed the ad's brief with CLIP's text half. Run an exact search over every second of the
   test films, group the hits by scene, and keep each scene once at its best second (the Phase 3
   decision). Return the top 10 scenes in score order.
2. **Genre:** pick random moments (uniform over seconds, so longer scenes are proportionally likelier,
   like a random point in the film) from test films whose genre is in the ad's `genres` list. Take the
   scene each moment falls in, skip repeats, and stop at 10. The order is the draw order.
3. **Random:** the same as Genre, but over all test films.

The random draws use `numpy.random.default_rng(1000 + ad index)`, so they are repeatable.

## Labels

- **Pairs:** every (ad, scene) pair returned by any method is pooled. A scene returned by two methods
  for the same ad is labelled once.
- **What the judge sees:** the ad's category and brief, plus **4 frames spaced evenly across the
  scene**. Every method's scenes are shown the same way. Pairs are shuffled
  (`default_rng(0)`), and nothing on screen says which method found the scene.
- **The question:** *Is this scene a natural place for this ad?*
  - **Fits:** the ad's subject or its natural setting is clearly visible in the scene. For the car ad,
    a car is clearly visible on a road or being driven. For the beer ad, people are drinking at a bar
    or in a saloon.
  - **Doesn't fit:** anything else, including "only loosely related" and "can't tell". If unsure,
    choose Doesn't fit.
  - Judge relevance only. Ignore safety (Phase 5's job), film quality, and colour versus black and
    white.
- **Labellers:**
  - **Primary:** an AI assistant (Claude), which also wrote the pipeline. It looks at the same frames,
    blind to the method. Its labels are used for every metric.
  - **Second:** the project author labels a random 100 of the pooled pairs (`default_rng(0)`) without
    seeing the primary labels.
- **Agreement:** reported as percent agreement and Cohen's kappa on those 100 pairs, whatever the
  result.

## Metrics

All metrics use the primary labels, the top 10 per ad, and an average over the 20 ads.

- **precision@10:** the share of the 10 scenes that fit.
- **nDCG@10:** binary gain with a log2 rank discount. The ideal ranking uses all the scenes judged to
  fit for that ad across all methods (at most 10). An ad with no fitting scene in the pool scores 0.
- **Uncertainty:** 95% percentile bootstrap over ads (10,000 resamples, `default_rng(0)`), for each
  method and for the paired differences CLIP − Genre and CLIP − Random.

## Decision rule

- **CLIP beats genre** if the 95% interval of the paired precision@10 difference (CLIP − Genre) is
  entirely above 0. Otherwise the result is reported as **"CLIP does not beat genre"**, as plainly as
  a win would be.
- The same rule applies to CLIP − Random.
- A per-ad table is reported too, including every ad where Genre or Random wins.

## Disclosures

- **Prior exposure:** example frames from some test films (e.g. `my_favorite_brunette`,
  `Quicksand_clear`) were seen in Phase 3's example sheets. Some briefs resemble Phase 3 test queries
  ("a car driving on a road", "a person talking on the telephone"). No ad, brief or setting was chosen
  by looking at test-film results.
- The ads, genres lists and advertisers are simulated. The films are public domain.
