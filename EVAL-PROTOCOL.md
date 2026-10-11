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

---

# Phase 4b amendment: general improvements to CLIP matching

Written and committed **after** the Phase 4 results and **before** any Phase 4b result on the test films.

## Question

Do general changes, applied the same way to every ad, make CLIP's picks fit more often than v1 (the
Phase 4 method)?

## What changes

The model doesn't change (ViT-B/32, the same embeddings). There's no re-ranking with a second model, and
no ad gets hand-written wording, exclusions or settings.

1. **Template ensemble.** Each brief is embedded in these 5 fixed templates. The 5 unit vectors are
   averaged and re-normalised:
   - `{brief}`
   - `a photo of {brief}.`
   - `a still from a film showing {brief}.`
   - `a black and white film still of {brief}.`
   - `a movie scene of {brief}.`
2. **Score relative to ordinary footage.** A second's score for an ad is its similarity to the ad minus
   its mean similarity to 5 fixed neutral descriptions:
   - "people talking in a room"
   - "a close-up of a person's face"
   - "a landscape"
   - "a group of people standing"
   - "an indoor scene"
3. **Junk filter.** A second is junk if its frame is nearly black (mean grey level < 0.05 on a 0–1 scale)
   or nearly blank (standard deviation of grey level < 0.03). Grey level is measured on every 8th pixel
   of the same 1 fps frames the embeddings came from.

   Junk seconds are never returned, and a scene is ranked by its best non-junk second.

   - **How it was chosen:** the thresholds were chosen by eye on practice-film frames. A first version
     used CLIP descriptions ("a completely black screen", "film credits", …), but it flagged 44% of
     practice seconds, mostly ordinary scenes, and was dropped before any labelling.
   - **Not caught:** title cards and credits on a visible background.
4. **Cut-off.** A scene is returned only if its score is above one threshold τ, the same for every ad.
   The method returns up to 10 scenes per ad, possibly fewer, possibly none.

Everything else is as in v1: exact search over every second of the test films, grouped by the Phase 3
scenes (threshold 0.75).

## Choosing τ (practice films only)

- **Labels:** on the 21 practice films, v2's top 15 scenes per ad (without the cut-off) are pooled,
  shuffled and labelled by the primary labeller. The rules and display are the same as Phase 4, and the
  judge is blind to the scores.
- **The rule:** τ is the value that maximises F1 between "score > τ" and "fits" over those pairs. A tie
  goes to the lower τ.
- **Freeze:** τ and the code are frozen before any v2 run on the test films.

## Evaluation (test films, once)

- **Run:** v2 runs once on the 14 test films.
- **Labelling:**
  - New (ad, scene) pairs not labelled in Phase 4 are labelled blind by the primary labeller.
  - Pairs already labelled in Phase 4 reuse their primary labels (same labeller, same rule).
  - The project author labels a random 50 of the new pairs (`default_rng(1)`). Agreement is reported as
    in Phase 4.
- **Metrics:** means over the 20 ads, 95% bootstrap over ads (10,000 resamples, `default_rng(0)`), for
  v1, v2 and the paired difference v2 − v1:
  - **precision@10:** as in Phase 4. A scene not returned counts as not fitting, so abstaining is
    penalised.
  - **precision of returned scenes:** fitting returned scenes ÷ returned scenes, over all ads, with the
    number of scenes returned.
  - **correct abstentions:** ads where v2 returns nothing and no labelled pair for that ad fits.
  - **nDCG@10:** as in Phase 4, with the ideal ranking taken from every labelled pair for the ad
    (v1, v2 and the baselines).

## Decision rule

- **v2 improves on v1** if the 95% interval of the paired precision@10 difference (v2 − v1) is entirely
  above 0.
- If it isn't, the result is reported as "no measurable improvement", along with the
  precision-of-returned trade-off if there is one.
- A per-ad table is reported, including every ad where v2 does worse.

## Disclosures

- **The test films are reused.** They were the Phase 4 test films, and their v1 results were seen before
  this amendment was written. The changes are general rules chosen in response to failure types seen in
  v1 (scenes that vaguely match everything, black scenes and credits, no "nothing here" signal), not
  fixes for specific ads or scenes.
- **Previously seen pairs.** Some pairs may already have been seen by the primary labeller. Their earlier
  labels are reused unchanged.

---

# Phase 4c amendment: a different model, same method

Written and committed **after** the Phase 4b results and **before** any Phase 4c result on the test films.

## Question

With the matching method held fixed, does a stronger image–text model pick better scenes than CLIP ViT-B/32?

## What changes

Only the model:

- **v1 (Phase 4):** CLIP ViT-B/32, OpenCLIP `laion2b_s34b_b79k`. Its Phase 4 picks and labels are reused unchanged.
- **4c:** SigLIP2 ViT-B/16, OpenCLIP `ViT-B-16-SigLIP2` / `webli`, in PyTorch fp16 on the laptop GPU (MPS), with
  the model's own standard preprocessing.

Everything else is v1:

- **Briefs:** each brief is embedded as written, with no templates.
- **Search:** exact search over every second of the 14 test films, at the same 1 fps seconds as the corpus
  embeddings.
- **Scenes:** each scene is returned once, at its best second; top 10 per ad, with no cut-off and no junk filter.
- **Scene boundaries:** the scenes are the Phase 3 scenes (detected from the ViT-B/32 embeddings, threshold 0.75), so
  both models choose from exactly the same set of scenes.

The Phase 4b changes (templates, neutral baseline, junk filter, cut-off) are left out on purpose: the cut-off would
need re-calibrating for the new model's score scale, and testing one change at a time keeps the answer clean.

## How the model was chosen (practice films only)

- **Candidates:** three models were tried on the 300 labelled Phase 4b practice pairs, all from OpenCLIP:
  ViT-L/14 (`dfn2b`), SigLIP2 ViT-B/16 (`webli`) and SigLIP2 SO400M/14 (`webli`).
- **Measure:** how well each one ranks the fitting pairs above the non-fitting ones, within each ad's 15 pairs (AUC,
  on the 16 ads that have both kinds).
- **Result:** ViT-B/32 scored 0.57. All three candidates scored about 0.64–0.65, with overlapping intervals.
- **Choice:** SigLIP2 ViT-B/16, the fastest of the three on the laptop.

Only this one model is run on the test films. That check was exploratory and is not a reported result.

## Evaluation (test films, once)

- **Labelling:**
  - (ad, scene) pairs not labelled in Phase 4 or 4b are pooled, shuffled (`default_rng(0)`) and labelled blind by the
    primary labeller (same display, same rule).
  - Pairs labelled earlier reuse their primary labels.
  - The project author labels a random 50 of the new pairs (`default_rng(2)`), and agreement is reported as before.
- **Metrics:** precision@10 and nDCG@10 for v1 and 4c, and the paired difference 4c − v1, with means over the 20 ads
  and a 95% bootstrap over ads (10,000 resamples, `default_rng(0)`).
- **nDCG's ideal ranking** uses every labelled pair for the ad (Phases 4, 4b and 4c).
- **Sensitivity:** the same comparison with the author's labels used wherever they exist.

## Decision rule

- **4c improves on v1** if the 95% interval of the paired precision@10 difference is entirely above 0. Otherwise,
  the result is reported as "no measurable improvement".
- A per-ad table is reported, including every ad where 4c does worse.

## Disclosures

- **The test films are used a third time.** v1 and v2 results on them were seen before this amendment. The model was
  chosen on practice films only, and the method is unchanged from v1.
- **One model, chosen in advance.** Running all three candidates on the test films and reporting the best would
  inflate the result.
