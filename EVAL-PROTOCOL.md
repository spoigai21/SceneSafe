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

---

# Phase 4d amendment: the model comparison, re-run bigger on fresh films

Written and committed **after** the Phase 4c results and **before** any Phase 4d film was embedded, viewed or
scored.

## Question

Same as 4c: with the method held fixed, does SigLIP2 ViT-B/16 pick better scenes than CLIP ViT-B/32? Phase 4c's
interval (±0.07 with 20 ads) was too wide to detect a gain of a few points, and its test films had been used three
times. Phase 4d asks the same question with three times the ads, fresh films and a written label guide.

## Data

**Films:** 14 new films, listed in `corpus/manifest_4d.json`.
- **Selection:** made by `scripts/build_manifest_4d.py`, with the same rules as the original corpus (public
  domain, 60–120 min, the same genre vocabulary and exclusions). It skips every title already in the corpus and
  takes the first 2 per genre in download-count order.
- **Use:** none of these films has been used for anything before.
- **Practice films:** none. Nothing is tuned in this phase.

**Ads:** 60.
- **The original 20** (`configs/ads.yaml`, unchanged).
- **40 new ads** (`configs/ads_extra.yaml`).
  - **Written:** after the new film titles were known, but before any frame of them was seen.
  - **Content:** everyday subjects, none written for a particular film.

**Label guide (`configs/label_guide.yaml`):** for each of the 60 ads, one line on what fits and one on what
doesn't.
- **Purpose:** it settles in advance the judgement calls that caused the Phase 4b and 4c disagreements. For example:
  - wine needs a dinner table
  - a held-up banknote counts for the bank ad
  - a lion is not a dog
- **Who sees it:** both labellers.
- **The general rule** from Phase 4 is unchanged. The guide narrows it; it doesn't replace it.

**Scenes:** the Phase 3 detector (threshold 0.75, minimum 3 s), run on the new films' ViT-B/32 embeddings.

## Methods

Both models use the v1 method, exactly as in Phase 4c:
- the brief as written
- exact search over every second
- each scene once, at its best second
- top 10 per ad, with no cut-off and no junk filter

The models:
- **ViT-B/32** embeds the films exactly as in Phase 2.
- **SigLIP2** uses its own preprocessing, as in 4c.

## Labels

- **Pool:** every (ad, scene) pair picked by either model, shuffled (`default_rng(0)`).
- **What the judge sees:** each card shows the ad's category and brief, its label-guide line, and 4 frames from the
  scene. Nothing on the card says which model picked it.
- **Primary labeller:** labels every pair, blind to the model.
- **Second labeller:** the project author labels a random 100 (`default_rng(3)`).
- **Agreement:** reported as before, with a list of any cards discussed during labelling.

## Metrics

- **precision@10 and nDCG@10:** per model; the ideal ranking uses every fitting pair in the pool for that ad.
- **Difference:** the paired difference SigLIP2 − ViT-B/32.
- **Uncertainty:** means over 60 ads, 95% bootstrap over ads (10,000 resamples, `default_rng(0)`).
- **Sensitivity:** the author's labels used wherever they exist.
- **Descriptive only:** picks that are mostly near-black by the Phase 4b pixel rule.

## Decision rule

- **SigLIP2 improves on ViT-B/32** if the 95% interval of the paired precision@10 difference is entirely above 0.
- **ViT-B/32 is better** if the interval is entirely below 0.
- Otherwise, "no measurable difference".
- A per-ad table is reported, including every ad where SigLIP2 does worse.

## Disclosures

- **Seen before:** the 4c result (+0.025, not measurable) and the model choice (made on practice films) were known
  before this amendment.
- **Who wrote the guide:** the primary labeller wrote it after labelling Phases 4–4c. Some lines reflect cases
  seen in those phases' test films, which are not used here.
- **Who labels:** the primary labeller is the AI assistant that wrote the pipeline, as in every earlier phase.
- **Data counts:** the new films are kept in a separate manifest, so the corpus counts reported for Phases 0–4c
  don't change.

---

# Phase 4e amendment: thematic relevance

Written and committed **after** the Phase 4d results and **before** the Phase 4e run or any Phase 4e label.

## Question

Phases 4–4d judged a scene relevant only if the ad's literal subject was visible. Contextual advertising also sells
**thematic** matches, where the scene is about the ad's world or mood: a romantic scene for jewellery, or a glamorous
woman for cosmetics. Phase 4e asks two things:

1. How often do the current system's picks suit the ad thematically?
2. Does describing each ad by its theme, as well as its literal brief, find suitable scenes more often?

## Labels: three levels

- **2, literal fit:** the Phase 4d guide's "fits" line (`configs/label_guide.yaml`). This is unchanged.
- **1, thematic fit:** not a literal fit, but the scene is about the ad's theme, as given by that ad's `theme` line in
  `configs/themes.yaml`.
- **0, neither.** Unsure still means 0.

The judge sees the brief, the literal guide line, the theme line and 4 frames, with nothing about which system
picked the scene.

## Systems (ViT-B/32, the 14 Phase 4d films, Phase 3 scenes, v1 method, top 10 per ad)

- **Literal:** the Phase 4d ViT-B/32 run (the brief as written). Its picks are reused unchanged.
- **Theme:** the same, except each ad's query vector is the unit mean of 4 ViT-B/32 text embeddings: the brief plus
  the ad's 3 `query` phrases from `configs/themes.yaml`. The phrases were written once per ad and are not tuned.

## Labelling

- **Carried over:** pairs the primary labeller judged a literal fit in Phase 4d keep label 2. The literal rule hasn't
  changed.
- **The pool:** every other pair from either system (Phase 4d non-fits plus the theme system's new pairs) goes into
  one shuffled pool (`default_rng(0)`). The primary labeller labels it 0/1/2, blind to the system.
  - **Disagreements with 4d:** a 4d non-fit may now receive a 2. If so, the 4e label stands, and the number of such
    pairs is reported.
- **Second labeller:** the project author labels a random 100 of the pool (`default_rng(4)`) on the same three levels.
- **Agreement:** exact agreement across all three levels, plus Cohen's kappa for "thematic or literal" vs neither,
  and for literal vs not.

## Metrics

Means over 60 ads, with a 95% bootstrap over ads (10,000 resamples, `default_rng(0)`). For each system and for the
paired difference theme − literal:

- **thematic precision@10 (decision metric):** the share of the top 10 labelled 1 or 2.
- **literal precision@10:** the share labelled 2. This shows the trade-off.
- **graded nDCG@10:** gain = label (0/1/2). The ideal ranking uses every labelled pair for the ad.
- **Sensitivity:** the author's labels used wherever they exist.

## Decision rule

- **The theme system improves on the literal system** if the 95% interval of the paired thematic precision@10
  difference is entirely above 0.
- **It is worse** if the interval is entirely below 0.
- Otherwise, "no measurable difference".

Whatever the result, the literal precision of both systems is reported next to it, with a per-ad table.

## Disclosures

- **This is a different and broader relevance question than Phases 4–4d.**
  - Thematic precision is not comparable with the earlier literal numbers and is never reported as an improvement on
    them.
  - Both levels are reported side by side.
- **The themes were written by the primary labeller after labelling the Phase 4d cards for these same films.**
  - They are written per ad, generically ("romance or a special occasion"), and none names a film or a scene.
  - Even so, the writer had seen what these films contain. Fresh films would remove this, and the cost was judged
    too high for this phase.
- **Thematic judgements are more subjective than literal ones.** The theme lines narrow them, and agreement is
  reported.

---

# Phase 4f amendment: adding dialogue

Written and committed **after** the Phase 4e labels (the second labeller hadn't labelled 4e) and **before** any
Phase 4f run or label. Whisper transcripts were produced before this commit, and only Whisper's own output was
inspected (see "Transcript check before this commit" below).

## Question

Does adding what is **said** in a scene to what is **shown** find scenes that suit the ad more often than frames alone?

## Data

- **Films, scenes and ads:** the 14 Phase 4d films and their Phase 3 scenes (threshold 0.75), and the 60 ads.
- **Labels and guides:** the Phase 4d literal guide and the Phase 4e theme lines, unchanged.

## Transcripts

- **Source:** Whisper (`mlx-community/whisper-large-v3-turbo`, mlx-whisper on the laptop GPU), English, run on each
  film's full audio (decoded with PyAV to 16 kHz mono).
- **What counts as speech:** a segment is dropped if Whisper's `no_speech_prob` is above 0.6 or its average
  log-probability is below −1.0 (Whisper's own defaults). mlx-whisper reports `no_speech_prob` as 0 for every
  segment, so that check does nothing here. Two more rules, set after reading the transcripts (below):
  - a segment of 10 s or more with under 0.5 words per second is dropped (Whisper filling music or silence with a
    stock phrase such as "Thank you." or "The End");
  - a segment with the same text as the one before it is dropped (Whisper stuck in a loop repeating one line).
- **A scene's dialogue:** the speech segments whose midpoint falls inside the scene.
- **Accuracy:** transcripts are not checked against human subtitles. The accuracy of Whisper on these films is
  unmeasured, and it's reported as a limit.
- **Transcript check before this commit:** the transcripts (not any scores, picks or labels) were read for
  problems. Ordinary dialogue read as plausible and on-plot. Two kinds of junk were found and led to the two rules
  above: 571 long, nearly empty segments ("The End" 105 times, "Thank you." 73 times, mostly exactly 30 s long) and
  repetition loops (e.g. one line repeated 20 times at 1 s steps).

## Systems (top 10 scenes per ad)

1. **Frames:** the Phase 4d ViT-B/32 system (the brief as written, each scene at its best second). Its picks are
   reused unchanged.
2. **Dialogue:**
   - **Model:** sentence-transformers `all-mpnet-base-v2`.
   - **The ad's text:** its brief, then ". ", then its theme line.
   - **How a scene is scored:** its dialogue is split into windows of 50 words (stride 25), and the scene scores as
     its best window's cosine similarity.
   - Scenes without speech are never picked.
3. **Both:**
   - **The combined score:** for each ad, the frame score and the dialogue score are each z-scored over the scenes
     that have them, then added with equal weight.
   - **Scenes without speech:** they get 0 for dialogue, the average.
   - **Fixed in advance:** the weight is never tuned.

## Labels

- **The pool:** every (ad, scene) pair picked by any system, shuffled (`default_rng(0)`). **All are labelled again.**
- **What every card shows:** the brief, the literal guide line, the theme line, the 4 frames, and the scene's
  dialogue (cut to 600 characters on the card), with nothing about the system.
- **Why relabel everything:** Phase 4d and 4e labels came from silent frames, so they would be unfair to dialogue.
- **Levels:** 2 = literal fit, 1 = thematic fit, 0 = neither, as in Phase 4e. The judge may use the dialogue to decide
  what a scene is about.
- **Primary labeller:** labels the whole pool.
- **Second labeller:** the project author labels a random 100 (`default_rng(5)`). Agreement is reported as in 4e.

## Metrics

Means over 60 ads, with a 95% bootstrap over ads (10,000 resamples, `default_rng(0)`).

- **For each system:**
  - thematic precision@10 (labels 1 or 2)
  - literal precision@10 (label 2)
  - graded nDCG@10
- **Paired differences:**
  - both − frames
  - dialogue − frames
  - both − dialogue
- **Also reported:**
  - how many scenes and picks have speech
  - sensitivity with the author's labels where present

## Decision rule

- **Dialogue improves matching** if the 95% interval of the paired **thematic precision@10** difference
  (both − frames) is entirely above 0.
- **It makes matching worse** if the interval is entirely below 0.
- Otherwise, "no measurable difference".

The other differences and literal precision are reported whatever the result.

## Disclosures

- **Fourth use of these films.** These films were used in Phases 4d and 4e, and the primary labeller has seen their
  frames. Their dialogue has not been read, apart from the check above. A confirmation on fresh films would be needed
  before the result is treated as general.
- **Different evidence from earlier phases.** Labels now include dialogue, so 4f numbers are not comparable with 4d
  or 4e numbers, even for the same frames system.
- **Who wrote the themes.** The primary labeller wrote the theme lines, and they are reused unchanged in the
  dialogue query.

---

# Phase 4g amendment: scene cards (a written description of each scene)

Written and committed **after** the Phase 4f result and two practice rounds (below), and **before** any card was
written for the test films or any Phase 4g pick was made or labelled.

## Question

Does matching ads against a short written description of each scene, made by a vision-language model that looks at
the frames and reads the dialogue, find scenes that suit the ad more often than frames alone?

## Why, and what was tried first (practice, not reported as results)

- **Phase 4f:** adding dialogue to frames gave no measurable gain (thematic precision@10 both − frames −0.017
  [−0.087, +0.052]).
- **Practice round 1, fusion rules (all 14 Phase 4d films, Phase 4f labels plus 278 extra practice labels):** none of
  the rules tried beat frames alone: stronger-of-two, average of available signals, frames-gated, mixing slots,
  re-ranking frames' top 20 with dialogue, and z-score + weight + threshold + max as in ContextIQ (WACV 2025). All were
  within ±0.02 of frames. Conclusion: the dialogue signal itself was weak, not the way it was combined.
- **Practice round 2, scene cards (2 films only: `calvary`, `Hitchcock_Secret_Agent`; top 5 per ad; 60 ads; every
  pick labelled):** thematic fit frames 0.287, cards 0.443, frames+cards 0.423; frames+cards − frames +0.137
  [+0.083, +0.193]. The prompt, decoding settings and matched fields were settled on these 2 films, which is why they
  are left out of the test.

## Data

- **Test films:** the 12 Phase 4d films other than the 2 practice films, with their Phase 3 scenes (threshold 0.75).
- **Ads, guide and theme lines:** the 60 Phase 4d ads, unchanged.
- **Transcripts:** the Phase 4f transcripts and speech rules, unchanged.

## Scene cards

- **Model:** `mlx-community/Qwen3-VL-4B-Instruct-4bit` (mlx-vlm, laptop GPU), greedy decoding (temperature 0),
  repetition penalty 1.15, at most 90 new tokens.
- **Input:** the scene's 4-frame strip (the same image the labeller sees) and its dialogue, cut to 600 characters at
  a word.
- **Prompt:** fixed in `scenesafe/match/cards.py` (`PROMPT`). It asks for five lines: place, visible (up to 6 things),
  activity, talk (up to 3 dialogue topics, or none), mood.
- **Matched text:** the place, visible, activity and talk lines joined, keys removed, "none" dropped. Mood is not
  matched.
- **Every card is written once and cached** (`data/cards/`); cards are not edited or regenerated.

## Systems (top 10 scenes per ad, over the 12 test films)

1. **Frames:** the Phase 4d ViT-B/32 scene scores (the brief as written, each scene at its best second).
2. **Cards:** cosine similarity between the card text and the ad text (brief, ". ", theme line), both embedded with
   `all-mpnet-base-v2` as in Phase 4f.
3. **Frames+cards:** for each ad, the frame score and the card score are each z-scored over all test scenes, then
   added with equal weight. The weight is never tuned.

## Labels

- **The pool:** every (ad, scene) pair picked by any system, shuffled (`default_rng(0)`), ids `G####`.
- **What every labelling card shows:** as in Phase 4f: the brief, literal guide line, theme line, 4 frames and the
  dialogue (cut to 600 characters). **The scene card text is never shown**, and nothing says which system picked it.
- **Levels:** 2 = literal fit, 1 = thematic fit, 0 = neither, as in 4e and 4f.
- **Primary labeller:** the AI assistant labels the whole pool. As in Phase 4f, the work may be split across parallel
  instances of the same model with the same written rules; this is reported.
- **Second labeller:** the project author labels a random 100 (`default_rng(6)`). Agreement is reported as in 4f.

## Metrics

Means over 60 ads, with a 95% bootstrap over ads (10,000 resamples, `default_rng(0)`).

- **For each system:** thematic precision@10, literal precision@10, graded nDCG@10.
- **Paired differences:** frames+cards − frames, cards − frames, frames+cards − cards.
- **Also reported:** sensitivity with the author's labels where present.

## Decision rule

- **Scene cards improve matching** if the 95% interval of the paired **thematic precision@10** difference
  (frames+cards − frames) is entirely above 0.
- **They make matching worse** if the interval is entirely below 0.
- Otherwise, "no measurable difference".

Literal precision and the other differences are reported whatever the result.

## Disclosures

- **Fifth use of these 12 films.** They were used in Phases 4d, 4e and 4f, and in fusion practice round 1. The
  primary labeller has seen their frames and their Phase 4f pool. The card method was never run on them before this
  commit. A confirmation on fresh films would be needed before the result is treated as general.
- **Not comparable with earlier phases:** different film set (12, not 14) and different pools.
- **The card model is not the labeller.** Cards come from Qwen3-VL; labels from the AI assistant (Claude) and the
  author. The labeller never sees the cards.
- **Who wrote the themes.** The primary labeller wrote the theme lines, which are part of the ad text matched here.

## Phase 4g addendum: relabel under the author's labelling rules (written after the result)

Written **after** the Phase 4g result and the author's 51 labels, so it is a sensitivity analysis, not the
pre-registered result. The pre-registered labels (`labels_primary.json`) and their result stay the headline; the
relabel (`labels_relabel.json`) is reported beside them.

- **Why:** agreement on 51 pairs was moderate (exact 61%, thematic-or-better κ 0.49). Going through the 20
  disagreements with the author showed the primary labeller was stricter on themes and looser on literal fits.
- **The author's rules, written down:**
  1. **Dialogue makes a theme fit (1) when the conversation is about the topic**, even if nothing on screen shows it
     (a captain talking about the voyage and the sharks fits a sea theme). One passing line in an unrelated
     conversation is not enough.
  2. **A real but loose link fits the theme (1):** a headline or sign about the topic (a newspaper headline about
     money for the money theme), or a setting that belongs to the theme (a street with goods outside shops for
     markets; a rider for horses).
  3. **A literal fit (2) needs the thing clearly in focus and clearly recognisable.** If it is small, partly seen or
     uncertain, it is at most 1.
  4. **A glamorous or beautiful woman as the clear focus fits a beauty theme (1).**
  5. **If it is unclear whether the thing is there at all** (dark, blurred), it is not 2; it is 1 only if the scene
     fits the theme anyway.
- **Author corrections** (author's own call, before the relabel): three of the author's labels changed while the
  author reviewed cards for the discussion, two 1 → 0 and one 1 → 2 (`corrections` in `labels_author.json`).
- **How:** the whole pool is relabelled from scratch under these rules, blind to the system and without the scene
  cards, as before. The primary labeller saw the 20 disagreement cards while learning the rules.

---

# Phase 4h amendment: confirmation on fresh films, and a re-rank by the card model

Written **after** the Phase 4g result and **before** the Phase 4h films were selected, downloaded or viewed.

## Questions

1. **Confirmation:** do scene cards still beat frames on films that no part of the method has been tuned or tested
   on?
2. **Re-rank:** does letting the card model judge a shortlist of candidate scenes improve on frames+cards?

## Data

- **Films:** fresh public-domain films chosen by `scripts/build_manifest_4h.py`: the corpus rules of
  `build_manifest.py`, skipping every title in `corpus/manifest.json` and `corpus/manifest_4d.json`, **1 per genre**
  (the first in download-count order). The list is written to `corpus/manifest_4h.json`.
- **Scenes:** the Phase 3 detector (threshold 0.75) on ViT-B/32 embeddings made exactly as in Phase 2.
- **Ads, guide and theme lines:** the 60 Phase 4d ads, unchanged. The frame queries are the Phase 4d ViT-B/32 brief
  embeddings (`data/phase4d/briefs_b32.npy`).
- **Transcripts:** Whisper as in Phase 4f, with the same speech rules.

## Systems (top 10 scenes per ad)

1. **Frames:** the Phase 4d ViT-B/32 scene scores.
2. **Frames+cards:** exactly as in Phase 4g (same card model, prompt, decoding, matched fields, text model and equal
   weight).
3. **Rerank:**
   - **Shortlist:** frames+cards' top 30 for the ad.
   - **Judge:** the card model (`mlx-community/Qwen3-VL-4B-Instruct-4bit`) sees the scene's 4-frame strip and dialogue
     (cut to 600 characters) and the ad's brief, literal guide line and theme line, and answers one digit (prompt
     `JUDGE_PROMPT` in `scenesafe/match/cards.py`; greedy, at most 3 tokens). 2 = clearly shows the literal fit,
     1 = fits the theme, 0 = neither. An answer with no digit counts as 0.
   - **Order:** by the judge's digit, high to low, ties by the frames+cards score. Top 10 kept.
   - **Checked before this commit only on the 2 Phase 4g practice films** (14 scene–ad checks, for speed and that the
     answer parses). No test film was used.

## Labels

- **The pool:** every (ad, scene) pair picked by any system, shuffled (`default_rng(0)`), ids `H####`.
- **What every labelling card shows:** as in Phase 4g. The scene cards and the judge's answers are never shown.
- **Rules:** the 4e levels with the author's rules from the Phase 4g addendum (dialogue about the topic fits the
  theme; loose but real links fit the theme; 2 only when clearly in focus; glamour fits beauty themes; unclear → not
  2).
- **Primary labeller:** the AI assistant, possibly split across parallel instances with the same written rules.
- **Second labeller:** the project author labels a random 100 (`default_rng(7)`), if they choose to.

## Metrics

As in Phase 4g: thematic precision@10, literal precision@10 and graded nDCG@10 per system; means over 60 ads with a
95% bootstrap over ads (10,000 resamples, `default_rng(0)`). Paired differences: frames+cards − frames,
rerank − frames+cards, rerank − frames. Also reported: the judge's answer counts, and the ads where the judge gave 0
to all 30 shortlisted scenes ("nothing here").

## Decision rules

Both on thematic precision@10:

1. **Cards confirmed** if the 95% interval of frames+cards − frames is entirely above 0; **not confirmed**
   otherwise (below 0: worse).
2. **Re-rank helps** if the interval of rerank − frames+cards is entirely above 0; **hurts** if entirely below 0;
   otherwise no measurable difference.

## Disclosures

- **The judge sees the same guide and theme lines as the labeller.** An advertiser would supply such a description,
  but it makes the judge's task close to the labelling task.
- **The judge and the card writer are the same model; the labeller is a different one** (Claude), and never sees
  the cards or the judge's answers.
- **The primary labeller built the system** and wrote the guide and theme lines.
- **Fewer films than Phase 4g** (1 per genre), chosen for time on the laptop.

## Phase 4h addendum: sharper labelling rules, a relabel, a repeat check, and the matcher (written after the result)

Written **after** the Phase 4h result and the author's 100 labels, so everything here is a sensitivity analysis or
a follow-up. The pre-registered labels (`labels_primary.json`) and their result stay the headline.

- **Why:** agreement with the author on 100 pairs was moderate (exact 64%, thematic-or-better κ 0.50, literal κ
  0.59). With the author's labels the re-rank gain was +0.022 [−0.003, +0.047].
- **What the 36 disagreements and the author's questions while labelling showed:** the author gives 2 only when the
  ad's picture is the main subject of the scene, and gives 2 more readily than the primary labeller when it is
  (a small boat filling the frame; a car on a road). The author gives 1 for settings from the theme's list even
  without dialogue, and 0 for a single topical word or a generic setting.
- **Rules added to the Phase 4g addendum's rules:**
  6. **2 means the guide's literal picture is the main subject: what the scene is about on screen.** If the thing is
     only the background or setting for something else (a bar behind a couple flirting, a bed in a breakfast
     conversation, a suit on a man in a murder scene), it is 1. If it is the main subject, it is 2 even when old
     footage is imperfect.
  7. **Size and number words in the guide count.** "A busy street with crowds and traffic" needs many people and
     vehicles; a quiet street with a few people is 1.
  8. **The literal case is part of the theme.** If a scene meets the literal line, it is 2, never 1.
  9. **The guide's "not" examples are not 2.** They are 1 only if the theme line fits for another reason (a radio
     operator at a control console is 0 for a radio ad whose theme is broadcasts at home).
  10. **A place or activity named in the theme line fits the theme (1) by itself** (rugged mountain terrain for a
      mountain-holiday theme, a garden for gardening). A single topical word in unrelated dialogue, or a generic
      setting (a corridor, an office for a hotel ad), is 0.
- **How:** the whole 4h pool (1,086 pairs) is relabelled from scratch under rules 1–10, blind to the system, without
  the cards or the judge's answers, and without the author's labels, split across 4 parallel instances of the same
  model. Result: `labels_relabel.json` and `match_confirm_relabel_cpu-laptop_seed0.json`.
- **Not independent:** the new rules were written from the author's 100 labels, so agreement on those 100 after the
  relabel is not an independent measure. It is reported, with this caveat.
- **Repeat check:** the card writer and the judge are re-run from scratch on a random sample (`default_rng(0)`; 30
  cards, 60 judgements) and compared with the stored outputs (`repeat.json`). Decoding is greedy, so they should be
  identical.
- **The matcher:** whatever the relabel shows, **frames+cards is the project's matcher**. The re-rank was not robust
  to the author's labels; it is reported as an optional extra with an unclear gain, and is not tuned further on
  these films (that would be tuning on the test).
