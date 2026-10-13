# Phase 5 — brand safety: results (with the 5b verification pass and the 5c tuning round)

**Question:** what does brand safety cost in sellable inventory, and how often does it block a scene that was fine?

**Rules:** the Phase 5 amendment and the Phase 5b addendum in `EVAL-PROTOCOL.md`, each frozen by hash before it was
run (`rules_hash.txt`), committed afterwards.

## Setup

- **Scenes:** the 19 films with scene cards (Phases 4g and 4h): 4,922 scenes, 4,903 ad breaks (one between each pair
  of consecutive scenes).
- **Ads:** the 60 Phase 4d ads.
- **Detector (no training):** the card model (Qwen3-VL-4B) looks at each scene's 4 frames and dialogue and lists which
  of 9 content tags are present: violence, weapons, accidents, death, horror, alcohol, smoking, sexual, drugs.
  0.36 s per scene on this laptop.
- **Three kinds of rule** (`configs/safety/`):
  - **category avoidance**, the only part that differs between rule sets:
    - lenient blocks sexual, drugs and death
    - standard adds violence and horror
    - strict adds weapons, accidents, alcohol and smoking
  - **adjacency norms by ad category**, for example no automotive ad next to an accident
  - **each ad's own avoid list**, written before Phase 4

  A break is checked against both neighbouring scenes.
- **Baseline:** block whole films by genre:
  - lenient: horror
  - standard: horror and crime
  - strict: adds western and adventure
- **Hand-check:** a stratified random sample of scenes, labelled blind by 4 fresh instances of the AI assistant from
  the written tag definitions. Rates are weighted back to all scenes.

## Phase 5: the detector blocks far too much

480 scenes labelled. Rates over all 4,922 scenes:

| Rule set | Should be blocked | Detector blocks | Wrong blocks (of blocked) | Missed (of kept) | Risky scenes caught |
|---|---|---|---|---|---|
| lenient | 1.9% | 11.3% | **83%** [77, 89] | 0.0% | 100% |
| standard | 6.2% | 17.7% | **69%** [62, 75] | 0.8% [0.0, 2.4] | 90% [72, 100] |
| strict | 14.6% | 22.5% | **52%** [47, 58] | 5.0% [1.7, 9.2] | 73% [60, 90] |
| *genre, standard* | 6.2% | 26.1% | *94%* | *6.4%* | *24%* |
| *genre, strict* | 14.6% | 58.7% | *83%* | *10.7%* | *70%* |

- **The detector rarely misses, but most of its flags are wrong.** Precision per tag ranged from 0.05 (sexual) and
  0.19 (death) to 0.48 (weapons). Asked to check nine things at once, the small model says yes too often.
- **Genre blocking is worse on both counts:** it blocks more good scenes *and* catches fewer risky ones.

Source: `safety_cpu-laptop_seed0.json`, `inventory_cpu-laptop_seed0.json`.

## Phase 5b: a second, narrow question fixes most of it

**Change (pre-registered after the Phase 5 result):** for each tag the detector gives a scene, the same model is asked
one question with that tag's definition ("is this clearly present … ? yes or no"). The tag is kept only on yes. 2,045
checks at 0.26 s each; 38% of tags kept.

**Test:** a **fresh** sample of 452 scenes (the 480 Phase 5 scenes left out), labelled blind the same way. Both
detectors are scored on the same new labels. Rates over the remaining 4,442 scenes:

| Rule set | Should be blocked | Blocks: detector → verified | Wrong blocks: detector → verified | Missed (of kept) | Risky scenes caught |
|---|---|---|---|---|---|
| lenient | 1.0% | 8.9% → **3.5%** | 89% → **74%** | 0.0% → 0.1% | 100% → 94% |
| standard | 3.8% | 13.9% → **5.6%** | 78% → **53%** | 0.9% → 1.2% | 80% → 71% |
| strict | 9.1% | 17.3% → **9.8%** | 65% → **38%** | 3.6% → 3.3% | 67% → 67% |
| *genre, standard* | 3.8% | *25.0%* | *95%* | *3.2%* | *36%* |

**Paired differences, verified − detector** (standard rule set):
- wrong blocks −25.8 points [−32.5, −17.7]
- missed +0.3 points [+0.0, +0.6]
- risky scenes caught −9 points [−16, −4]

**Decision (pre-registered rule): the verifier helps.** The false-positive rate falls (interval entirely below 0)
and the miss rate rises by less than 3 points.

**The cost, reported as prominently:** the verifier lets through about 1 in 10 more of the risky scenes under
standard rules (caught 80% → 71%). The miss rate barely moves only because risky scenes are rare (3.8%). At strict,
nothing extra is let through.

Source: `results/phase5b/safety_cpu-laptop_seed0.json`.

## Phase 5c: tuning on a development set did not carry over to fresh scenes

**Change (Phase 5c addendum):** tuned on the 932 scenes labelled in 5 and 5b (the development set):
- **more candidates:** the first pass, plus scene-card keywords, plus CLIP's top 5% per tag
- **sharper frames:** a 2 × 2 grid with 4× the pixels per frame
- **a probability threshold:** P(yes) ≥ 0.2

On the development set, under standard rules, F1 went from 0.53 (5b) to 0.57.

**Test:** 326 fresh scenes (all 932 development scenes left out), labelled blind **from the sharper grids**. Rates are
weighted to the remaining 3,990 scenes.

| Standard rule set | 5b verifier | 5c | First pass |
|---|---|---|---|
| blocks | 1.5% | 2.2% | 7.6% |
| wrong blocks | 28% [11, 70] | 62% [52, 71] | 78% |
| risky scenes caught | 27% [5, 65] | 20% [11, 50] | 39% |
| F1 | 0.39 [0.09, 0.73] | 0.26 [0.16, 0.46] | 0.28 |

**Decision (pre-registered rule): no measurable improvement.** 5c − 5b on F1 is −0.13 [−0.46, +0.20]. At strict it
is +0.06 [−0.12, +0.20]. The development gain did not carry over.

**The bigger finding: recall is much worse than Phases 5 and 5b showed.**
- With the sharper frames, the labellers marked weapons in 5.6% of scenes, against 2–4% from the small strips
  before.
- Every detector catches only about 1 in 5 to 2 in 5 of the risky scenes. Weapons recall is 5–12%.
- Most risky scenes in this sample are ones **no candidate source flagged at all**: 4 of the 120 scenes with no tag,
  standing for about 126 scenes in the population.
- So the earlier "caught 71–90%" was measured against labels that could not see small guns either.

**What this means:**
- The limit is finding candidates, not checking them.
- The first pass runs on the small strips, and CLIP and the card keywords do not find most guns.
- Any next step has to look at sharp frames from the start.

**Caveats:**
- The intervals are very wide. The recall figures rest on a handful of risky scenes among the unflagged ones, each
  standing for about 31 scenes.
- The fresh scenes are what is left after the development samples took many flagged scenes, so they hold fewer easy
  positives.
- The labels are still the AI assistant's.

Source: `results/phase5c/safety_cpu-laptop_seed0.json`; inventory on the 5c tags in
`results/phase5c/inventory_cpu-laptop_seed0.json` (standard: 6.5% of scenes blocked, tight-book fill 76.2%).

## Phase 5d: "obvious" definitions; a sharp-frame weapons check helps

**Change (Phase 5d addendum, at the author's request):**
- **New definitions.** A tag now counts only if it is **obvious at a glance**. Holstered guns, shadows in a fight, blurry
  rifles and swords drawn in the credits no longer count.
- **This changes what is measured.** 5d numbers are not comparable with 5, 5b and 5c.

**Development** (the 326 5c scenes, relabelled blind under the new definitions; `results/phase5d/dev_log.txt`):
- **Fewer scenes need blocking.** Under the new definitions, weapons labels fall from 38 to 19, violence from 23 to 10
  and alcohol from 26 to 10. Scenes that standard rules should block fall from 45 to 30.
- **The existing 5b verifier does much better against obvious content.** Under standard rules, it catches 47% of risky
  scenes and 31% of its blocks are wrong. Under lenient rules, it catches 93% and 34% are wrong.
- **Running the 9-tag detector on the sharp grids was worse for everything except weapons.** Violence recall was 0.08.
  It was dropped.
- **Chosen and frozen:** the 5b tags **plus weapons** found on the grid and confirmed at P(yes) ≥ 0.7. This changes
  nothing under lenient or standard rules.

**Test:** 146 fresh scenes, labelled blind under the new definitions from the grids and weighted to 3,664 scenes.

| Strict rule set | 5b verifier | 5d | Grid first pass |
|---|---|---|---|
| blocks | 0.4% | 0.7% | 7.1% |
| wrong blocks | 79% [56, 100] | 58% [38, 77] | 73% |
| risky scenes caught | 3% [0, 25] | 11% [4, 100] | 68% |
| F1 | 0.05 [0.00, 0.23] | 0.17 [0.07, 0.59] | 0.39 |

**Decision (pre-registered rule): 5d improves on 5b at strict.** 5d − 5b on F1 is +0.12 [+0.05, +0.36].
- Of the 9 test scenes labelled weapons, 5b flagged none and 5d flagged 8.
- Weapons precision was 0.57 and recall 0.89.

**The losses, reported as prominently:**
- **Few risky scenes.** Earlier rounds had already used up almost every flagged scene, so the fresh pool had 26
  flagged and 120 unflagged scenes. Only 14 test scenes are risky under strict rules and 5 under standard.
- **Under standard rules, 5b blocked only 1 of the 5 risky test scenes, and that only through a wrong tag** (a fight
  tagged sexual). 4 of the 5 are horror: a robot and skeletons from two sci-fi films.
  5b flagged none of the 4. The grid first pass flagged 3 of them, but horror was not part of the frozen change. It
  is a candidate for the next round, to be tuned on a development set and not on this test.
- **These estimates are mostly unknown.** Each unflagged scene stands for about 30 scenes. Under standard rules the
  "caught" interval for 5b runs from 0% to 40%.
- **Author check (50 random test scenes, 42 of them unflagged).**
  - **First pass:** 47 of 50 matched the AI labels exactly. The standard block decision matched on 49 of 50.
    Weapons agreement was κ 0.85.
  - **After the author reviewed the 3 disagreements**, 49 of 50 matched and weapons agreement rose to κ 1.00. The
    author changed 2 labels: a pistol held up in a tent counts as weapons, and smoke with small background flames is
    not accidents. Changes made after seeing a disagreement can only raise agreement, so both figures are given.
  - **Remaining disagreement:** a robot standing still in a lab. The AI labelled it horror; the author did not
    ("just a robot doing nothing").
  - **Why this matters:** 2 of the 4 horror scenes that 5b missed under standard rules are robots from the same
    film, so the standard-rules misses partly depend on this definition. Next round's horror definition should say
    whether a creature has to be menacing (attacking, threatening or gruesome) to count.

Source: `results/phase5d/safety_cpu-laptop_seed0.json`, `results/phase5d/inventory_cpu-laptop_seed0.json` (strict: 13.2%
of scenes blocked; tight-book fill 67.5%).

## Phase 5e: sharp-frame checks for horror, smoking and alcohol did not help

**Research (development set: the 472 scenes with obvious-definition labels; `results/phase5e/dev_log.txt`).**
- **Where 5d fails:** it caught horror in 0 of 14 scenes and violence in 2 of 11.
- **Tried on the sharp grids:** checks for horror, smoking and alcohol, done the same way as the 5d weapons check.
  - These looked good on the development set: horror 0 → 7 of 14, alcohol 3 → 7 of 10.
  - Standard F1 rose from 0.56 to 0.61.
- **Tried for violence; nothing worked:**

| Approach | Result |
|---|---|
| Asking the violence question on every scene | P(yes) near 0 for 8 of the 11 fights |
| Asking whether a gun is being fired | Near 0, even with a visible muzzle flash |
| Pretrained sound tags (AST, AudioSet) | Screams AUC 0.78, gunshots 0.69 |
| The 8B model | The same 2–3 of 11 fights, with more false alarms |

  On these frames, violence is beyond what the local models can see.

**Test:** the 452 Phase 5b scenes, which were not used to develop 5d or 5e. They were relabelled blind under the obvious
definitions, from the grids, and weighted to 4,442 scenes.

**Decision (pre-registered rule): no measurable improvement.** 5e − 5d on standard F1 is −0.04 [−0.12, 0.00].
- 5e caught no risky scene that 5d missed.
- 5e's horror tags were wrong more often: horror precision fell from 0.55 to 0.28.
- The development gain did not carry over, as in 5c. **5d stays the system.**

**This larger test is also the best estimate yet of the 5d system:**

| Rule set | Should be blocked | 5d blocks | Wrong blocks | Risky scenes caught | F1 |
|---|---|---|---|---|---|
| lenient | 0.7% | 3.5% | 83% [76, 89] | 95% [83, 100] | 0.30 [0.20, 0.39] |
| standard | 2.9% | 5.6% | 64% [57, 70] | 71% [44, 99] | 0.48 [0.38, 0.58] |
| strict | 6.0% | 10.5% | 50% [42, 58] | 87% [66, 100] | 0.63 [0.54, 0.72] |

- **5d against the 5b verifier at strict:**
  - F1 +0.07, with an interval of [0.00, +0.18]
  - weapons recall 0.46 → 0.89
  - risky scenes caught 75% → 87%
- **The weak spots:**
  - **Violence:** precision 0.37, recall 0.50
  - **Lenient rules:** few scenes should be blocked (0.7%), so most blocks are wrong
- **Genre blocking is worse at every level.**

Source: `results/phase5e/safety_cpu-laptop_seed0.json`.

## What safety costs in inventory (verified detector)

Share of relevant ad slots removed (a slot is relevant if its scene is in the ad's top 10% by the frames+cards
matcher), and fill rate when each ad wants 10 breaks (slack book) or 80 breaks (tight book):

| Rule set | Scenes blocked | Relevant slots removed | Fill, slack book | Fill, tight book |
|---|---|---|---|---|
| no rules | 0% | 0% | 100% | 85.9% |
| lenient | 3.9% | 14.0% | 100% | 77.6% |
| standard | 6.8% | 15.0% | 100% | 75.6% |
| strict | 12.8% | 22.7% | 100% | 68.1% |
| *before verification, standard* | *17.7%* | *32.7%* | *100%* | *61.0%* |
| *genre, standard / strict* | *25.4% / 60.7%* | *24.3% / 62.3%* | *100%* | *65.0% / 33.8%* |

- **Slack book:** every rule set is free; there is always another safe break.
- **Tight book:** standard rules cost about 10 points of fill (85.9% → 75.6%). Before the verifier they cost about 25
  points, and genre blocking at strict costs 52 points.
- **Relevant slots go faster than scenes** (15% vs 6.8%): each ad's own avoid list and the adjacency norms remove
  breaks for specific ads, and blocked scenes take out two breaks each.

Source: `results/phase5b/inventory_cpu-laptop_seed0.json` (verified), `inventory_cpu-laptop_seed0.json` (detector).

## Findings

1. **Watching scenes beats judging by genre for safety.** At every strictness, scene rules block less good inventory
   and catch more risky scenes than blocking whole genres.
2. **One question per tag beats nine at once.** The same model, asked narrowly, cut wrong blocks under standard rules
   from 78% to 53% and blocked inventory from 13.9% to 5.6%, at the cost of letting through about 1 in 10 more of the
   risky scenes.
3. **Under standard rules, safety costs about 10 points of fill on a tight book and nothing on a slack one.**
4. **Tuning on a development set did not carry over (5c), and sharper frames showed that recall was overstated.**
   Once labellers could see small objects, every detector caught only about 1 in 5 to 2 in 5 of the risky scenes.
   The bottleneck is finding candidates from small frames.
5. **Under the obvious definitions (5d), the existing verifier holds up much better, and sharp frames help with
   weapons.** Adding a sharp-frame weapons check improved strict F1 on fresh scenes (+0.12 [+0.05, +0.36]). The fresh
   test had few risky scenes, and horror (robots and skeletons) was still missed under standard rules.
6. **More sharp-frame checks (5e) did not help, and no local model, prompt or sound tag fixed violence.** On 452
   relabelled scenes, the final 5d system catches about 7 in 10 risky scenes under standard rules. About 2 in 3 of its
   blocks are wrong there. Under strict rules it catches about 9 in 10, and half of its blocks are wrong.

## Deviations from the protocol

- **Rules frozen by hash, committed later** (`rules_hash.txt`), as in 4g and 4h.
- **Prompt practice:** a second detector prompt ("judge from the frames …") was tried on 60 practice scenes. It
  produced more spurious tags and was rejected before the test.
- **Labelling instructions** were added to the amendment while the test films were being tagged, before any pool or
  label existed (logged).
- **"Risky scenes caught"** (recall of the block decision) was added to the scoring after the 5b result. It is
  descriptive; the decision rule does not use it. Reported because a low miss rate can hide a real loss when risky
  scenes are rare.
- **The 5b addendum was written after the Phase 5 result.** It was tested on fresh scenes, so the Phase 5 labels were
  not reused for the decision.

## Limits

- **The labels are by the same AI model that built the system,** told "if you can't tell, don't mark it". That
  leans towards calling scenes fine, which raises the false-positive rates. No human labels yet.
- **Labellers were not fully consistent on edge cases** (animals fighting as violence, skeletons as horror, cattle
  stampedes as accidents), as their own notes say.
- **Small numbers of risky scenes** in each sample, so the "caught" intervals are wide (for example 48–93% at
  standard).
- **The detector can only find what it flags first.** The verifier removes tags, never adds them, so its recall is
  capped by the first pass (weapons and violence recall about 0.5–0.6).
- **Old films:** few sexual or drug scenes, so those tags are barely tested.
- **Breaks at every scene boundary** is a simplification (Phase 7 chooses real break positions). The adjacency norms
  and rule sets are the project's own, written from common practice.
