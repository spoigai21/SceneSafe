# Phase 7 — scene signals become priced inventory: results

**Question:** does scene-level understanding change what the inventory is worth?

**Rules:** the Phase 7 amendment in `EVAL-PROTOCOL.md` and `configs/phase7.yaml`, frozen by hash before any number was
computed (`rules_hash.txt`). Script: `scripts/run_phase7.py run`. There are no labels: this is a seeded simulation
on top of the measured scene signals (the 5d safety tags and the Phase 4g frames+cards matcher). Every value, budget
and premium is an assumption fixed in advance.

## Verdict

**Not a clear win. By the pre-registered rule, segmentation does not "help".** That rule needs segmented revenue to
beat the flat pool in at least 9 of 10 seeds in both books.
- **Slack book** (demand below supply): segmentation wins in **10 of 10** seeds. It earns 1.13× the flat pool's revenue
  (median; range 1.08–1.26), sells 1.07× as many breaks and gives advertisers 1.26× the value.
- **Tight book** (demand above supply): segmentation wins in only **7 of 10** seeds, at 1.06× revenue (0.98–1.18).
  It sells **only 0.75× as many breaks**, and advertisers get the same value (1.00×). This is the "shrinks the
  sellable pool" risk SPEC named: the segmented book refuses to sell breaks that buyers' rules forbid, while the flat
  pool sells them anyway.

## How much depends on the assumed context premium

| Premium for a relevant break (m) | Slack: revenue segmented ÷ flat | Seeds higher | Tight: revenue segmented ÷ flat | Seeds higher |
|---|---|---|---|---|
| 1 (no premium: safety only) | 1.09 (1.02–1.16) | 10/10 | **0.86** (0.75–0.90) | **0/10** |
| 1.5 | 1.11 (1.07–1.19) | 10/10 | **0.96** (0.86–1.00) | **0/10** |
| **2 (main run)** | **1.13** (1.08–1.26) | 10/10 | **1.06** (0.98–1.18) | 7/10 |
| 3 | 1.20 (1.13–1.36) | 10/10 | 1.28 (1.15–1.36) | 10/10 |

- **Safety alone (m = 1) earns more only when demand is slack.** When demand is tight, the flat pool sells everything,
  including breaks buyers' rules forbid. That earns more on paper but delivers 0.79× the advertiser value.
- **Tight demand needs a premium of about 2 or more** for segmentation to pay. Whether real advertisers pay a 2×
  premium for a relevant scene is not measured here. That number decides the answer.

## The restricted share

**350 of 4,903 breaks (7.1%) are restricted:** a neighbouring scene carries a tag every rule set blocks (sexual,
drugs, death). No simulated advertiser can buy them, so they sell 0% in both books. Every advertiser here uses at least
lenient rules, so this is a design choice of the simulation, not a market result.

Safety classes of all breaks: **clean 77.5%**, mild 10.7% (only strict blocks), sensitive 4.7%, restricted 7.1%.

Contexts:

| Context | Share of breaks |
|---|---|
| travel/outdoors | 32.2% |
| home/family | 24.6% |
| style | 15.6% |
| media/tech | 10.5% |
| money/health | 3.1% |
| nightlife | 1.2% |
| general (relevant to no ad) | 12.7% |

## Segments (seed 0, premium 2)

Valuation ordering: the mean value per break to the advertisers allowed to buy the segment.

| Segment | Breaks | Share | Advertisers allowed | Matched | Value per break | Fill, slack | Fill, tight |
|---|---|---|---|---|---|---|---|
| clean · home/family | 934 | 19.0% | 60 | 18 | 1.26 | 4% | 69% |
| clean · travel/outdoors | 1,155 | 23.6% | 60 | 17 | 1.19 | 100% | 76% |
| clean · style | 638 | 13.0% | 60 | 10 | 1.16 | 100% | 68% |
| clean · nightlife | 34 | 0.7% | 60 | 3 | 1.16 | 100% | 100% |
| clean · media/tech | 408 | 8.3% | 60 | 8 | 1.15 | 100% | 100% |
| clean · money/health | 120 | 2.4% | 60 | 4 | 1.14 | 100% | 100% |
| mild · home/family | 119 | 2.4% | 42 | 11 | 1.14 | 0% | 100% |
| mild · nightlife | 21 | 0.4% | 42 | 1 | 1.13 | 0% | 100% |
| clean · general | 509 | 10.4% | 60 | 0 | 1.07 | 0% | 100% |
| mild · style | 45 | 0.9% | 42 | 7 | 1.03 | 0% | 100% |
| mild · money/health | 14 | 0.3% | 42 | 4 | 0.97 | 0% | 100% |
| mild · general | 53 | 1.1% | 42 | 0 | 0.90 | 0% | 100% |
| mild · media/tech | 54 | 1.1% | 42 | 4 | 0.87 | 0% | 100% |
| mild · travel/outdoors | 221 | 4.5% | 42 | 15 | 0.84 | 0% | 100% |
| sensitive · (7 contexts) | 228 | 4.7% | 14 | 0–6 | 0.26–0.45 | 0% | 0–100% |
| restricted · (7 contexts) | 350 | 7.1% | 0 | 0 | 0 | 0% | 0% |

**Reading the table:**
- **The ordering is defensible:** clean and context-rich segments come first, general and mild come next, sensitive
  is far below, and restricted is worth nothing.
- **Clean home/family is valued highest but sold only 4% in the slack book.** The price-setting rule gave it a high
  price aimed at the few buyers who value it most. This is a known weakness of posted prices set one segment at a
  time, and it is reported rather than tuned away.
- **In the slack book**, the budgets run out on the higher-ordered segments before the lower ones are priced, so clean · general and all
  mild segments sell 0%.

## Limits

- **Assumed parameters:** the values, budgets, rule-set mix and the premium are all assumptions. Only the safety
  classes and contexts come from measured signals.
- **The safety tags carry the 5d error rates** (Phase 5e): at standard settings, 64% of what is blocked was fine and
  71% of risky scenes are caught. Some "clean" breaks are not clean, and some "restricted" ones are not.
- **One price per segment, no auction**, as SPEC scoped. A real exchange would price per impression.
- **Advertisers are the 60 test ads**, so demand follows their categories (travel and fashion are over-represented).
