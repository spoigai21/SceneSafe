# SceneSafe — contextual ad targeting and brand safety from video

**One line:** watch a catalogue of video, understand each scene, and decide which ads are safe
and relevant to play next to it — built as a GPU pipeline with a measured cost per hour of
catalogue analysed.

Move this file into its own public repo before starting. Keep any "why this company" note
gitignored, the way `WHY-NETFLIX.md` is for AdFill.

---

## Why it exists

Third-party cookies are going away, so ad targeting is moving back to **context**: what is on
screen, rather than who is watching. A streaming service owns its catalogue, so it is the only
party that can do this properly for its own inventory.

Two commercial jobs, both real:

- **Relevance** — a cooking scene is worth more to a food brand than a car chase is.
- **Brand safety** — a car ad immediately after a crash scene is how an advertiser stops
  returning your calls.

Both require actually watching the content at scale, which is a throughput problem before it is
a machine-learning problem. That is the project.

**Where it sits next to the others.** AdFill decides *who gets the slot* once candidates exist.
SceneSafe produces the signal that says which ads are even eligible, and what they are worth.
Fusion Bench is one kernel made as fast as possible; SceneSafe is a whole pipeline made fast
enough to afford. Do not let the two GPU projects blur — they answer different questions.

---

## Honest limits, written before any results

State these in the README, not in a footnote.

1. **This is not a streaming service's catalogue.** It is public video. Nothing here is a claim
   about how any company's ads perform.
2. **The ad side is simulated.** There is no public corpus of real video ad creatives with
   outcomes. Ad categories and constraints are constructed, and that is stated wherever a result
   depends on them — exactly as AdFill says its viewers are simulated.
3. **Every number carries its hardware.** Laptop numbers and rented-GPU numbers are two lanes and
   are never blended into one figure. ClickRank gets this right and it is worth copying.
4. **Quality is measured against exact search on the same embeddings**, never against a
   leaderboard. The question is "what did the fast path cost me", not "am I state of the art".

---

## Phase 0 — predict before renting

Before any GPU is rented:

- Process **ten minutes of video on the laptop**. Record frames/sec for decode and frames/sec for
  the vision model separately. The ratio tells you whether the pipeline is decode-bound or
  inference-bound, which decides whether hardware decode matters at all.
- **Write down the predicted speedup and the predicted full-run cost, and commit it to git.**
  Then measure against it.

This is the `cu-predict` discipline from Fusion Bench and it is the single cheapest thing that
makes the whole project credible. A prediction that turns out wrong, kept in the repo, is worth
more than a right one written afterwards.

**Done when:** a committed `PREDICTION.md` with the measured laptop rates and the extrapolated
cost of the full run.

---

## Phase 1 — decode and sample

**Question:** how fast can frames be got off disk and onto the GPU, and how many are needed?

**Build:** a decode path with two backends — CPU (ffmpeg) and **NVDEC** hardware decode — and a
sampling policy. A two-hour film is roughly 172,000 frames, most of which are near-identical, so
the sampling policy matters more than the decoder for total work.

**Measure:**
- frames/sec for each decode backend, same clips, same machine
- total wall-clock for a fixed corpus under each
- how much can be dropped before downstream quality moves: uniform sampling at several rates
  against scene-change-triggered sampling

**Baseline:** decode every frame, CPU only.

**Done when:** a table of decode throughput per backend, and a curve of sampling rate against
downstream retrieval quality showing where it starts to hurt.

---

## Phase 2 — embed

**Question:** what does making inference fast cost in quality?

**Build:** batched vision-model inference over the sampled frames, exported once and swept across
runtimes and precisions — PyTorch eager, ONNX Runtime, TensorRT at fp32 / fp16 / int8 with a
calibration set for int8.

**Measure:**
- latency and throughput per backend and precision, at several batch sizes
- **retrieval quality of each precision against exact fp32 embeddings on the same frames** —
  recall of the exact top-k, not an external benchmark
- where the bottleneck sits at each batch size (transfer vs compute)

**Baseline:** PyTorch eager, fp32.

**Done when:** a speed/quality frontier where each point names its hardware, and a clear statement
of which precision is worth using and why.

**Trap:** int8 that is faster and quietly worse is the easiest mistake here. Measure the quality
cost or do not ship the int8 row.

---

## Phase 3 — index

**Question:** can the catalogue be searched at all, and at what memory cost?

**Build:** a vector index over the scene embeddings — exact, IVF, and IVF-PQ, with a GPU build.

**Measure:** recall of exact top-k, query latency, and index size in MB for each configuration.

**Done when:** a memory/recall/latency table, with the configuration chosen for the rest of the
project named and justified.

---

## Phase 4 — contextual matching

**Question:** is the signal real, or is this an expensive way to do nothing?

**Build:** given an ad described by category and a short text brief, retrieve the scenes it fits.

**Measure:** matching quality against two baselines that must be beaten:
- **random scenes** — the floor
- **genre-only matching** — the honest competitor, because genre metadata is free and already
  exists. If scene-level understanding cannot beat genre-level metadata, say so plainly; that is
  a real finding and a more interesting one than a win.

**Done when:** a quantified comparison against both baselines on a held-out set of titles, with
the evaluation protocol written down before the numbers are generated.

---

## Phase 5 — brand safety

**Question:** what does safety cost in inventory?

**Build:** rules over detected scene content — categories to avoid, adjacency constraints (no car
ad after a crash), and a per-advertiser exclusion list.

**Measure:**
- **what fraction of inventory each rule set removes.** AdFill's framing carries over: removal is
  harmless on a slack book and expensive on a tight one, so report both.
- false positives: scenes wrongly excluded, sampled and hand-checked, with the sample size stated

**Done when:** an inventory-loss figure per rule set, and an honest error rate from a hand-checked
sample rather than an assumed one.

---

## Phase 6 — cost

**Question:** would anyone actually run this?

**Build:** nothing new. Combine the measured throughput with real rented-GPU pricing.

**Measure:** **dollars per 1,000 hours of catalogue analysed**, stated with the hardware, the
sampling policy and the precision that produced it, plus what the same figure would be on the
naive path from Phase 1's baseline.

**Done when:** one number an engineer could put in a planning document, with its derivation in the
numbers ledger.

This is the headline. Most projects cannot answer it.

---

## Phase 7 — scene signals become priced inventory

**Why it was added:** four of the ten open Netflix Ads SWE roles are Pricing & Packaging or
Forecasting & Simulation, and they all ask for the same things — **inventory segmentation, yield
optimization, sell-through and fill rates, rate cards, inventory valuation**. Scene understanding
is upstream of exactly that: once you know which breaks are safe and contextually relevant, you
know which inventory is premium.

**Question:** does scene-level understanding actually change what the inventory is worth?

**Build:** turn the Phase 4 and 5 outputs into **segments** — buckets of ad breaks grouped by
safety class and contextual category — and attach a simple valuation to each. Not a pricing model;
a defensible ordering plus the evidence for it.

**Measure:**
- **fill rate per segment** — what share of each segment's breaks can actually be sold given the
  brand-safety exclusions from Phase 5
- **the segmentation's effect on yield** against one flat undifferentiated pool, on the same
  simulated demand
- **how much inventory lands in the restricted bucket**, which is the number that decides whether
  segmentation helps or just shrinks the sellable pool

**Baseline:** one undifferentiated pool priced flat. If segmentation does not beat it, say so —
that is a real finding, and AdFill already demonstrated that brand safety is harmless on a slack
book and costly on a tight one.

**Done when:** a fill-rate and yield table per segment, with the restricted-inventory share named.

**Connects to:** AdFill's allocator consumes exactly this. Do not rebuild AdFill's auction here —
produce the segments and stop.

---

## Phase 8 — the event spine

**Why it was added:** three of the ten roles — Ads Eventing and both Conversions API roles — are
about streaming. They name **Kafka and Flink**, *"billions of ad events daily"*, **deduplicated
"gold" datasets**, and monitoring that guarantees **accuracy, timeliness and completeness**. That
is the single largest gap in the portfolio: AdFill is batch simulation, Cine-Infer is Airflow,
and nothing anywhere is streaming.

**Vocabulary warning, because it inverts.** The Meta research says to avoid streaming-tool nouns
entirely — Kafka, Flink, Spark appear in none of Meta's ads reqs. **At Netflix they are
everywhere.** Same domain, opposite dialect. Never carry the Meta framing into a Netflix
application or the reverse.

**Question:** when ads play against these scenes, can the events be counted correctly?

**Build:** a streaming path that ingests ad events — impression, completion, abandonment, each
with an event ID, a break ID and a scene ID — deduplicates them, and aggregates into rollups.
Kafka for ingestion, a stream processor for windowed aggregation, and a queryable rollup store.

Generate the event stream from the Phase 7 segments, with deliberate real-world damage injected:
duplicates, out-of-order arrival, late events past the window, and dropped events.

**Measure, in their three words:**
- **accuracy** — counted impressions against ground truth under 5-20% duplication. AdFill already
  found naive counting overbills about 5%; this rebuilds that as a real system rather than a
  simulation, so the comparison is the point.
- **completeness** — what fraction of emitted events survive to the rollup, and where the rest go
- **timeliness** — lag from emit to queryable, at several throughput levels

Plus **sustained events/sec on one machine**, with the hardware named. Not billions; the honest
figure and what the bottleneck was.

**Baseline:** naive counting with no deduplication, and a batch job doing the same aggregation —
so the cost of streaming is visible, not assumed.

**Done when:** an accuracy/completeness/timeliness table under injected damage, a throughput
figure with its bottleneck, and an auditing check that *fails loudly* when completeness drops.

**Stack note:** the Eventing role asks for a **JVM language or C++**. His Java is current (Adorus:
Spring Boot, 24 modules, 590 tests), so writing this half in Java rather than Python is a
deliberate, defensible choice and worth stating in the README.

---

## Phase 9 — optional: fused preprocessing kernel

Only if Phases 1-8 are finished and measured.

Decode produces frames in one colour format; the model wants another size, another layout,
normalised. Done naively that is three passes with memory traffic between each. Fused into one
CUDA kernel it is one pass. NVIDIA built DALI because this is a real bottleneck.

**Measure:** the fused kernel against the naive path, same frames, same output verified
bit-for-bit or within a stated tolerance.

**Note:** this is the only hand-written CUDA in the project, and it is optional because Fusion
Bench already carries the kernel-authorship story. The skill on display in Phases 1-8 is systems
integration — keeping data resident on the GPU, choosing precision, sizing batches, finding the
real bottleneck — which is a different and more widely useful skill.

---

## Data

- **Video:** public-domain film, or a research corpus such as Condensed Movies / MovieNet. Pick
  one, state the exact version and size, and do not mix corpora inside a single figure.
- **Scene labels / tags:** the MovieLens tag genome is already familiar from AdFill and can
  provide a controlled vocabulary.
- **Ad catalogue:** constructed. Say so.

**Verify every dataset's own documentation by counting rows yourself.** Criteo's README was wrong
by roughly 10x during AdFill — the real figures were 435,810 conversions and 675 campaigns, not
what the docs claimed. Assume nothing until measured.

---

## Measurement discipline

- A `NUMBERS.md` ledger where **every figure traces to the results file and seed that produced
  it**, as AdFill does. A number without a file does not go in the README.
- Every row carries its hardware and, if the machine was shared, its load.
- Repeats, not single runs, for anything timed. Report the range.
- Re-runnable from a fresh clone.
- A bug log: what broke and what caught it. AdFill logged 11; that log is one of the most
  convincing things in the repo.

---

## Stack

Python · PyTorch · CUDA via NVDEC / TensorRT / cuDNN · ONNX · FAISS (GPU) · ffmpeg ·
optionally one hand-written CUDA kernel in Phase 7.

**Hardware:** develop on laptop or a cheap consumer GPU; rent an A100 only for the measured runs.
Checkpoint per shard so a preempted spot instance costs minutes, not hours. Keep embeddings, not
decoded frames — storage is the hidden cost.

---

## How it goes on a résumé

**Not until Phase 6 has a committed results file.** Same rule AdFill lives under. A half-built GPU
pipeline is worse than no project, because the first question asked will be "what did you
measure?"

Phases 7 and 8 are what make this a Netflix Ads project rather than a video-ML project. They map
onto seven of the ten open Ads SWE roles: segmentation and yield (four roles) and streaming
events (three). Phases 1-6 alone map onto none of them directly — they are the infrastructure
that earns the right to do 7 and 8.

Lead differently by audience:

- **Streaming / ads reqs** — lead with brand safety and contextual relevance, the inventory-loss
  figure, and the finding from Phase 4. The GPU work is how it was made affordable.
- **Systems / GPU reqs** (NVIDIA, Marvell, Tesla, cloud infra) — lead with the throughput table,
  the precision frontier and the cost figure. The ads framing is why the pipeline exists.

Bullets go through `VOICE.md` like everything else: plain first clause, at most three numbers,
tool names kept.
