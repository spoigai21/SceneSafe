# SceneSafe — build spec

SceneSafe watches a catalogue of video, understands each scene, and decides which ads are **safe**
and **relevant** to play next to it. It then turns those decisions into priced ad inventory and
counts the resulting ad events correctly on a streaming pipeline. Every stage is measured, and
the whole pipeline has a stated cost per hour of video analysed.

This file is the implementation contract: what to build, in what order, what to measure, and
when each phase is done.

---

## 1. Why it exists

Ad targeting is moving away from *who is watching* (third-party cookies are going away) and back
to *what is on screen*. A video owner is the only party that can do this properly for its own
inventory. Two commercial jobs:

- **Relevance** — a cooking scene is worth more to a food brand than a car chase is.
- **Brand safety** — a car ad straight after a crash scene loses the advertiser.

Both require actually watching the content at scale. That is a **throughput problem before it is
a machine-learning problem**, and that is the project.

---

## 2. Honest limits (copy these into the README, not a footnote)

1. **The video is public-domain / research video**, not any streaming service's catalogue.
   Nothing here is a claim about how any company's ads perform.
2. **The ad catalogue and ad demand are simulated.** There is no public corpus of video ad
   creatives with outcomes. Say so wherever a result depends on them.
3. **Every number names its hardware.** Laptop (CPU) and rented-GPU numbers are separate lanes
   and are never blended into one figure.
4. **Quality is measured against exact fp32 search on the same embeddings**, never an external
   leaderboard. The question is "what did the fast path cost", not "is this state of the art".

---

## 3. Ground rules for every phase

| Rule | What it means in practice |
|---|---|
| Results files | Every measured run writes a JSON file to `results/phase<N>/`. It records the git commit, hardware, machine load, seed, config and raw timings. |
| Ledger | `NUMBERS.md` lists every figure used anywhere, with the results file and seed that produced it. **No file means the number doesn't go in the README.** |
| Two lanes | Each results row is tagged `lane: cpu-laptop` or `lane: gpu-<model>`. The lanes are never averaged together. |
| Repeats | Anything timed runs at least 3 times. Report the median and the range. |
| Baselines first | Each phase builds its baseline before its fast path. |
| Bug log | `BUGS.md`: what broke, how it was caught, what fixed it. |
| Count data yourself | Never trust a dataset's README for row counts or sizes. Count them, and record the counts in `NUMBERS.md`. |
| Losses count | If a phase loses to its baseline, that is the result. Write it up as one. |
| Reproducible | `make phase<N>` (or the equivalent script) re-runs a phase from a fresh clone. |

---

## 4. Repo layout

```
scenesafe/              Python package (phases 1-7)
  decode/               ffmpeg + NVDEC backends, sampling policies
  embed/                model export, PyTorch / ONNX Runtime / TensorRT runners
  index/                exact / IVF / IVF-PQ index build + query
  match/                ad brief -> scenes retrieval, baselines
  safety/               rule engine, adjacency checks, exclusion lists
  segments/             segment builder, simulated demand, fill/yield
  cost/                 $/1,000 h calculator
  common/               results-file writer, timing, hardware detection, seeds
events/                 Phase 8 streaming pipeline (Java, Gradle)
kernels/                Phase 9 optional CUDA kernel
scripts/                one entrypoint per phase (run_phase1.py, ...)
configs/                YAML configs per phase and run
results/phase<N>/       committed JSON results (small); large artefacts go in data/ (ignored)
tests/
README.md  PREDICTION.md  NUMBERS.md  BUGS.md  EVAL-PROTOCOL.md
```

The `data/` and `frames/` directories and the `*.npy` / `*.parquet` files are gitignored. Keep the
embeddings and throw the decoded frames away, because storage is the hidden cost.

---

## 5. Phases

Do them in order. Phases 7 and 8 matter most, and Phases 1–6 are the infrastructure they need.

### Phase 0 — Predict before renting a GPU

**Build:** a script that runs **10 minutes of video on the laptop** through decode and then the
vision model. It times each stage on its own.

**Measure:**
- frames/sec for decode
- frames/sec for the vision model
- the ratio between them, which shows whether the pipeline is **decode-bound or
  inference-bound** and so whether hardware decode is worth anything

**Deliverable:** commit `PREDICTION.md` **before renting anything**. It holds the measured laptop
rates, the predicted GPU speedup, and the predicted cost of the full run. Never edit it
afterwards. Later phases compare against it, and a wrong prediction stays in the repo.

**Done when:** `PREDICTION.md` is committed.

---

### Phase 1 — Decode and sample

**Question:** how fast can frames be got off disk and onto the GPU, and how many are needed?

**Build:**
- a decoder interface with backends:
  - `cpu` (ffmpeg)
  - `videotoolbox` (the laptop's hardware decoder)
  - `nvdec` (NVIDIA hardware decoder, measured only in the GPU session)
- a parallel mode that decodes several titles at once, one process per title
- sampling policies:
  - `all` (every frame)
  - `uniform@k` for several values of k, e.g. 0.25, 0.5, 1 and 2 fps
  - `keyframes` (decode only the self-contained frames and skip the rest without decoding
    them; about one every 1.5 s in the Phase 0 film)
  - `scene-change` (sample when the content changes, e.g. with PySceneDetect or a histogram
    difference)

A two-hour film is about 172,000 frames, mostly near-duplicates. **The sampling policy matters
more than the decoder** for total work.

**Measure:**
- frames/sec per backend, on the same clips and the same machine
- total wall-clock for the fixed corpus under each backend
- a curve of sampling policy against downstream retrieval quality: recall@k against the `all`
  frames index, using the Phase 2 fp32 embeddings

**Baseline:** decode every frame on the CPU.

**Done when:** there's a decode-throughput table per backend, and a sampling curve showing where
quality starts to drop.

---

### Phase 2 — Embed

**Question:** what does fast inference cost in quality?

**Build:**
- one vision model (pick a CLIP-style image/text model so Phase 4 can embed the text of ad
  briefs into the same space)
- export it once and run it on the laptop under:
  - PyTorch eager on CPU and MPS (Apple GPU), at fp32 and fp16
  - ONNX Runtime at fp32 and int8 (int8 calibrated on a held-out set)
- in the GPU session only: TensorRT at fp32, fp16 and int8

**Measure:**
- latency and throughput per runtime × precision × batch size (e.g. 1, 8, 32, 128)
- **recall@k against exact fp32 top-k on the same frames**, for every precision
- the bottleneck at each batch size: host→device transfer or compute

**Baseline:** PyTorch eager, fp32.

**Trap:** int8 that is faster but quietly worse. **Don't publish an int8 row without its quality
number.**

**Done when:** there's a speed/quality frontier where every point names its hardware, plus a
written choice of precision with the reason.

---

### Phase 3 — Index

**Question:** can the catalogue be searched, and at what memory cost?

**Build:** FAISS indexes over the scene embeddings: exact (flat), IVF and IVF-PQ, on the CPU. At
this corpus size a GPU build isn't needed.

**Measure:** recall of the exact top-k, query latency (p50/p99) and index size in MB for each
configuration.

**Done when:** there's a memory/recall/latency table, and the configuration used from here on is
named and justified.

---

### Phase 4 — Contextual matching

**Question:** is the signal real, or is this an expensive way to do nothing?

**Build:**
- a simulated ad catalogue in `configs/ads.yaml`. Each ad has a category, a short text brief
  and constraints.
- retrieval: embed the brief, then query the index for matching scenes
- two baselines:
  - **random scenes** (the floor)
  - **genre-only matching** (the honest competitor, because genre metadata is free and already
    exists)

**Before generating any numbers:** commit `EVAL-PROTOCOL.md`. It defines the held-out titles, how
the relevance labels are made (e.g. hand-labelled scene/category pairs, with the sample size
stated), and the metrics, e.g. precision@k and nDCG@k.

**Measure:** scene matching against both baselines, on the held-out titles.

**Done when:** there's a quantified comparison against both baselines. If scene understanding
does **not** beat genre metadata, report that plainly. It's a real finding.

---

### Phase 5 — Brand safety

**Question:** what does safety cost in sellable inventory?

**Build:** a rule engine over the detected scene content, with three kinds of rule:
- **category avoidance** (e.g. violence, accidents, alcohol)
- **adjacency constraints** (e.g. no car ad straight after a crash scene)
- **per-advertiser exclusion lists**

Define the rule sets as data in `configs/safety/*.yaml`, e.g. `lenient`, `standard` and `strict`.

**Measure:**
- **the fraction of inventory each rule set removes**, reported under a slack book (supply far
  above demand) and a tight one. The same removal is cheap on the first and costly on the second.
- **the false-positive rate**: a random sample of excluded scenes, hand-checked, with the sample
  size and a confidence interval stated

**Done when:** there's an inventory-loss figure per rule set, and a false-positive rate taken
from a hand-checked sample.

---

### Phase 6 — Cost

**Question:** would anyone actually run this?

**Build:** nothing new. Combine the measured throughput with real pricing, and record the price
source and the date.

**Measure:** for each lane:
- **machine-hours per 1,000 hours of video analysed**, on the laptop and on the rented GPU
- **dollars per 1,000 hours of video analysed**, using the rented GPU's price and the price of a
  comparable cloud CPU instance

State the hardware, sampling policy and precision behind each figure, plus the same figures for
the naive Phase 1 baseline. Compare them with `PREDICTION.md`.

**Done when:** there's one figure someone could put in a planning document, with its derivation
in `NUMBERS.md`.

---

### Phase 7 — Scene signals become priced inventory

**Question:** does scene-level understanding change what the inventory is worth?

**Build:**
- **ad breaks**: positions in each title where an ad can play, e.g. at scene boundaries
- **segments**: breaks grouped by **safety class** (Phase 5) × **contextual category** (Phase 4).
  One bucket is **restricted**: breaks that are safe for almost no advertiser.
- a simple **valuation per segment**. This is not a pricing model. It's a defensible ordering of
  the segments plus the evidence for it.
- **simulated demand**: advertisers with budgets, target categories and safety requirements,
  seeded and reproducible

**Measure:**
- **fill rate per segment**: the share of each segment's breaks that can actually be sold under
  the Phase 5 exclusions
- **yield**: the segmented book against **one flat, undifferentiated pool priced flat**, on the
  same simulated demand
- **the restricted share**: how much inventory ends up in the restricted bucket. This decides
  whether segmentation helps or only shrinks the sellable pool.

**Scope limit:** produce the segments and their numbers, then stop. Don't build an auction or an
allocator here.

**Done when:** there's a fill-rate and yield table per segment, with the restricted share named.

---

### Phase 8 — The event spine

**Question:** when ads play against these segments, can the events be counted correctly?

**Build (in `events/`, Java):**
- an **event generator** driven by the Phase 7 segments. It emits `impression`, `completion` and
  `abandonment` events, each with an `event_id`, `break_id`, `scene_id`, `segment_id` and event
  time.
- **damage injection**, configurable per run:
  - duplicates (5–20%)
  - out-of-order arrival
  - late events past the window
  - dropped events
- **Kafka** for ingestion
- a **stream processor** (Flink, or Kafka Streams) for deduplication by `event_id` and windowed
  aggregation per segment
- a **queryable rollup store**, e.g. Postgres, ClickHouse or Druid
- an **audit job** that compares the rollups with the generator's ground-truth log and **fails
  loudly** (non-zero exit, alert line) when completeness drops below a threshold

Java is a deliberate choice here. It's the native language of the Kafka and Flink ecosystems.
Run everything locally through `docker compose`.

**Measure:**
- **accuracy**: counted impressions against ground truth, at each duplication level
- **completeness**: the fraction of emitted events that reach the rollup, and where the missing
  ones went (dropped, late, filtered)
- **timeliness**: the lag from emit to queryable, at several input rates
- **sustained events/sec on one machine**, with the hardware named and the bottleneck identified

**Baselines:**
- naive counting with no deduplication
- a batch job doing the same aggregation

Together these make the cost of streaming visible rather than assumed.

**Done when:** there's an accuracy/completeness/timeliness table under injected damage, a
throughput figure with its bottleneck, and the failing audit check demonstrated.

---

### Phase 9 — Optional: fused preprocessing kernel

Only after Phases 1–8 are finished and measured, and only if a second GPU session is worth it.
Otherwise skip it. The project is complete without it.

Decode outputs one colour format, while the model wants a different size, layout and
normalisation. Done naively that's three passes with memory traffic between them. Fused into
one CUDA kernel it's a single pass. NVIDIA built DALI for this reason.

**Measure:** the fused kernel against the naive path on the same frames. The output must match
bit-for-bit or within a stated tolerance.

---

## 6. Data

- **Video:** about **50 hours (roughly 35 films)** of public-domain feature films from the
  Internet Archive, which have genre tags for the Phase 4 baseline. Record every item ID, file and
  byte size in a manifest, and never mix corpora inside one figure. Add more films only if
  Phase 4 needs them. Delete each video once its embeddings are saved.
- **Scene tags:** the MovieLens tag genome can supply a controlled vocabulary for categories.
- **Genre metadata** (for the Phase 4 baseline): from the corpus's own metadata. Record the
  source.
- **Ad catalogue and demand:** constructed and seeded. Say so.

---

## 7. Stack and hardware

**Phases 1–7:** Python · PyTorch (CPU / MPS) · ONNX Runtime · PyAV/ffmpeg with VideoToolbox ·
FAISS (CPU). Only in the GPU session: TensorRT · NVDEC.
**Phase 8:** Java · Gradle · Kafka · Flink or Kafka Streams · a rollup store · docker compose.
**Phase 9:** CUDA C++.

**Hardware: laptop first, one short GPU session.**

- **The laptop runs everything:** all development, every phase, and the full corpus.
- **One rented NVIDIA GPU session, budget ≤ 3 hours** (a cheap card such as an L4 or A10; check
  the price when renting). It adds the NVIDIA lane:
  - NVDEC decode vs CPU decode (Phase 1)
  - TensorRT fp32 / fp16 / int8 (Phase 2)
  - the rented-GPU cost figure (Phase 6)
- **No debugging on rented time.** The session runs one script that has already been tested on
  the laptop, then copies `results/` back.
- If one session isn't enough, ask before renting more.

---

## 8. Definition of done (whole project)

- [ ] Phases 0–8 meet their "done when" lines
- [ ] `NUMBERS.md` traces every figure to a results file and seed
- [ ] `BUGS.md` logs what broke and what caught it
- [ ] the README states the honest limits (section 2) up front
- [ ] everything re-runs from a fresh clone
