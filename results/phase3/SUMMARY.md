# Phase 3 — index: results

**Question:** can the catalogue be searched, and at what memory cost?

Two questions, kept apart:
- **A. What is one searchable entry?** One second, a fixed 10-second chunk, or a detected scene.
- **B. Which index?** FAISS exact (flat), IVF and IVF-PQ.

## Setup

**Data:** the full corpus embeddings from Phase 2: 179,545 seconds from 35 titles (CLIP ViT-B/32,
OpenCLIP `laion2b_s34b_b79k`, unit-normalised).

**Queries:**
- **100 text queries** (`configs/phase3_queries.yaml`): the 20 Phase 1 queries plus 80 more, written
  before any Phase 3 results.
- **1,000 image queries:** random corpus seconds, seed 0.

**Machine:** Apple M5 laptop, CPU only, FAISS 1.15.1. Load average 2.2–3.0 on 10 cores.

**Timing:**
- single-query latency on 1 thread, after a warm-up
- batch throughput on all 10 threads

## A. What counts as a scene

**Scene detection** uses the existing fingerprints; the video is not touched. A new scene starts when
a second's fingerprint falls below a cosine threshold to the running mean of the current scene, with a
minimum scene length of 3 s. Three thresholds were tried: 0.75, 0.80 and 0.85.

**Two ways to search scenes:**
- **pooled:** each scene is one fingerprint, the mean of its seconds
- **second → scene:** search every second, then return each scene once, at its best second

**Measures** (exact search, so only the unit changes; 100 text queries, top 10):
- **distinct@10:** how many of the 10 results are separate moments, meaning not within 30 s of a
  higher-ranked result from the same film
- **best moments found:** the 10 best separate seconds for the query (greedy, 30 s apart) are the
  target. This counts how many of them fall inside a returned result.
- **peak@10:** for each result, the best per-second similarity inside it, averaged over the 10 results.
  The ceiling (the 10 best separate seconds) is 0.2895.

| Unit | Entries | Length median / p90 / max (s) | distinct@10 | best moments found (of 10) | peak@10 | p50 / p99 latency |
|---|---|---|---|---|---|---|
| second | 179,545 | 1 / 1 / 1 | 5.67 | 5.68 | 0.295 | 4.73 / 5.59 ms |
| fixed 10 s, pooled | 17,972 | 10 / 10 / 10 | 7.78 | 2.89 | 0.276 | 0.49 / 1.55 ms |
| scene@0.75, pooled | 10,241 | 7 / 41 / 634 | 8.73 | 3.30 | 0.275 | 0.25 / 0.27 ms |
| scene@0.80, pooled | 16,143 | 6 / 24 / 334 | 8.57 | 4.15 | 0.279 | 0.40 / 0.45 ms |
| scene@0.85, pooled | 22,738 | 5 / 16 / 196 | 8.34 | 4.43 | 0.281 | 0.56 / 0.60 ms |
| **second → scene@0.75** | 179,545 (10,241 scenes) | 7 / 41 / 634 | 8.66 | **9.26** | 0.290 | 4.94 / 5.22 ms |
| second → scene@0.80 | 179,545 (16,143 scenes) | 6 / 24 / 334 | 8.56 | 8.93 | 0.291 | 4.93 / 5.14 ms |
| second → scene@0.85 | 179,545 (22,738 scenes) | 5 / 16 / 196 | 8.44 | 8.69 | 0.291 | 4.94 / 5.26 ms |

Source: `units_cpu-laptop_seed0.json`. Examples: `examples/*.jpg` (top 5 for 4 queries, every option).

**Findings:**
1. **Searching single seconds returns duplicates.** Only 5.7 of the top 10 are separate moments; the
   rest are neighbouring seconds of the same shot. The `people_dancing.jpg` sheet shows this:
   `AsYouLikeIt1936` at 89:58 and 89:59.
2. **Pooling seconds into chunks or scenes removes duplicates but blurs short moments.** Pooled units
   found only **2.9–4.4 of the 10 best moments**. A strong 3-second moment inside a long scene gets
   averaged away.
3. **Second → scene gets both.** It finds **9.3 of the 10 best moments** with 8.7 separate results, at
   the same speed as searching seconds (4.9 vs 4.7 ms). The extra fetch of 1,000 hits and the grouping
   step cost about 0.2 ms. Every query returned a full 10 scenes.
4. **Threshold 0.75 scored best** on both distinct@10 and best moments found. The differences between
   thresholds are small.

**Caveats on A:**
- **"Best moments" is measured against per-second similarity**, so it favours methods that keep
  per-second detail. There are no human relevance labels.
- **By eye**, pooled scenes also returned relevant results, sometimes sustained ones that the
  per-second methods miss (e.g. `royal_wedding` for "people dancing"). So pooled search isn't wrong;
  it answers "which scene is about this overall?" rather than "where is the strongest moment?" Phase 4
  can test whether that matters for ad matching.
- **Detected "scenes" are shorter than story scenes** (median 7 s). They're closer to shots or short
  sequences. One long fight can still split into several scenes: `two_people_fighting.jpg` returns
  `Grave_of_the_Vampire` at 83:55 and at 84:43.
- **Weak hits exist:** a man in a doorway for "two people fighting", a dark frame for "people
  dancing", a pipe for "a man smoking a cigarette". These are CLIP limits, not index limits.

## B. Which index

Recall@10 is measured against exact search on the same vectors.

**Index types:**
- **IVF:** the vectors are split into `nlist` groups by k-means. A search checks the `nprobe` nearest
  groups.
- **IVF-PQ:** additionally compresses each vector to 64 or 32 bytes.

**Every second** (179,545 vectors, nlist 2,048):

| Index | Size | Build | Setting | Recall@10 text / image | p50 / p99 latency (1 thread) | Batch throughput (10 threads) |
|---|---|---|---|---|---|---|
| **flat (exact)** | 368 MB | 0.03 s | — | **1.000 / 1.000** | 4.72 / 5.09 ms | 7,451 queries/s |
| IVF | 373 MB | 4.1 s | nprobe 8 | 0.683 / 0.970 | 0.081 / 0.097 ms | 48,160 queries/s |
| IVF | | | nprobe 32 | 0.856 / 0.995 | 0.154 / 0.195 ms | 16,870 queries/s |
| IVF | | | nprobe 64 | 0.915 / 0.999 | 0.251 / 0.292 ms | 9,466 queries/s |
| IVF | | | nprobe 128 | 0.961 / 1.000 | 0.45 ms | 5,030 queries/s |
| IVF-PQ, 64 B/vector | 17.7 MB | 7.6 s | nprobe 64 | 0.446 / 0.778 | 0.182 ms | 28,116 queries/s |
| IVF-PQ, 32 B/vector | 11.9 MB | 12.3 s | nprobe 64 | 0.320 / 0.666 | 0.114 ms | 46,151 queries/s |

**Pooled scenes@0.75** (10,241 vectors, nlist 256): flat 21 MB, 0.25 ms, recall 1.000. IVF at nprobe 64
reaches 0.964 text recall at 0.097 ms. IVF-PQ tops out at 0.52 text recall.

Source: `index_cpu-laptop_seed0.json`, all nprobe values 1–128. Chart: `index_frontier.png`.

**Exact search latency against corpus size** (1 thread, p50):

| Vectors | Latency |
|---|---|
| 17,954 | 0.44 ms |
| 44,886 | 1.15 ms |
| 89,772 | 2.33 ms |
| 179,545 | 4.72 ms |

It grows linearly: about 26 ns per vector.

**Findings:**
5. **Exact search is fast enough at this size: 4.7 ms per query** on one CPU core, over all 50 hours.
   Nothing needs approximating yet.
6. **Measuring IVF with image queries overstates its recall for text search.** At nprobe 8, IVF finds
   97% of the exact results for image queries but only 68% for text queries. The groups are built from
   image fingerprints, and CLIP's text fingerprints sit in a different region of the shared space, so
   "the nearest groups" is a worse guide for a text query. All the ad and safety searches in this
   project are text queries. **Benchmarks that use corpus vectors as queries would mislead here.**
7. **PQ compression costs too much accuracy here.** It is 20–30× smaller, but text recall stops at
   0.32–0.45 however many groups are searched; the compressed codes themselves are the limit.
   Re-ranking the shortlist with the full vectors would recover some of it. That wasn't tested.
8. **IVF doesn't save memory.** It stores all the full vectors plus the group lists, so it is 1.5%
   *bigger* than flat. It only buys speed.

## Decision

**Index: exact (flat) search over every second, with results grouped into detected scenes at
threshold 0.75** (`second → scene@0.75`). It is:
- 368 MB of memory
- 4.9 ms per query on 1 CPU thread
- finds 9.3 of the 10 best moments, with 8.7 separate results

Approximation isn't needed at 50 hours. **If the catalogue grows,** IVF at nprobe 64 is the measured
next step: 19× faster for 0.915 text recall. PQ is rejected on accuracy.

Phase 5 scores safety on every second, not on scenes, so short dangerous moments can't be averaged away.

## Projection, not measured

Exact search grows linearly. 1,000 video-hours at 1 fps would be 3.6 M vectors, which works out to
about **95 ms per query and 7.4 GB of memory**. That is the point where IVF (or a GPU) starts to pay.
This is extrapolated from the four sizes above, not measured at 3.6 M.

## Limits

- No human relevance labels. Part A's measures assume per-second CLIP similarity is the truth, and the
  example sheets are a small eyeball check (4 queries × top 5), not a measurement.
- One seed, one machine, CPU only.
- IVF-PQ was tried without re-ranking. HNSW (another common index) was out of scope.
- The scene detector has only one setting per threshold (minimum length 3 s), and it was not checked
  against hand-marked scene boundaries.
