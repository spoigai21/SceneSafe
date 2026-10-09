"""Phase 3: can the catalogue be searched, and at what memory cost?

Two questions, kept separate:
  A. What is one searchable entry ("unit")? one second, a fixed 10-second chunk, a detected scene (mean of its
     seconds), or seconds grouped by scene (search every second, return each scene once at its best second).
     Measured with exact search, so only the unit changes:
       - distinct@10: how many of the top 10 are separate moments (not within GAP_S of a higher-ranked result)
       - peak@10: for each result, the best per-second similarity inside it, averaged over the top 10.
         Per-second similarity is the finest signal we have; averaging seconds into a unit can bury a short peak.
       - ceiling: peak of the 10 best separate seconds (greedy), the most that 10 distinct moments can reach.
     There are no human relevance labels here; these are proxies and are reported as such.
  B. Which index? FAISS flat (exact), IVF and IVF-PQ over two vector sets (every second; detected scenes).
     recall@10 against flat on the same vectors, single-query latency p50/p99 (1 thread), batch throughput
     (all threads), build time and size. Text queries and image queries (random corpus seconds) reported apart.
Writes results/phase3/{units,index}_cpu-laptop_seed<seed>.json, index_frontier.png and, with --examples,
contact sheets of the top results in results/phase3/examples/.
"""

import argparse
import statistics
import time

import faiss
import numpy as np
import yaml

from scenesafe.common.results import REPO_ROOT, Timer, load_avg, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.ranking import collapse, distinct, greedy_distinct
from scenesafe.index.search import IndexSpec, build, default_nlist, set_nprobe, size_bytes
from scenesafe.index.units import Corpus, Units, fixed_chunks, load_corpus, per_second, scenes

PHASE = 3
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase3"
K = 10
GAP_S = 30.0               # results closer than this (same title) count as the same moment
OVERFETCH = 1000           # seconds fetched before collapsing them into scenes
THRESHOLDS = (0.75, 0.80, 0.85)
NPROBES = (1, 2, 4, 8, 16, 32, 64, 128)
PQ_M = (64, 32)


QUERY_FILE = REPO_ROOT / "configs" / "phase3_queries.yaml"
QUERY_CACHE = REPO_ROOT / "data" / "phase3" / "text_queries.npy"


# Runs in its own process: torch and faiss each bundle an OpenMP runtime, and the process aborts if both load.
EMBED_QUERIES = f"""
import numpy as np, yaml
from scenesafe.embed.clip import ClipEmbedder
queries = yaml.safe_load(open({str(QUERY_FILE)!r}).read())["queries"]
np.save({str(QUERY_CACHE)!r}, ClipEmbedder("mps").embed_text(queries).numpy().astype(np.float32))
"""


def text_queries() -> tuple[list[str], np.ndarray]:
    import subprocess
    import sys
    queries = yaml.safe_load(QUERY_FILE.read_text())["queries"]
    if not QUERY_CACHE.exists() or QUERY_CACHE.stat().st_mtime < QUERY_FILE.stat().st_mtime:
        QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-c", EMBED_QUERIES], check=True, cwd=REPO_ROOT)
    q = np.load(QUERY_CACHE)
    assert len(q) == len(queries)
    return queries, q


# ---------- A. units ----------

def unit_options(corpus: Corpus) -> dict[str, Units]:
    opts = {"second": per_second(corpus), "fixed10": fixed_chunks(corpus, 10)}
    for th in THRESHOLDS:
        u = scenes(corpus, th)
        opts[u.name] = u
    return opts


def evaluate_units(corpus: Corpus, opts: dict[str, Units], q: np.ndarray) -> dict:
    per_sec = corpus.emb @ q.T                                   # (N seconds, Q) similarity
    best = [greedy_distinct(per_sec[:, j], corpus.title, corpus.time, K, GAP_S) for j in range(len(q))]
    sec = opts["second"]
    flat_sec = build(IndexSpec("flat"), sec.vectors)
    rows = {"ceiling_peak@10": float(np.mean([per_sec[b, j].mean() for j, b in enumerate(best)]))}

    def score(u: Units, ids: np.ndarray, j: int) -> tuple[int, float, int]:
        keep = distinct(u.title[ids], u.start[ids], u.end[ids], GAP_S)
        peaks = np.maximum.reduceat(per_sec[:, j], u.first)[ids]
        found = sum(any(f <= r < f + n for f, n in zip(u.first[ids], u.count[ids])) for r in best[j])
        return int(keep.sum()), float(peaks.mean()), int(found)

    def timed(search) -> dict:
        faiss.omp_set_num_threads(1)
        for j in range(20):
            search(j)
        lat = []
        for j in range(len(q)):
            t0 = time.perf_counter()
            search(j)
            lat.append((time.perf_counter() - t0) * 1e3)
        return {"latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p99": float(np.percentile(lat, 99))}

    for name, u in opts.items():
        base = {"units": len(u), "length_s_median": float(np.median(u.count)),
                "length_s_p90": float(np.percentile(u.count, 90)), "length_s_max": int(u.count.max()),
                "flat_index_mb": len(u) * u.vectors.shape[1] * 4 / 1e6}
        # mean-pooled units: search the unit vectors directly
        index = build(IndexSpec("flat"), u.vectors)
        _, ids = index.search(q, K)
        d, p, f = zip(*(score(u, ids[j], j) for j in range(len(q))))
        rows[name] = {**base, "search": "unit vectors (mean of seconds)", "distinct@10": float(np.mean(d)),
                      "peak@10": float(np.mean(p)), "best_moments_found@10": float(np.mean(f)),
                      **timed(lambda j: index.search(q[j:j + 1], K))}
        if name.startswith("scene@"):
            # seconds grouped by scene: search every second, keep each scene once at its best second
            owner = u.of_second()
            scores, hits = flat_sec.search(q, OVERFETCH)
            results = [collapse(hits[j], scores[j], owner, K)[0] for j in range(len(q))]
            d, p, f = zip(*(score(u, results[j], j) for j in range(len(q))))

            def hybrid(j):
                s_, h_ = flat_sec.search(q[j:j + 1], OVERFETCH)
                collapse(h_[0], s_[0], owner, K)

            rows[f"second→{name}"] = {**base, "units": len(sec), "flat_index_mb": rows["second"]["flat_index_mb"],
                                      "search": f"every second, top {OVERFETCH}, collapsed to scenes", "scenes": len(u),
                                      "distinct@10": float(np.mean(d)), "peak@10": float(np.mean(p)),
                                      "best_moments_found@10": float(np.mean(f)),
                                      "queries_with_fewer_than_10": int(sum(len(r) < K for r in results)),
                                      **timed(hybrid)}
    for name, r in rows.items():
        if isinstance(r, dict):
            print(f"  {name:20s} units {r['units']:7d}  median {r['length_s_median']:5.1f}s  "
                  f"distinct@10 {r['distinct@10']:.2f}  peak@10 {r['peak@10']:.4f}  "
                  f"best moments found {r['best_moments_found@10']:.2f}/10  p50 {r['latency_ms_p50']:.2f} ms", flush=True)
    print(f"  ceiling peak@10 {rows['ceiling_peak@10']:.4f}")
    return rows


# ---------- B. indexes ----------

def latencies(index, q: np.ndarray) -> list[float]:
    out = []
    for i in range(len(q)):
        t0 = time.perf_counter()
        index.search(q[i:i + 1], K)
        out.append(time.perf_counter() - t0)
    return out


def recall(exact: np.ndarray, got: np.ndarray) -> float:
    return float(np.mean([len(set(e) & set(g)) / K for e, g in zip(exact, got)]))


def evaluate_indexes(name: str, vectors: np.ndarray, qsets: dict[str, np.ndarray], seed: int, threads: int) -> dict:
    nlist = default_nlist(len(vectors))
    specs = [IndexSpec("flat"), IndexSpec("ivf", nlist)] + [IndexSpec("ivfpq", nlist, m) for m in PQ_M]
    allq = np.concatenate(list(qsets.values()))
    exact = {}
    out = {"vectors": len(vectors), "nlist": nlist, "indexes": {}}
    for spec in specs:
        faiss.omp_set_num_threads(threads)
        with Timer() as tb:
            index = build(spec, vectors, seed)
        row = {"build_s": tb.seconds, "size_mb": size_bytes(index) / 1e6, "settings": {}}
        for nprobe in (NPROBES if spec.kind != "flat" else (None,)):
            if nprobe is not None and nprobe > nlist:
                continue
            set_nprobe(index, nprobe or 0)
            faiss.omp_set_num_threads(threads)
            res = {}
            for qname, q in qsets.items():
                _, ids = index.search(q, K)
                if spec.kind == "flat":
                    exact[qname] = ids
                res[f"recall@10_{qname}"] = recall(exact[qname], ids)
            with Timer() as tt:
                index.search(allq, K)
            res["batch_qps"] = len(allq) / tt.seconds
            faiss.omp_set_num_threads(1)
            latencies(index, allq[:50])                              # warm-up
            lat = np.array(latencies(index, allq)) * 1e3
            res.update({"latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p99": float(np.percentile(lat, 99))})
            row["settings"][str(nprobe or "exact")] = res
            print(f"  {name:10s} {spec.name:16s} nprobe {str(nprobe or '-'):>4s}  size {row['size_mb']:7.1f} MB  "
                  f"R@10 text {res['recall@10_text']:.3f} image {res['recall@10_image']:.3f}  "
                  f"p50 {res['latency_ms_p50']:.3f} ms  p99 {res['latency_ms_p99']:.3f} ms", flush=True)
        out["indexes"][spec.name] = row
    return out


def flat_scaling(vectors: np.ndarray, q: np.ndarray) -> list[dict]:
    """Exact-search latency (1 thread) on growing slices of the corpus: measured, for a later projection."""
    faiss.omp_set_num_threads(1)
    rows = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        n = int(len(vectors) * frac)
        index = build(IndexSpec("flat"), vectors[:n])
        latencies(index, q[:20])
        lat = np.array(latencies(index, q)) * 1e3
        rows.append({"vectors": n, "latency_ms_p50": float(np.median(lat)), "latency_ms_p99": float(np.percentile(lat, 99))})
        print(f"  flat on {n:7d} vectors: p50 {rows[-1]['latency_ms_p50']:.3f} ms")
    return rows


def plot(index_results: dict, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(index_results), figsize=(6 * len(index_results), 4.5), squeeze=False)
    colors = {"flat": "black", "ivf": "#1f77b4", "pq64": "#d62728", "pq32": "#ff7f0e"}
    for ax, (vname, res) in zip(axes[0], index_results.items()):
        for iname, row in res["indexes"].items():
            key = "flat" if iname == "flat" else iname.split("-")[-1] if "pq" in iname else "ivf"
            pts = [(s["latency_ms_p50"], s["recall@10_text"]) for s in row["settings"].values()]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=4, color=colors[key], label=f"{iname} ({row['size_mb']:.0f} MB)")
        ax.set_xscale("log")
        ax.set_xlabel("single-query latency p50, ms (1 thread, log scale)")
        ax.set_ylabel("recall@10 vs exact (100 text queries)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{vname}: {res['vectors']:,} vectors (points = nprobe 1…128)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


# ---------- examples ----------

def contact_sheets(corpus: Corpus, opts: dict[str, Units], queries: list[str], q: np.ndarray, picks: list[str], th: str):
    """For a few queries: one row per unit option, the top 5 results, each shown at its best second."""
    from PIL import Image, ImageDraw
    from scenesafe.corpus import video_path
    from scenesafe.decode.decoder import iter_frames

    def frame_at(row: int) -> Image.Image:
        path = video_path(corpus.ids[corpus.title[row]])
        for f in iter_frames(path, start_s=float(corpus.time[row]), duration_s=2.0):
            return f.frame.to_image().resize((192, 144))
        return Image.new("RGB", (192, 144))

    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    sec = opts["second"]
    flat_sec = build(IndexSpec("flat"), sec.vectors)
    for text in picks:
        j = queries.index(text)
        per_sec = corpus.emb @ q[j]
        lines = []
        for name in ("second", "fixed10", th, f"second→{th}"):
            if name.startswith("second→"):
                u = opts[th]
                s, h = flat_sec.search(q[j:j + 1], OVERFETCH)
                ids, _, _ = collapse(h[0], s[0], u.of_second(), 5)
            else:
                u = opts[name]
                _, ids = build(IndexSpec("flat"), u.vectors).search(q[j:j + 1], 5)
                ids = ids[0]
            best = [int(u.first[i] + np.argmax(per_sec[u.first[i]:u.first[i] + u.count[i]])) for i in ids]
            lines.append((name, best))
        sheet = Image.new("RGB", (150 + 5 * 196, 30 + len(lines) * 170), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'query: "{text}"  (top 5, each shown at its best second)', fill="black")
        for r, (name, best) in enumerate(lines):
            y = 30 + r * 170
            draw.text((8, y + 60), name.replace("→", " -> "), fill="black")
            for c, row in enumerate(best):
                sheet.paste(frame_at(row), (150 + c * 196, y))
                t = int(corpus.time[row])
                draw.text((150 + c * 196, y + 146), f"{corpus.ids[corpus.title[row]][:18]} {t // 60}:{t % 60:02d}",
                          fill="black")
        out = OUT / "examples" / f"{text.replace(' ', '_')}.jpg"
        sheet.save(out, quality=80)
        print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-queries", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=faiss.omp_get_max_threads())
    ap.add_argument("--index-units", nargs="+", default=["second", "scene@0.75"])
    ap.add_argument("--examples", nargs="*", default=None, help="queries to draw contact sheets for")
    ap.add_argument("--examples-scene", default="scene@0.75")
    ap.add_argument("--part", choices=["all", "units", "index"], default="all")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    corpus = load_corpus()
    queries, q_text = text_queries()
    q_image = corpus.emb[np.sort(rng.choice(len(corpus.emb), args.image_queries, replace=False))]
    print(f"corpus: {len(corpus.emb)} seconds from {len(corpus.ids)} titles; {len(queries)} text queries, "
          f"{len(q_image)} image queries")
    opts = unit_options(corpus)

    if args.examples is not None:
        contact_sheets(corpus, opts, queries, q_text, args.examples, args.examples_scene)
        return

    common = {"model": DEFAULT_MODEL, "pretrained": DEFAULT_PRETRAINED, "faiss_version": faiss.__version__,
              "text_queries": len(queries), "query_file": "configs/phase3_queries.yaml",
              "corpus_seconds": int(len(corpus.emb)), "titles": len(corpus.ids)}

    if args.part in ("all", "units"):
        run_units(args, corpus, opts, q_text, common)
    if args.part in ("all", "index"):
        run_index(args, opts, q_text, q_image, common)


def run_units(args, corpus, opts, q_text, common):
    print("A. units (exact search, text queries):")
    units = evaluate_units(corpus, opts, q_text)
    print("  ->", write_result(PHASE, "units", LANE, args.seed,
                               {**common, "k": K, "gap_s": GAP_S, "overfetch": OVERFETCH, "thresholds": THRESHOLDS,
                                "scene_min_len_s": 3}, {"units": units, "load_avg": load_avg()}))


def run_index(args, opts, q_text, q_image, common):
    print("B. indexes:")
    load_before = load_avg()
    index_results = {name: evaluate_indexes(name, opts[name].vectors, {"text": q_text, "image": q_image},
                                            args.seed, args.threads) for name in args.index_units}
    print("C. exact search vs corpus size:")
    scaling = flat_scaling(opts["second"].vectors, np.concatenate([q_text, q_image[:100]]))
    load_after = load_avg()
    print("  ->", write_result(PHASE, "index", LANE, args.seed,
                               {**common, "k": K, "image_queries": len(q_image), "threads_batch": args.threads,
                                "threads_latency": 1, "nprobes": NPROBES, "pq_m": PQ_M, "pq_bits": 8},
                               {"vector_sets": index_results, "flat_scaling": scaling,
                                "load_avg_before": load_before, "load_avg_after": load_after}))
    plot(index_results, OUT / "index_frontier.png")
    print("  ->", OUT / "index_frontier.png")


if __name__ == "__main__":
    main()
