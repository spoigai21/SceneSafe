"""Phase 4f: does adding what is SAID in a scene (Whisper transcripts) find scenes that suit the ad more often than
frames alone? Rules: EVAL-PROTOCOL.md, "Phase 4f amendment".

  transcribe  Whisper on the 14 Phase 4d films (data/transcripts/). No results.
  test        three systems on the Phase 4d films and scenes, top 10 per ad: frames (the Phase 4d ViT-B/32 picks,
              reused), dialogue (ad brief + theme vs the scene's speech), both (equal-weight z-scores). Every picked
              pair goes into one shuffled pool; every card shows 4 frames and the scene's dialogue.
  score       frames vs dialogue vs both; writes results/phase4f/match_dialogue_cpu-laptop_seed0.json and a chart.
"""

import argparse
import json
import textwrap

import numpy as np

from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import dialogue as dl
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d, label_guide, themes
from scenesafe.match.display import FRAMES_PER_SCENE, THUMB, frame_strip
from scenesafe.match.metrics import agreement, bootstrap, graded_ndcg_at_k, precision_at_k
from scenesafe.match.transcribe import WHISPER_MODEL, scene_text, transcribe

PHASE = 4
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase4f"
P4D = REPO_ROOT / "results" / "phase4d"
DATA = REPO_ROOT / "data" / "phase4f"
FRAMES = REPO_ROOT / "data" / "phase4d" / "frames"
SCENE_THRESHOLD = 0.75
K = 10
DIALOGUE_SHOWN = 600         # characters of dialogue shown on a card (the full text is in pool.json)
AUTHOR_SAMPLE, AUTHOR_SEED = 100, 5
SYSTEMS = ("frames", "dialogue", "both")


def films() -> list[str]:
    return json.loads((P4D / "pool.json").read_text())["films"]


def describe(corpus, units, u: int) -> dict:
    title = corpus.ids[units.title[u]]
    return {"scene": f"{title}@{units.start[u]:.0f}", "title": title,
            "start_s": round(float(units.start[u]), 2), "end_s": round(float(units.end[u]), 2)}


def ad_texts(catalogue) -> list[str]:
    th = themes()
    return [f"{a.brief}. {th[a.id]['theme']}" for a in catalogue]


def test():
    catalogue = ads_4d()
    ids = films()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    tr = {t: transcribe(t)["segments"] for t in ids}
    text = [scene_text(tr[corpus.ids[units.title[u]]], units.start[u], units.end[u]) for u in range(len(units))]

    chunk_text, chunk_scene = [], []
    for u, t in enumerate(text):
        for c in dl.chunks(t):
            chunk_text.append(c)
            chunk_scene.append(u)
    chunk_vecs = dl.embed_text(chunk_text, DATA / "chunks.npy")
    ad_vecs = dl.embed_text(ad_texts(catalogue), DATA / "ads.npy")
    dia = dl.scene_dialogue_scores(chunk_vecs, np.array(chunk_scene), len(units), ad_vecs)

    q = np.load(REPO_ROOT / "data" / "phase4d" / "briefs_b32.npy")      # the Phase 4d frame queries, unchanged
    frame = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    both = dl.combined(frame, dia)

    d4 = json.loads((P4D / "picks.json").read_text())["picks"]["b32"]
    picks = {"frames": d4, "dialogue": {}, "both": {}}
    for i, ad in enumerate(catalogue):
        picks["dialogue"][ad.id] = [{**describe(corpus, units, u), "score": float(dia[u, i])}
                                    for u in dl.top_k(dia[:, i], K)]
        picks["both"][ad.id] = [{**describe(corpus, units, u), "score": float(both[u, i])}
                                for u in dl.top_k(both[:, i], K)]
        assert [s["scene"] for s in picks["frames"][ad.id]] == \
               [describe(corpus, units, u)["scene"] for u in dl.top_k(frame[:, i], K)], "frames system drifted from 4d"

    scene_index = {describe(corpus, units, u)["scene"]: u for u in range(len(units))}
    pairs = {}
    for system in SYSTEMS:
        for ad_id, lst in picks[system].items():
            for s in lst:
                pairs.setdefault((ad_id, s["scene"]), {k: s[k] for k in ("scene", "title", "start_s", "end_s")})
    keys = list(pairs)
    by_ad = {a.id: a for a in catalogue}
    order = np.random.default_rng(0).permutation(len(keys))
    pool = [{"pair": f"F{n:04d}", "ad": keys[j][0], "category": by_ad[keys[j][0]].category,
             "brief": by_ad[keys[j][0]].brief, **pairs[keys[j]], "dialogue": text[scene_index[keys[j][1]]]}
            for n, j in enumerate(order, 1)]
    for key, p in {p["scene"]: p for p in pool}.items():
        if not (FRAMES / f"{key}.jpg").exists():
            frame_strip(p["title"], p["start_s"], p["end_s"], FRAMES / f"{key}.jpg")
    sheets(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    speech = np.array([bool(t) for t in text])
    meta = {"films": ids, "ads": len(catalogue), "scenes": int(len(units)), "scenes_with_speech": int(speech.sum()),
            "whisper_model": WHISPER_MODEL, "text_model": dl.TEXT_MODEL, "frame_model": f"{DEFAULT_MODEL} {DEFAULT_PRETRAINED}",
            "dialogue_query": "brief + '. ' + theme line", "chunk_words": dl.CHUNK_WORDS, "stride_words": dl.STRIDE_WORDS}
    (OUT / "picks.json").write_text(json.dumps({**meta, "picks": picks}, indent=1) + "\n")
    (OUT / "pool.json").write_text(json.dumps({**meta, "pairs": pool}, indent=1) + "\n")
    print(f"test: {len(units)} scenes, {int(speech.sum())} with speech; {len(pool)} unique pairs to label; "
          f"sheets in {DATA / 'sheets'}")


def card_text(p: dict) -> str:
    t = p["dialogue"] or "(no speech)"
    return t if len(t) <= DIALOGUE_SHOWN else t[:DIALOGUE_SHOWN].rsplit(" ", 1)[0] + " …"


def sheets(pool: list[dict], per_sheet: int = 5):
    """Primary labeller's sheets: brief, guide and theme lines, 4 frames, then the scene's dialogue. No system shown."""
    from PIL import Image, ImageDraw
    out = DATA / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    guide, th = label_guide(), themes()
    w = THUMB[0] * FRAMES_PER_SCENE + 4 * (FRAMES_PER_SCENE - 1) + 16
    wrap = 150
    for s in range(0, len(pool), per_sheet):
        chunk = pool[s:s + per_sheet]
        blocks = [textwrap.wrap("dialogue: " + card_text(p), wrap) for p in chunk]
        heights = [64 + THUMB[1] + 6 + 14 * len(b) + 12 for b in blocks]
        sheet = Image.new("RGB", (w, sum(heights)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for p, b, h in zip(chunk, blocks, heights):
            g = guide[p["ad"]]
            draw.text((8, y + 4), f"{p['pair']}  [{p['category']}]  {p['brief']}", fill="black")
            draw.text((8, y + 20), f"literal: {g['fits']}", fill="#1b5e20")
            draw.text((8, y + 34), f"theme: {th[p['ad']]['theme']}", fill="#1a3d8b")
            sheet.paste(Image.open(FRAMES / f"{p['scene']}.jpg"), (8, y + 52))
            for i, line in enumerate(b):
                draw.text((8, y + 58 + THUMB[1] + 14 * i), line, fill="#5a3a00")
            y += h
        sheet.save(out / f"sheet_{s // per_sheet + 1:03d}.jpg", quality=85)


def compare(picks: dict, label: dict, seed: int):
    per_ad, cols = {}, {f"{m}_{s}": [] for m in ("thematic", "literal", "ndcg") for s in SYSTEMS}
    for ad in ads_4d():
        known = [v for (a, _), v in label.items() if a == ad.id]
        row = {}
        for s in SYSTEMS:
            g = [label[(ad.id, x["scene"])] for x in picks[s][ad.id]]
            row[s] = {"gains": g, "thematic_precision@10": precision_at_k([int(v >= 1) for v in g], K),
                      "literal_precision@10": precision_at_k([int(v == 2) for v in g], K),
                      "graded_ndcg@10": graded_ndcg_at_k(g, known, K)}
            for m, key in (("thematic", "thematic_precision@10"), ("literal", "literal_precision@10"),
                           ("ndcg", "graded_ndcg@10")):
                cols[f"{m}_{s}"].append(row[s][key])
        per_ad[ad.id] = row
    arr = {k: np.array(v) for k, v in cols.items()}
    summary = {}
    for m in ("thematic", "literal", "ndcg"):
        summary[m] = {s: {"mean": float(arr[f"{m}_{s}"].mean()), "ci95": bootstrap(arr[f"{m}_{s}"], seed=seed)}
                      for s in SYSTEMS}
        for a, b in (("frames", "both"), ("frames", "dialogue"), ("dialogue", "both")):
            d = arr[f"{m}_{b}"] - arr[f"{m}_{a}"]
            ci = bootstrap(d, seed=seed)
            summary[m][f"{b}-{a}"] = {"mean": float(d.mean()), "ci95": ci, "above_zero": bool(ci[0] > 0),
                                      "below_zero": bool(ci[1] < 0), "ads_better": int((d > 0).sum()),
                                      "ads_same": int((d == 0).sum()), "ads_worse": int((d < 0).sum())}
    return summary, per_ad


def score(seed: int):
    meta = json.loads((OUT / "pool.json").read_text())
    pool = meta["pairs"]
    picks = json.loads((OUT / "picks.json").read_text())["picks"]
    primary = json.loads((OUT / "labels_primary.json").read_text())["labels"]
    missing = [p["pair"] for p in pool if p["pair"] not in primary]
    assert not missing, f"unlabelled pairs: {missing[:10]}"
    label = {(p["ad"], p["scene"]): primary[p["pair"]] for p in pool}
    summary, per_ad = compare(picks, label, seed)
    has_speech = {p["scene"]: bool(p["dialogue"]) for p in pool}
    data = {"summary": summary, "per_ad": per_ad, "pairs_labelled": len(pool),
            "scenes": meta["scenes"], "scenes_with_speech": meta["scenes_with_speech"],
            "label_counts": {str(v): sum(1 for x in primary.values() if x == v) for v in (0, 1, 2)},
            "picks_with_speech": {s: sum(has_speech[x["scene"]] for v in picks[s].values() for x in v) for s in SYSTEMS}}
    author_file = OUT / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [p["pair"] for p in pool if p["pair"] in author]
        a, b = [primary[k] for k in both], [author[k] for k in both]
        data["agreement"] = {"n": len(both), "exact_3_level": float(np.mean(np.array(a) == np.array(b))),
                             "thematic_or_better": agreement([int(x >= 1) for x in a], [int(x >= 1) for x in b]),
                             "literal": agreement([int(x == 2) for x in a], [int(x == 2) for x in b]),
                             "disagreements": [k for k in both if primary[k] != author[k]]}
        mixed = label | {(p["ad"], p["scene"]): author[p["pair"]] for p in pool if p["pair"] in author}
        data["sensitivity_author_labels_where_present"] = compare(picks, mixed, seed)[0]["thematic"]
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 4f amendment)", "k": K, "bootstrap_resamples": 10_000,
              "whisper_model": WHISPER_MODEL, "text_model": dl.TEXT_MODEL,
              "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    print("  ->", write_result(PHASE, "match_dialogue", LANE, seed, config, data)
          .rename(OUT / f"match_dialogue_{LANE}_seed{seed}.json"))
    for m, name in (("thematic", "thematic precision@10"), ("literal", "literal precision@10"), ("ndcg", "graded nDCG@10")):
        s = summary[m]
        print(f"{name}: " + "  ".join(f"{k} {s[k]['mean']:.3f}" for k in SYSTEMS))
        for k in ("both-frames", "dialogue-frames", "both-dialogue"):
            d = s[k]
            print(f"   {k}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
                  f"better/same/worse {d['ads_better']}/{d['ads_same']}/{d['ads_worse']}")
    print("scenes with speech:", data["scenes_with_speech"], "of", data["scenes"], "  picks with speech:",
          data["picks_with_speech"], "  label counts:", data["label_counts"])
    if "agreement" in data:
        g = data["agreement"]
        print(f"agreement on {g['n']}: exact {g['exact_3_level']:.2f}; thematic-or-better kappa "
              f"{g['thematic_or_better']['cohen_kappa']:.2f}; literal kappa {g['literal']['cohen_kappa']:.2f}")
    plot(per_ad, summary, OUT / "match_dialogue.png")


def plot(per_ad, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_ad)
    fig, ax = plt.subplots(figsize=(18, 4.8))
    x = np.arange(len(names))
    for j, (s, c) in enumerate((("frames", "#9e9e9e"), ("dialogue", "#c77c1e"), ("both", "#2b55a8"))):
        ax.bar(x + (j - 1) * 0.27, [per_ad[a][s]["thematic_precision@10"] for a in names], 0.27, color=c, label=s)
    ax.set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("thematic precision@10")
    s = summary["thematic"]
    ax.set_title("Phase 4f: frames {:.2f}, dialogue {:.2f}, both {:.2f}".format(
        s["frames"]["mean"], s["dialogue"]["mean"], s["both"]["mean"]), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("  ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["transcribe", "test", "score"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.step == "transcribe":
        for t in films():
            print(t, len(transcribe(t)["segments"]), flush=True)
    else:
        test() if args.step == "test" else score(args.seed)


if __name__ == "__main__":
    main()
