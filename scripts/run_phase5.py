"""Phase 5: brand safety. What does safety cost in sellable inventory, and how often does it block a scene that was
fine? Rules: EVAL-PROTOCOL.md, "Phase 5 amendment".

  practice   tag 60 random scenes of the 2 practice films, to try the detector prompt (not a result).
  tag        content tags for every scene of the 19 test films (data/phase5/tags.json).
  inventory  inventory removed per rule set (scene rules and the genre baseline), and fill rates on a slack and a tight
             book. Writes results/phase5/inventory_cpu-laptop_seed0.json.
  pool       the stratified labelling sample and its sheets (data/phase5/sheets).
  score      false-positive and miss rates from the labels. Writes results/phase5/safety_cpu-laptop_seed0.json.

Phase 5b (EVAL-PROTOCOL.md, "Phase 5b addendum"), with --detector verified:
  verify     a narrow yes/no question for each tag the detector gave a scene; the tag is kept only on yes
             (data/phase5/tags_verified.json).
  inventory, pool, score   as above, on the verified tags, written to results/phase5b/. The 5b pool is drawn fresh,
             leaving out the 480 Phase 5 scenes, and score compares both detectors on the same new labels.

Phase 5c (EVAL-PROTOCOL.md, "Phase 5c addendum"), with --detector 5c:
  grids      a larger 2 × 2 frame grid for every scene (data/phase5/grids).
  verify5c   candidates (first pass, card keywords, CLIP top 5%) checked on the grids; a tag is kept if P(yes) >= 0.2
             (data/phase5/tags_5c.json).
  inventory, pool, score   as above, written to results/phase5c/. The pool leaves out every scene labelled in 5 and 5b
             (the development set) and its sheets show the grids; score compares 5c with the 5b verifier.
"""

import argparse
import json
import textwrap

import numpy as np

from run_phase4f import FRAMES, SCENE_THRESHOLD, ad_texts, describe
from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.index.units import load_corpus, scenes
from scenesafe.match import cards as cd
from scenesafe.match import dialogue as dl
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d
from scenesafe.match.display import FRAMES_PER_SCENE, GRID_CELL, THUMB, frame_grid
from scenesafe.match.metrics import agreement
from scenesafe.match.transcribe import scene_text, transcribe
from scenesafe.safety import rules as rl
from scenesafe.match.textvec import embed_texts
from scenesafe.safety import candidates as cand
from scenesafe.safety.tags import (GRID_OBVIOUS_PROMPT, GRID_TAG_PROMPT, TAGS, TAGS_OBVIOUS, VERIFY_THRESHOLD,
                                   tag_scenes, verify, verify_p)

PHASE = 5
LANE = "cpu-laptop"
OUT = REPO_ROOT / "results" / "phase5"
DATA = REPO_ROOT / "data" / "phase5"
PRACTICE_FILMS = ("calvary", "Hitchcock_Secret_Agent")
RELEVANT_SHARE = 0.10                  # an ad buys only breaks whose preceding scene is in its top 10%
DEMAND = {"slack": 10, "tight": 80}    # breaks per ad
PER_STRATUM = 120
AUTHOR_SAMPLE, AUTHOR_SEED = 100, 8          # Phase 5b: default_rng(9)
STRATA = ("A", "B", "C", "D")          # blocked by lenient / standard only / strict only / kept by strict
TAG_FILES = {"detector": "tags.json", "verified": "tags_verified.json", "5c": "tags_5c.json", "5d": "tags_5d.json",
             "5d_first": "tags_5d_first.json", "5e": "tags_5e.json"}
OUT_DIRS = {"detector": OUT, "verified": REPO_ROOT / "results" / "phase5b", "5c": REPO_ROOT / "results" / "phase5c",
            "5d": REPO_ROOT / "results" / "phase5d", "5e": REPO_ROOT / "results" / "phase5e"}
PREFIX = {"detector": "S", "verified": "T", "5c": "U", "5d": "V"}
COMPARE = {"detector": ("detector",), "verified": ("detector", "verified"), "5c": ("verified", "5c", "detector"),
           "5d": ("verified", "5d", "5d_first", "5c"), "5e": ("5d", "5e", "verified")}
VERIFY5D_THRESHOLD = 0.7               # Phase 5d, chosen on the development set
VERIFY5D_TAGS = ("weapons",)           # Phase 5d: only weapons is taken from the grid (chosen on the development set)
VERIFY5E_TAGS = ("horror", "smoking", "alcohol")   # Phase 5e: also taken from the grid, P(yes) >= 0.7
GRIDS = DATA / "grids"
SHEETS = {"detector": "sheets", "verified": "sheets_5b", "5c": "sheets_5c", "5d": "sheets_5d", "5e": "sheets_5e"}


def films() -> list[str]:
    return sorted(p.stem for p in cd.CARD_DIR.glob("*.json"))


def genre_of() -> dict[str, str]:
    out = {}
    for m in ("manifest_4d.json", "manifest_4h.json"):
        out |= {t["id"]: t["genre"] for t in json.loads((REPO_ROOT / "corpus" / m).read_text())["titles"]}
    return out


def setup():
    ids = films()
    corpus = load_corpus(ids)
    units = scenes(corpus, SCENE_THRESHOLD)
    names = [describe(corpus, units, u)["scene"] for u in range(len(units))]
    film = [corpus.ids[units.title[u]] for u in range(len(units))]
    cards = {}
    for t in ids:
        cards |= json.loads((cd.CARD_DIR / f"{t}.json").read_text())
    missing = [n for n in names if n not in cards]
    assert not missing and len(cards) == len(names), f"scenes and cards differ: {missing[:5]}"
    return corpus, units, names, film, cards


def practice():
    corpus = load_corpus(list(PRACTICE_FILMS))
    units = scenes(corpus, SCENE_THRESHOLD)
    tr = {t: transcribe(t)["segments"] for t in PRACTICE_FILMS}
    pick = np.random.default_rng(0).choice(len(units), 60, replace=False)
    items = []
    for u in sorted(pick):
        d = describe(corpus, units, int(u))
        items.append((d["scene"], FRAMES / f"{d['scene']}.jpg",
                      scene_text(tr[d["title"]], units.start[u], units.end[u])))
    out = tag_scenes(items, DATA / "practice_tags.json")
    for name, _, dlg in items:
        print(f"{name:32s} {out[name]['tags']}  <- {out[name]['answer'][:80]!r}  | {dlg[:60]!r}")


def tag():
    _, _, names, _, cards = setup()
    tag_scenes([(n, FRAMES / f"{n}.jpg", cards[n]["dialogue"]) for n in names], DATA / "tags.json")


def run_verify():
    _, _, names, _, cards = setup()
    t = json.loads((DATA / "tags.json").read_text())
    items = [(n, tag, FRAMES / f"{n}.jpg", cards[n]["dialogue"]) for n in names for tag in t[n]["tags"]]
    ans = verify(items, DATA / "verify.json")
    out = {n: {"tags": [tag for tag in t[n]["tags"] if ans[f"{n}|{tag}"]["yes"]]} for n in names}
    (DATA / "tags_verified.json").write_text(json.dumps(out, indent=1) + "\n")
    kept = sum(len(v["tags"]) for v in out.values())
    print(f"verify: {len(items)} tags checked, {kept} kept ({kept / len(items):.1%})")


def grids():
    corpus, units, names, film, _ = setup()
    GRIDS.mkdir(parents=True, exist_ok=True)
    for u, n in enumerate(names):
        if not (GRIDS / f"{n}.jpg").exists():
            frame_grid(film[u], units.start[u], units.end[u], GRIDS / f"{n}.jpg")
    print(f"grids: {len(names)} scenes in {GRIDS}")


def verify5c():
    corpus, units, names, film, cards = setup()
    first = json.loads((DATA / "tags.json").read_text())
    tags = list(TAGS)
    q = embed_texts([p for t in tags for p in cand.CLIP_PROMPTS[t]], DATA / "clip_tags.npy")
    per_prompt = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    bounds = np.cumsum([0] + [len(cand.CLIP_PROMPTS[t]) for t in tags])
    scores = np.stack([per_prompt[:, bounds[j]:bounds[j + 1]].max(1) for j in range(len(tags))], 1)
    by_clip = cand.clip_tags(scores, tags)
    items, source = [], {}
    for u, n in enumerate(names):
        c = set(first[n]["tags"]) | cand.keyword_tags(cards[n]["card"]) | by_clip[u]
        for t in sorted(c, key=tags.index):
            items.append((n, t, GRIDS / f"{n}.jpg", cards[n]["dialogue"]))
            source[f"{n}|{t}"] = [k for k, ok in (("first_pass", t in first[n]["tags"]),
                                                  ("keywords", t in cand.keyword_tags(cards[n]["card"])),
                                                  ("clip", t in by_clip[u])) if ok]
    p = verify_p(items, DATA / "verify_5c.json")
    out = {n: {"tags": []} for n in names}
    for n, t, _, _ in items:
        if p[f"{n}|{t}"] >= VERIFY_THRESHOLD:
            out[n]["tags"].append(t)
    (DATA / "tags_5c.json").write_text(json.dumps(out, indent=1) + "\n")
    (DATA / "candidates_5c.json").write_text(json.dumps(source) + "\n")
    kept = sum(len(v["tags"]) for v in out.values())
    print(f"verify5c: {len(items)} candidates ({sum('first_pass' in v for v in source.values())} first pass), "
          f"{kept} kept ({kept / len(items):.1%})")


def tag5d():
    """Phase 5d first pass: the 9-tag question with the obvious definitions, on the grids."""
    _, _, names, _, cards = setup()
    tag_scenes([(n, GRIDS / f"{n}.jpg", cards[n]["dialogue"]) for n in names], DATA / "tags_5d_first.json",
               GRID_TAG_PROMPT)


def verify5d():
    """Phase 5d: the 5b verifier's tags, plus weapons where the grid first pass gave it and the grid check with the
    obvious definition says P(yes) >= VERIFY5D_THRESHOLD (both chosen on the development set)."""
    _, _, names, _, cards = setup()
    first = json.loads((DATA / "tags_5d_first.json").read_text())
    base = json.loads((DATA / "tags_verified.json").read_text())
    items = [(n, t, GRIDS / f"{n}.jpg", cards[n]["dialogue"]) for n in names for t in VERIFY5D_TAGS
             if t in first[n]["tags"]]
    p = verify_p(items, DATA / "verify_5d.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
    out = {n: {"tags": list(base[n]["tags"])} for n in names}
    added = 0
    for n, t, _, _ in items:
        if p[f"{n}|{t}"] >= VERIFY5D_THRESHOLD and t not in out[n]["tags"]:
            out[n]["tags"] = [x for x in TAGS if x in out[n]["tags"] or x == t]
            added += 1
    (DATA / "tags_5d.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"verify5d: {len(items)} grid candidates checked, {added} tags added to the 5b tags")


def verify5e():
    """Phase 5e: the 5d tags, plus horror, smoking and alcohol where either first pass (grid or strip) gave them and the
    grid check with the obvious definition gives P(yes) >= VERIFY5D_THRESHOLD."""
    _, _, names, _, cards = setup()
    first = json.loads((DATA / "tags_5d_first.json").read_text())
    strip = json.loads((DATA / "tags.json").read_text())
    base = json.loads((DATA / "tags_5d.json").read_text())
    items = [(n, t, GRIDS / f"{n}.jpg", cards[n]["dialogue"]) for n in names for t in VERIFY5E_TAGS
             if t in set(first[n]["tags"]) | set(strip[n]["tags"])]
    p = verify_p(items, DATA / "verify_5d.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
    out = {n: {"tags": list(base[n]["tags"])} for n in names}
    added = 0
    for n, t, _, _ in items:
        if p[f"{n}|{t}"] >= VERIFY5D_THRESHOLD and t not in out[n]["tags"]:
            out[n]["tags"] = [x for x in TAGS if x in out[n]["tags"] or x == t]
            added += 1
    (DATA / "tags_5e.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"verify5e: {len(items)} grid candidates checked, {added} tags added to the 5d tags")


def pool5e():
    """Phase 5e test: the 452 Phase 5b scenes (never used to develop 5d or 5e), with their 5b strata, shown as grids
    for relabelling under the obvious definitions."""
    items = json.loads((OUT_DIRS["verified"] / "pool.json").read_text())["items"]
    meta = json.loads((DATA / "strata_verified.json").read_text())
    OUT_DIRS["5e"].mkdir(parents=True, exist_ok=True)
    (DATA / "strata_5e.json").write_text(json.dumps(meta, indent=1) + "\n")
    (OUT_DIRS["5e"] / "pool.json").write_text(json.dumps({"items": items}, indent=1) + "\n")
    grid_sheets(items, DATA / SHEETS["5e"])
    print(f"pool5e: {len(items)} scenes; strata sizes {meta['strata_sizes']}")


def load_tags(names, which: str = "detector") -> list[set]:
    t = json.loads((DATA / TAG_FILES[which]).read_text())
    return [set(t[n]["tags"]) for n in names]


def strata(tags: list[set]) -> np.ndarray:
    sets = {r: set(rl.rule_set(r)["block_all"]) for r in rl.RULE_SETS}
    out = []
    for t in tags:
        out.append("A" if t & sets["lenient"] else "B" if t & sets["standard"] else "C" if t & sets["strict"] else "D")
    return np.array(out)


def inventory(seed: int, which: str = "detector"):
    out_dir = OUT_DIRS[which]
    catalogue = ads_4d()
    corpus, units, names, film, cards = setup()
    tags = load_tags(names, which)
    genre = genre_of()

    q = np.load(REPO_ROOT / "data" / "phase4d" / "briefs_b32.npy")         # the frames+cards matcher, as in 4g/4h
    frame = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    card = dl.embed_text([cd.card_text(cards[n]["card"]) for n in names], DATA / "cards.npy") @ \
        dl.embed_text(ad_texts(catalogue), DATA / "ads.npy").T
    both = dl.zscore(frame) + dl.zscore(card)

    brk = rl.breaks(film)
    pre = np.array([a for a, _ in brk])
    score = both[pre]                                                    # a break is as relevant as the scene before it
    cut = np.quantile(both, 1 - RELEVANT_SHARE, axis=0)
    relevant = score >= cut

    rows, no_rules = {}, {}
    everything = np.ones_like(relevant)
    for book, d in DEMAND.items():
        sold = rl.allocate(relevant & everything, score, d, seed)
        no_rules[book] = {"demand": d * len(catalogue), "sold": int(sold.sum()),
                          "fill_rate": float(sold.sum() / (d * len(catalogue)))}
    for r in rl.RULE_SETS:
        cfg = rl.rule_set(r)
        block = set(cfg["block_all"])
        scene_blocked = rl.blocked_for_all(tags, block)
        ok = rl.sellable(tags, brk, catalogue, block)
        all_removed = np.array([scene_blocked[a] or scene_blocked[b] for a, b in brk])
        gblock = np.array([genre[f] in cfg["block_genres"] for f in film])
        g_removed = np.array([gblock[a] or gblock[b] for a, b in brk])
        g_ok = np.repeat(~g_removed[:, None], len(catalogue), axis=1)
        entry = {"block_all": sorted(block), "scenes_blocked": float(scene_blocked.mean()),
                 "breaks_removed_for_every_ad": float(all_removed.mean()),
                 "slots_removed_mean_over_ads": float(1 - ok.mean()),
                 "slots_removed_per_ad": {ad.id: float(1 - ok[:, j].mean()) for j, ad in enumerate(catalogue)},
                 "relevant_slots_removed": float(1 - (ok & relevant).sum() / relevant.sum()),
                 "genre_baseline": {"block_genres": cfg["block_genres"], "scenes_blocked": float(gblock.mean()),
                                    "breaks_removed_for_every_ad": float(g_removed.mean()),
                                    "relevant_slots_removed": float(1 - (g_ok & relevant).sum() / relevant.sum())},
                 "books": {}}
        for book, d in DEMAND.items():
            for key, allowed in (("scene_rules", ok), ("genre_baseline", g_ok)):
                sold = rl.allocate(relevant & allowed, score, d, seed)
                entry["books"].setdefault(book, {})[key] = {
                    "fill_rate": float(sold.sum() / (d * len(catalogue))),
                    "ads_fully_filled": int((sold >= d).sum()), "ads_with_nothing": int((sold == 0).sum())}
        rows[r] = entry

    tag_rate = {t: float(np.mean([t in s for s in tags])) for t in TAGS}
    data = {"films": films(), "scenes": len(names), "breaks": len(brk), "ads": len(catalogue), "tags": which,
            "tag_rate": tag_rate, "strata_sizes": {k: int(v) for k, v in zip(*np.unique(strata(tags), return_counts=True))},
            "no_rules": no_rules, "rule_sets": rows}
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 5 amendment" + {"detector": ")", "verified": ", Phase 5b addendum)",
                                                                   "5c": ", Phase 5c addendum)",
                                                                   "5d": ", Phase 5d addendum)",
                                                                   "5e": ", Phase 5e addendum)"}[which],
              "tags": which, "relevant_share": RELEVANT_SHARE,
              "demand_per_ad": DEMAND, "detector": cd.CARD_MODEL, "matcher": "frames+cards (Phase 4g)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    print("  ->", write_result(PHASE, "inventory", LANE, seed, config, data)
          .rename(out_dir / f"inventory_{LANE}_seed{seed}.json"))
    print(f"{len(names)} scenes, {len(brk)} breaks; tag rates: " +
          ", ".join(f"{t} {v:.1%}" for t, v in tag_rate.items()))
    for book in DEMAND:
        print(f"  {book} book, no rules: fill {no_rules[book]['fill_rate']:.1%}")
    for r, e in rows.items():
        g = e["genre_baseline"]
        print(f"{r:9s} scenes blocked {e['scenes_blocked']:.1%} | breaks removed for all {e['breaks_removed_for_every_ad']:.1%}"
              f" | slots removed {e['slots_removed_mean_over_ads']:.1%} | relevant slots removed "
              f"{e['relevant_slots_removed']:.1%}  || genre: scenes {g['scenes_blocked']:.1%}, relevant slots "
              f"{g['relevant_slots_removed']:.1%}")
        for book in DEMAND:
            b = e["books"][book]
            print(f"    {book}: fill scene rules {b['scene_rules']['fill_rate']:.1%}, genre {b['genre_baseline']['fill_rate']:.1%}")


def pool(which: str = "detector"):
    out_dir = OUT_DIRS[which]
    _, _, names, film, cards = setup()
    st = strata(load_tags(names, which))
    used = set()
    if which != "detector":                                   # 5b and 5c: fresh scenes only
        earlier = {"verified": ("detector",), "5c": ("detector", "verified"), "5d": ("detector", "verified", "5c")}[which]
        for e in earlier:
            used |= {it["scene"] for it in json.loads((OUT_DIRS[e] / "pool.json").read_text())["items"]}
    free = np.array([n not in used for n in names])
    rng = np.random.default_rng(0)
    picked = []
    for s in STRATA:
        idx = np.flatnonzero((st == s) & free)
        picked += [int(i) for i in rng.choice(idx, min(PER_STRATUM, len(idx)), replace=False)]
    order = np.random.default_rng(0).permutation(len(picked))
    items = [{"id": f"{PREFIX[which]}{n:04d}", "scene": names[picked[j]], "title": film[picked[j]],
              "dialogue": cards[names[picked[j]]]["dialogue"]} for n, j in enumerate(order, 1)]
    meta = {"tags": which, "strata_sizes": {s: int(((st == s) & free).sum()) for s in STRATA},
            "per_stratum": PER_STRATUM, "left_out": len(used),
            "strata": {it["id"]: str(st[names.index(it["scene"])]) for it in items}}
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"strata_{which}.json").write_text(json.dumps(meta, indent=1) + "\n")   # kept out of the labellers' way
    (out_dir / "pool.json").write_text(json.dumps({"items": items}, indent=1) + "\n")
    if which in ("5c", "5d"):
        grid_sheets(items, DATA / SHEETS[which])
    else:
        sheets(items, DATA / SHEETS[which])
    print(f"pool: {len(items)} scenes; strata sizes {meta['strata_sizes']}; left out {len(used)}")


def sheets(items, out, per_sheet: int = 5):
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    w = THUMB[0] * FRAMES_PER_SCENE + 4 * (FRAMES_PER_SCENE - 1) + 16
    for s in range(0, len(items), per_sheet):
        chunk = items[s:s + per_sheet]
        blocks = [textwrap.wrap("dialogue: " + (p["dialogue"] or "(no speech)"), 150) for p in chunk]
        heights = [24 + THUMB[1] + 6 + 14 * len(b) + 12 for b in blocks]
        sheet = Image.new("RGB", (w, sum(heights)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for p, b, h in zip(chunk, blocks, heights):
            draw.text((8, y + 4), p["id"], fill="black")
            sheet.paste(Image.open(FRAMES / f"{p['scene']}.jpg"), (8, y + 20))
            for i, line in enumerate(b):
                draw.text((8, y + 26 + THUMB[1] + 14 * i), line, fill="#5a3a00")
            y += h
        sheet.save(out / f"sheet_{s // per_sheet + 1:03d}.jpg", quality=85)


def weights(ids, stratum: dict, sizes: dict) -> np.ndarray:
    """Each labelled scene stands for (stratum size ÷ scenes labelled in that stratum) scenes."""
    n = {s: sum(stratum[i] == s for i in ids) for s in STRATA}
    return np.array([sizes[stratum[i]] / n[stratum[i]] for i in ids])


def error_rates(did, should, wt, ix):
    fp = (wt[ix] * (did[ix] & ~should[ix])).sum() / max((wt[ix] * did[ix]).sum(), 1e-12)
    miss = (wt[ix] * (~did[ix] & should[ix])).sum() / max((wt[ix] * ~did[ix]).sum(), 1e-12)
    caught = (wt[ix] * (did[ix] & should[ix])).sum() / max((wt[ix] * should[ix]).sum(), 1e-12)
    precision = 1 - fp
    f1 = 2 * precision * caught / max(precision + caught, 1e-12)
    return np.array([fp, miss, caught, f1])


def grid_sheets(items, out, per_sheet: int = 3):
    """Phase 5c sheets: the larger 2 × 2 grid for each scene, then its dialogue."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    w, gh = GRID_CELL[0] * 2 + 4 + 16, GRID_CELL[1] * 2 + 4
    for s in range(0, len(items), per_sheet):
        chunk = items[s:s + per_sheet]
        blocks = [textwrap.wrap("dialogue: " + (p["dialogue"] or "(no speech)"), 120) for p in chunk]
        heights = [24 + gh + 6 + 14 * len(b) + 14 for b in blocks]
        sheet = Image.new("RGB", (w, sum(heights)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for p, b, h in zip(chunk, blocks, heights):
            draw.text((8, y + 4), p["id"] + "   (frames in time order: top left, top right, bottom left, bottom right)",
                      fill="black")
            sheet.paste(Image.open(GRIDS / f"{p['scene']}.jpg"), (8, y + 20))
            for i, line in enumerate(b):
                draw.text((8, y + 26 + gh + 14 * i), line, fill="#5a3a00")
            y += h
        sheet.save(out / f"sheet_{s // per_sheet + 1:03d}.jpg", quality=85)


def rates(ids, label: dict, stratum: dict, sizes: dict, decisions: dict, block: set, seed: int) -> dict:
    """Weighted false-positive and miss rates for each blocking rule in decisions ({name: bool per id}), with a
    bootstrap within strata (10,000 resamples). The truth is whether the labeller marked any tag in block. If there
    are two or more rules, the paired difference of each against the first is reported too."""
    should = np.array([bool(set(label[i]) & block) for i in ids])
    wt = weights(ids, stratum, sizes)
    groups = [g for g in (np.flatnonzero(np.array([stratum[i] for i in ids]) == s) for s in STRATA) if len(g)]
    did = {k: np.array([bool(v[i]) for i in ids]) for k, v in decisions.items()}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in did}
    for _ in range(10_000):
        ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for k in did:
            boot[k].append(error_rates(did[k], should, wt, ix))
    boot = {k: np.array(v) for k, v in boot.items()}
    allix = np.arange(len(ids))
    out = {"should_block_share": float((wt * should).sum() / wt.sum())}
    first = next(iter(did))
    for k in did:
        pt = error_rates(did[k], should, wt, allix)
        lo, hi = np.percentile(boot[k], [2.5, 97.5], axis=0)
        out[k] = {"false_positive_rate": float(pt[0]), "fp_ci95": [float(lo[0]), float(hi[0])],
                  "miss_rate": float(pt[1]), "miss_ci95": [float(lo[1]), float(hi[1])],
                  "caught_share": float(pt[2]), "caught_ci95": [float(lo[2]), float(hi[2])],
                  "f1": float(pt[3]), "f1_ci95": [float(lo[3]), float(hi[3])],
                  "blocked_share": float((wt * did[k]).sum() / wt.sum())}
        if k != first:
            d = boot[k] - boot[first]
            dpt = pt - error_rates(did[first], should, wt, allix)
            lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
            out[k][f"minus_{first}"] = {"false_positive_rate": float(dpt[0]), "fp_ci95": [float(lo[0]), float(hi[0])],
                                        "miss_rate": float(dpt[1]), "miss_ci95": [float(lo[1]), float(hi[1])],
                                        "caught_share": float(dpt[2]), "caught_ci95": [float(lo[2]), float(hi[2])],
                                        "f1": float(dpt[3]), "f1_ci95": [float(lo[3]), float(hi[3])]}
    return out


def score(seed: int, which: str = "detector"):
    out_dir = OUT_DIRS[which]
    items = {it["id"]: it for it in json.loads((out_dir / "pool.json").read_text())["items"]}
    meta = json.loads((DATA / f"strata_{which}.json").read_text())
    stratum, sizes = meta["strata"], meta["strata_sizes"]
    tagsets = {k: json.loads((DATA / TAG_FILES[k]).read_text()) for k in COMPARE[which]}
    label = json.loads((out_dir / "labels_primary.json").read_text())["labels"]
    assert set(label) == set(items), "every pool scene needs a label"
    ids = list(label)
    genre = genre_of()
    out = {"tags": which, "pool": len(items), "strata_sizes": sizes, "left_out": meta.get("left_out", 0),
           "rule_sets": {}, "per_tag": {}}
    for r in rl.RULE_SETS:
        cfg = rl.rule_set(r)
        block = set(cfg["block_all"])
        dec = {k: {i: set(t[items[i]["scene"]]["tags"]) & block for i in ids} for k, t in tagsets.items()}
        dec["genre_baseline"] = {i: genre[items[i]["title"]] in cfg["block_genres"] for i in ids}
        out["rule_sets"][r] = rates(ids, label, stratum, sizes, dec, block, seed)
    wt = weights(ids, stratum, sizes)
    for k, t in tagsets.items():
        out["per_tag"][k] = {}
        for tag in TAGS:
            truth = np.array([tag in label[i] for i in ids])
            said = np.array([tag in t[items[i]["scene"]]["tags"] for i in ids])
            tp = (wt * (truth & said)).sum()
            out["per_tag"][k][tag] = {"labelled_share": float((wt * truth).sum() / wt.sum()),
                                      "precision": float(tp / (wt * said).sum()) if said.any() else None,
                                      "recall": float(tp / (wt * truth).sum()) if truth.any() else None,
                                      "n_detected_in_pool": int(said.sum()), "n_labelled_in_pool": int(truth.sum())}
    author_file = out_dir / "labels_author.json"
    if author_file.exists():
        author = json.loads(author_file.read_text())["labels"]
        both = [i for i in ids if i in author]
        std = set(rl.rule_set("standard")["block_all"])
        out["agreement"] = {"n": len(both),
                            "standard_decision": agreement([int(bool(set(label[i]) & std)) for i in both],
                                                           [int(bool(set(author[i]) & std)) for i in both]),
                            "per_tag": {t: agreement([int(t in label[i]) for i in both], [int(t in author[i]) for i in both])
                                        for t in TAGS}}
    config = {"protocol": "EVAL-PROTOCOL.md (Phase 5 amendment" + {"detector": ")", "verified": ", Phase 5b addendum)",
                                                                   "5c": ", Phase 5c addendum)",
                                                                   "5d": ", Phase 5d addendum)",
                                                                   "5e": ", Phase 5e addendum)"}[which],
              "tags": which, "per_stratum": PER_STRATUM, "bootstrap_resamples": 10_000,
              "author_sample": {"size": AUTHOR_SAMPLE, "seed": AUTHOR_SEED}}
    print("  ->", write_result(PHASE, "safety", LANE, seed, config, out).rename(out_dir / f"safety_{LANE}_seed{seed}.json"))
    print("weighted to", sum(sizes.values()), "scenes")
    for r, e in out["rule_sets"].items():
        print(f"{r}: should block {e['should_block_share']:.1%}")
        for k, x in e.items():
            if not isinstance(x, dict):
                continue
            line = (f"   {k:15s} blocks {x['blocked_share']:.1%}  false positives {x['false_positive_rate']:.1%} "
                    f"[{x['fp_ci95'][0]:.1%}, {x['fp_ci95'][1]:.1%}]  misses {x['miss_rate']:.1%} "
                    f"[{x['miss_ci95'][0]:.1%}, {x['miss_ci95'][1]:.1%}]  caught {x['caught_share']:.0%} "
                    f"[{x['caught_ci95'][0]:.0%}, {x['caught_ci95'][1]:.0%}]  F1 {x['f1']:.2f} "
                    f"[{x['f1_ci95'][0]:.2f}, {x['f1_ci95'][1]:.2f}]")
            for dk, dv in x.items():
                if dk.startswith("minus_"):
                    line += (f"\n      {dk}: fp {dv['false_positive_rate']:+.1%} [{dv['fp_ci95'][0]:+.1%}, "
                             f"{dv['fp_ci95'][1]:+.1%}]  miss {dv['miss_rate']:+.1%} [{dv['miss_ci95'][0]:+.1%}, "
                             f"{dv['miss_ci95'][1]:+.1%}]  caught {dv['caught_share']:+.1%} [{dv['caught_ci95'][0]:+.1%}, "
                             f"{dv['caught_ci95'][1]:+.1%}]  F1 {dv['f1']:+.3f} [{dv['f1_ci95'][0]:+.3f}, {dv['f1_ci95'][1]:+.3f}]")
            print(line)
    for k, per in out["per_tag"].items():
        print(k + ":")
        for t, x in per.items():
            pr = "-" if x["precision"] is None else f"{x['precision']:.2f}"
            rc = "-" if x["recall"] is None else f"{x['recall']:.2f}"
            print(f"  {t:10s} labelled {x['labelled_share']:.1%}  precision {pr}  recall {rc}")
    if "agreement" in out:
        print("agreement (standard decision):", out["agreement"]["standard_decision"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["practice", "tag", "verify", "grids", "verify5c", "tag5d", "verify5d", "verify5e", "pool5e", "inventory",
                                     "pool", "score"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detector", choices=list(OUT_DIRS), default="detector")
    args = ap.parse_args()
    if args.step == "practice":
        practice()
    elif args.step == "tag":
        tag()
    elif args.step == "verify":
        run_verify()
    elif args.step == "grids":
        grids()
    elif args.step == "verify5c":
        verify5c()
    elif args.step == "tag5d":
        tag5d()
    elif args.step == "verify5d":
        verify5d()
    elif args.step == "verify5e":
        verify5e()
    elif args.step == "pool5e":
        pool5e()
    elif args.step == "inventory":
        inventory(args.seed, args.detector)
    elif args.step == "pool":
        pool(args.detector)
    else:
        score(args.seed, args.detector)


if __name__ == "__main__":
    main()
