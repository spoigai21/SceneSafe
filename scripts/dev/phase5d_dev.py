"""Phase 5d development: tune on the 326 5c scenes relabelled under the obvious definitions."""
import json, sys
import numpy as np
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2]), str(__import__("pathlib").Path(__file__).resolve().parents[1])]
import run_phase5 as r5
from scenesafe.safety import rules as rl
from scenesafe.safety.tags import TAGS, TAGS_OBVIOUS, GRID_OBVIOUS_PROMPT, verify_p
R = str(__import__("pathlib").Path(__file__).resolve().parents[2]) + "/"
D = r5.DATA
label = json.load(open(R + "results/phase5d/labels_dev.json"))["labels"]
items = {it["id"]: it for it in json.load(open(r5.OUT_DIRS["5c"] / "pool.json"))["items"]}
meta = json.load(open(D / "strata_5c.json")); stratum, sizes = meta["strata"], meta["strata_sizes"]
ids = list(label)
scene = {i: items[i]["scene"] for i in ids}
dlg = {i: items[i]["dialogue"] for i in ids}
load = lambda f: {k: set(v["tags"]) for k, v in json.load(open(D / f)).items()}
first, strip, v5b, v5c = load("tags_5d_first.json"), load("tags.json"), load("tags_verified.json"), load("tags_5c.json")
# check every candidate on the dev scenes (grid first pass ∪ strip first pass)
cand = [(scene[i], t, r5.GRIDS / f"{scene[i]}.jpg", dlg[i]) for i in ids for t in TAGS
        if t in first[scene[i]] | strip[scene[i]]]
p = verify_p(cand, r5.DATA / "verify_5d_dev.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
wt = r5.weights(ids, stratum, sizes)
def report(name, tagsets, rule="standard"):
    block = set(rl.rule_set(rule)["block_all"])
    should = np.array([bool(set(label[i]) & block) for i in ids])
    did = np.array([bool(tagsets(scene[i]) & block) for i in ids])
    fp, miss, caught, f1 = r5.error_rates(did, should, wt, np.arange(len(ids)))
    blk = (wt * did).sum() / wt.sum()
    return f"{name:28s} {rule:8s} blocks {blk:5.1%} wrong {fp:4.0%} caught {caught:4.0%} F1 {f1:.2f}"
def ver(th, use_strip):
    return lambda s: {t for t in TAGS if t in (first[s] | (strip[s] if use_strip else set())) and p[f"{s}|{t}"] >= th}
for rule in ("lenient", "standard", "strict"):
    print(report("5b verifier", lambda s: v5b[s], rule))
    print(report("5c", lambda s: v5c[s], rule))
    print(report("5d first pass (grid)", lambda s: first[s], rule))
    for us in (False, True):
        for th in (0.1, 0.2, 0.3, 0.5, 0.7):
            print(report(f"5d check {'grid+strip' if us else 'grid'} p>={th}", ver(th, us), rule))
    print()
# per tag recall/precision, grid first pass and check at 0.5
for nm, f in (("first", lambda s: first[s]), ("check0.5", ver(0.5, False))):
    out = []
    for t in TAGS:
        tr = np.array([t in label[i] for i in ids]); sd = np.array([t in f(scene[i]) for i in ids])
        tp = (wt * (tr & sd)).sum()
        out.append(f"{t} p{tp / max((wt * sd).sum(), 1e-9):.2f}/r{tp / max((wt * tr).sum(), 1e-9):.2f}")
    print(nm, ", ".join(out))
