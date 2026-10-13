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
def ver(th, use_strip, grid=True):
    return lambda s: {t for t in TAGS if t in ((first[s] if grid else set()) | (strip[s] if use_strip else set())) and p[f"{s}|{t}"] >= th}
for rule in ("lenient", "standard", "strict"):
    print(report("5b verifier", lambda s: v5b[s], rule))
    for th in (0.2, 0.5, 0.7):
        print(report(f"strip cands, obvious chk {th}", ver(th, True, grid=False), rule))
        print(report(f"5b + grid chk {th}", lambda s, f=ver(th, False): v5b[s] | f(s), rule))
        print(report(f"5b + grid chk {th} weapons only", lambda s, f=ver(th, False): v5b[s] | (f(s) & {"weapons"}), rule))
    print()
block = set(rl.rule_set("standard")["block_all"])
print("standard-risky dev scenes (obvious labels), by stratum:",
      {s: sum(1 for i in ids if stratum[i] == s and set(label[i]) & block) for s in "ABCD"},
      "weights", {s: round(sizes[s] / sum(stratum[i] == s for i in ids), 1) for s in "ABCD"})
