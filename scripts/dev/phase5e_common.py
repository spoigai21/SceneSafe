import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "scripts")]
import run_phase5 as r5
from scenesafe.safety import rules as rl
from scenesafe.safety.tags import TAGS, TAGS_OBVIOUS, GRID_OBVIOUS_PROMPT, verify_p
SP = Path(__file__).resolve().parents[2] / "data" / "phase5" / "dev5e"
D = r5.DATA
POOLS = []   # (name, labels, scene by id, stratum by id, sizes)
for name, lab, pool, st in (("dev", "results/phase5d/labels_dev.json", "results/phase5c/pool.json", "strata_5c.json"),
                            ("t5d", "results/phase5d/labels_primary.json", "results/phase5d/pool.json", "strata_5d.json")):
    L = json.load(open(R / lab))["labels"]
    sc = {i["id"]: i["scene"] for i in json.load(open(R / pool))["items"]}
    dl = {i["id"]: i["dialogue"] for i in json.load(open(R / pool))["items"]}
    m = json.load(open(D / st))
    POOLS.append((name, L, sc, dl, m["strata"], m["strata_sizes"]))
load = lambda f: {k: set(v["tags"]) for k, v in json.load(open(D / f)).items()}
first, strip, v5b, t5d = load("tags_5d_first.json"), load("tags.json"), load("tags_verified.json"), load("tags_5d.json")
STRICT = set(rl.rule_set("strict")["block_all"])

def evaluate(name, tagsets, show_tags=True):
    lines = [f"{name}"]
    for pname, L, sc, _, stratum, sizes in POOLS:
        ids = list(L); wt = r5.weights(ids, stratum, sizes); ix = np.arange(len(ids))
        row = []
        for rule in ("lenient", "standard", "strict"):
            block = set(rl.rule_set(rule)["block_all"])
            should = np.array([bool(set(L[i]) & block) for i in ids])
            did = np.array([bool(tagsets(sc[i]) & block) for i in ids])
            fp, miss, caught, f1 = r5.error_rates(did, should, wt, ix)
            row.append(f"{rule[:3]} wrong {fp:3.0%} caught {caught:3.0%} F1 {f1:.2f}")
        lines.append(f"   {pname:4s} " + " | ".join(row))
    if show_tags:
        hit, tot, fl = {}, {}, {}
        for _, L, sc, *_ in POOLS:
            for i, t in L.items():
                said = tagsets(sc[i])
                for tag in STRICT:
                    tot[tag] = tot.get(tag, 0) + (tag in t); hit[tag] = hit.get(tag, 0) + (tag in t and tag in said)
                    fl[tag] = fl.get(tag, 0) + (tag in said)
        lines.append("   raw caught/labelled (flagged): " + ", ".join(f"{t} {hit[t]}/{tot[t]} ({fl[t]})" for t in TAGS if t in STRICT))
    print("\n".join(lines), flush=True)
