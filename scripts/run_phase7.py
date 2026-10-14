"""Phase 7: scene signals become priced inventory. Does scene-level understanding change what the inventory is worth?
Rules: EVAL-PROTOCOL.md, "Phase 7 amendment"; parameters in configs/phase7.yaml.

  run    segments (safety class × context) for the 4,903 breaks of the 19 Phase 5 films, a valuation ordering, and a
         flat pool against a segmented book on the same seeded demand (seeds 0–9, premium 2 and the sensitivity
         premiums, slack and tight books). Writes results/phase7/inventory_cpu-laptop_seed0.json.
"""

import argparse
import json

import numpy as np
import yaml

from run_phase4f import ad_texts
from run_phase5 import DATA, setup
from scenesafe.common.results import REPO_ROOT, write_result
from scenesafe.match import cards as cd
from scenesafe.match import dialogue as dl
from scenesafe.match import v2
from scenesafe.match.catalogue import ads_4d
from scenesafe.safety import rules as rl

PHASE = 7
LANE = "cpu-laptop"
CONFIG = REPO_ROOT / "configs" / "phase7.yaml"
CLASSES = ("clean", "mild", "sensitive", "restricted")
ALLOWED = {"lenient": {"clean", "mild", "sensitive"}, "standard": {"clean", "mild"}, "strict": {"clean"}}


def signals():
    """Per break: safety class and context; per (break, ad): relevant and allowed under each rule set."""
    cfg = yaml.safe_load(CONFIG.read_text())
    catalogue = ads_4d()
    corpus, units, names, film, cards = setup()
    t = json.loads((DATA / "tags_5d.json").read_text())
    tags = [set(t[n]["tags"]) for n in names]
    q = np.load(REPO_ROOT / "data" / "phase4d" / "briefs_b32.npy")          # the Phase 5 inventory scores
    frame = v2.scene_scores(corpus.emb @ q.T, np.zeros(len(corpus.emb), bool), units.first)
    card = dl.embed_text([cd.card_text(cards[n]["card"]) for n in names], DATA / "cards.npy") @ \
        dl.embed_text(ad_texts(catalogue), DATA / "ads.npy").T
    both = dl.zscore(frame) + dl.zscore(card)
    brk = rl.breaks(film)
    score = both[np.array([a for a, _ in brk])]
    relevant = score >= np.quantile(both, 1 - cfg["relevant_share"], axis=0)

    sets = {r: set(rl.rule_set(r)["block_all"]) for r in rl.RULE_SETS}
    near = [tags[a] | tags[b] for a, b in brk]
    cls = np.array(["restricted" if n & sets["lenient"] else "sensitive" if n & sets["standard"] else
                    "mild" if n & sets["strict"] else "clean" for n in near])
    allowed = {r: rl.sellable(tags, brk, catalogue, sets[r]) for r in rl.RULE_SETS}

    group_of = {c: g for g, cats in cfg["contexts"].items() for c in cats}
    groups = list(cfg["contexts"])
    ad_group = np.array([groups.index(group_of[ad.category]) for ad in catalogue])
    counts = np.stack([relevant[:, ad_group == g].sum(1) for g in range(len(groups))], 1)
    sums = np.stack([np.where(relevant[:, ad_group == g], score[:, ad_group == g], 0).sum(1)
                     for g in range(len(groups))], 1)
    best = np.array([max(range(len(groups)), key=lambda g: (counts[b, g], sums[b, g])) for b in range(len(brk))])
    context = np.where(counts.max(1) > 0, np.array(groups)[best], "general")
    return cfg, catalogue, brk, cls, context, relevant, allowed, ad_group, groups


def demand(cfg, catalogue, seed):
    rng = np.random.default_rng(seed)
    rules, shares = list(cfg["rule_set_shares"]), list(cfg["rule_set_shares"].values())
    rule = [("strict" if ad.category in cfg["always_strict"] else None) for ad in catalogue]
    draws = rng.choice(rules, size=len(catalogue), p=shares)
    rule = [r or d for r, d in zip(rule, draws)]
    base = rng.uniform(*cfg["base_value"], size=len(catalogue))
    budget = rng.integers(cfg["budget_breaks"][0], cfg["budget_breaks"][1] + 1, size=len(catalogue))
    return rule, base, budget


def best_price(values: np.ndarray, budgets: np.ndarray, supply: int) -> tuple[float, int]:
    """Revenue-maximising posted price over the buyers' own values: (price, breaks sold)."""
    best = (0.0, 0, 0.0)
    for p in np.unique(values[values > 0]):
        sold = min(supply, int(budgets[values >= p].sum()))
        if p * sold > best[2]:
            best = (float(p), sold, p * sold)
    return best[0], best[1]


def flat_book(v, budget):
    """One pool, one price. Each buyer values the pool at its average value per break."""
    avg = v.mean(0)
    p, sold = best_price(avg, budget, v.shape[0])
    buyers = avg >= p if sold else np.zeros(len(avg), bool)
    want = np.where(buyers, budget, 0).astype(float)
    got = want * (sold / want.sum()) if want.sum() else want                 # rationed in proportion when short
    return {"price": p, "breaks_sold": sold, "revenue": p * sold, "value_delivered": float((got * avg).sum()),
            "buyers": int(buyers.sum())}


def segmented_book(v, budget, seg, order, seed):
    """Posted price per segment, set from the most valuable segment down on the budgets still unspent; then buyers
    spend on their highest-surplus segments first (round-robin, one break at a time, seeded order)."""
    segs = list(order)
    E = np.stack([v[seg == s].mean(0) for s in segs])                     # segments × ads
    supply = np.array([(seg == s).sum() for s in segs])
    left = budget.astype(int).copy()
    price = np.zeros(len(segs))
    for i in range(len(segs)):
        p, sold = best_price(np.where(left > 0, E[i], 0), left, supply[i])
        price[i] = p
        if sold:
            take = np.where((E[i] >= p) & (left > 0), left, 0).astype(float)
            left = left - np.floor(take * sold / take.sum()).astype(int)
    stock = supply.copy()
    rem = budget.astype(int).copy()
    sold = np.zeros_like(stock)
    value = 0.0
    surplus = np.where((E >= price[:, None]) & (price[:, None] > 0), E - price[:, None], -1)
    prefs = [[i for i in np.argsort(-surplus[:, j], kind="stable") if surplus[i, j] >= 0] for j in range(len(budget))]
    order_b = np.random.default_rng(seed).permutation(len(budget))
    active = [j for j in order_b if prefs[j] and rem[j] > 0]
    while active:
        nxt = []
        for j in active:
            while prefs[j] and stock[prefs[j][0]] == 0:
                prefs[j].pop(0)
            if prefs[j]:
                i = prefs[j][0]
                stock[i] -= 1
                sold[i] += 1
                rem[j] -= 1
                value += E[i, j]
                if rem[j] > 0:
                    nxt.append(j)
        active = nxt
    return {"revenue": float((price * sold).sum()), "breaks_sold": int(sold.sum()), "value_delivered": value,
            "segments": {s: {"price": float(price[i]), "sold": int(sold[i]), "supply": int(supply[i]),
                             "fill_rate": float(sold[i] / supply[i])} for i, s in enumerate(segs)}}


def run():
    cfg, catalogue, brk, cls, context, relevant, allowed, ad_group, groups = signals()
    seg = np.array([f"{c}|{x}" for c, x in zip(cls, context)])
    names = sorted(set(seg), key=lambda s: (CLASSES.index(s.split("|")[0]), s))
    n = len(brk)
    print(f"{n} breaks; classes: " + ", ".join(f"{c} {np.mean(cls == c):.1%}" for c in CLASSES))
    print("contexts: " + ", ".join(f"{g} {np.mean(context == g):.1%}" for g in groups + ["general"]))

    def values(rule, base, m):
        ok = np.stack([allowed[r][:, j] for j, r in enumerate(rule)], 1)
        return ok * base[None, :] * np.where(relevant, m, 1.0)

    runs = []
    for seed in cfg["seeds"]:
        rule, base, budget = demand(cfg, catalogue, seed)
        for m in [cfg["premium"]] + cfg["premium_sensitivity"]:
            v = values(rule, base, m)
            # valuation ordering: mean value per break to the advertisers whose rule set allows the segment's class
            elig = {s: np.array([s.split("|")[0] in ALLOWED[r] for r in rule]) for s in names}
            val = {s: float(v[seg == s][:, elig[s]].mean()) if elig[s].any() else 0.0 for s in names}
            order = sorted(names, key=lambda s: -val[s])
            for book, mult in (("slack", 1), ("tight", cfg["tight_multiplier"])):
                b = budget * mult
                f = flat_book(v, b)
                s = segmented_book(v, b, seg, order, seed)
                runs.append({"seed": seed, "premium": m, "book": book, "demand_breaks": int(b.sum()),
                             "flat": f, "segmented": s, "valuation": val,
                             "eligible": {k: int(e.sum()) for k, e in elig.items()},
                             "matched": {k: int((elig[k] & (np.array(groups + ["general"])[ad_group]
                                                            == k.split("|")[1])).sum()) for k in names}})
    def pick(seed, m, book):
        return next(r for r in runs if r["seed"] == seed and r["premium"] == m and r["book"] == book)

    summary = {}
    for m in [cfg["premium"]] + cfg["premium_sensitivity"]:
        for book in ("slack", "tight"):
            rs = [pick(sd, m, book) for sd in cfg["seeds"]]
            ratio = [r["segmented"]["revenue"] / r["flat"]["revenue"] for r in rs]
            sold_ratio = [r["segmented"]["breaks_sold"] / max(r["flat"]["breaks_sold"], 1) for r in rs]
            vratio = [r["segmented"]["value_delivered"] / r["flat"]["value_delivered"] for r in rs]
            summary[f"m{m}_{book}"] = {
                "revenue_ratio": {"median": float(np.median(ratio)), "min": min(ratio), "max": max(ratio)},
                "seeds_segmented_higher": int(sum(x > 1 for x in ratio)),
                "breaks_sold_ratio": {"median": float(np.median(sold_ratio)), "min": min(sold_ratio),
                                      "max": max(sold_ratio)},
                "value_delivered_ratio": {"median": float(np.median(vratio)), "min": min(vratio), "max": max(vratio)}}
            print(f"m={m} {book:5s}: segmented/flat revenue {np.median(ratio):.2f} ({min(ratio):.2f}-{max(ratio):.2f}),"
                  f" higher in {sum(x > 1 for x in ratio)}/10 seeds; breaks sold {np.median(sold_ratio):.2f};"
                  f" advertiser value {np.median(vratio):.2f}")

    table = []
    r0s, r0t = pick(0, cfg["premium"], "slack"), pick(0, cfg["premium"], "tight")
    for s in sorted(names, key=lambda s: -r0s["valuation"][s]):
        table.append({"segment": s, "breaks": int((seg == s).sum()), "share": float((seg == s).mean()),
                      "eligible_advertisers": r0s["eligible"][s], "matched_advertisers": r0s["matched"][s],
                      "value_per_break": r0s["valuation"][s],
                      "price_slack": r0s["segmented"]["segments"][s]["price"],
                      "fill_slack": r0s["segmented"]["segments"][s]["fill_rate"],
                      "fill_tight": r0t["segmented"]["segments"][s]["fill_rate"]})
    print(f"\nseed 0, m={cfg['premium']}: {'segment':32s} breaks  share  elig match value  fill(slack) fill(tight)")
    for t in table:
        print(f"  {t['segment']:38s} {t['breaks']:5d} {t['share']:6.1%} {t['eligible_advertisers']:4d} "
              f"{t['matched_advertisers']:4d} {t['value_per_break']:6.2f} {t['fill_slack']:8.0%} {t['fill_tight']:9.0%}")
    restricted = cls == "restricted"
    rest = {"share": float(restricted.mean()), "breaks": int(restricted.sum()),
            "fill_slack": float(sum(r0s["segmented"]["segments"][s]["sold"] for s in names if s.startswith("restricted"))
                                / max(restricted.sum(), 1)),
            "fill_tight": float(sum(r0t["segmented"]["segments"][s]["sold"] for s in names if s.startswith("restricted"))
                                / max(restricted.sum(), 1))}
    print(f"\nrestricted: {rest['breaks']} breaks ({rest['share']:.1%}); fill slack {rest['fill_slack']:.0%}, "
          f"tight {rest['fill_tight']:.0%}")
    for book in ("slack", "tight"):
        r = pick(0, cfg["premium"], book)
        print(f"seed 0 {book}: flat price {r['flat']['price']:.2f}, sold {r['flat']['breaks_sold']}, revenue "
              f"{r['flat']['revenue']:.0f} | segmented sold {r['segmented']['breaks_sold']}, revenue "
              f"{r['segmented']['revenue']:.0f}")
    data = {"breaks": n, "classes": {c: float(np.mean(cls == c)) for c in CLASSES},
            "contexts": {g: float(np.mean(context == g)) for g in groups + ["general"]},
            "restricted": rest, "segments_seed0": table, "summary": summary, "runs": runs}
    write_result(PHASE, "inventory", LANE, 0, {"protocol": "EVAL-PROTOCOL.md (Phase 7 amendment)",
                                               "parameters": yaml.safe_load(CONFIG.read_text()),
                                               "tags": "5d (data/phase5/tags_5d.json)",
                                               "matcher": "frames+cards (Phase 4g)"}, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run"])
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
