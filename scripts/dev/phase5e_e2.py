"""E2: ask violence and horror (obvious definitions, grid, P(yes)) on every labelled scene, not only candidates."""
from phase5e_common import *
ALL = ("violence", "horror", "weapons", "smoking", "alcohol")
items = []
for _, L, sc, dl, *_ in POOLS:
    for i in L:
        s = sc[i]
        items += [(s, t, r5.GRIDS / f"{s}.jpg", dl[i]) for t in ALL]
p = verify_p(items, SP / "verify_all_dev.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
pc = json.load(open(D / "verify_5d_dev.json"))
gw = lambda s: {t for t in ("weapons",) if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
g3 = lambda s: {t for t in ("horror", "smoking", "alcohol") if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
base = lambda s: v5b[s] | gw(s) | g3(s)
evaluate("E1 best (5b + grid weapons/horror/smoking/alcohol @0.7)", base)
for tags in (("violence",), ("horror",), ("violence", "horror"), ("violence", "horror", "weapons", "smoking", "alcohol")):
    for th in (0.3, 0.5, 0.7, 0.9):
        f = lambda s, tags=tags, th=th: base(s) | {t for t in tags if p[f"{s}|{t}"] >= th}
        evaluate(f"E1 + ask-all {'+'.join(tags)} @{th}", f)
# how separable is P(violence)?
for t in ("violence", "horror"):
    pos, neg = [], []
    for _, L, sc, *_ in POOLS:
        for i, lab in L.items():
            (pos if t in lab else neg).append(p[f"{sc[i]}|{t}"])
    pos, neg = np.array(pos), np.array(neg)
    print(t, "labelled:", np.round(np.sort(pos), 2), " unlabelled >0.5:", int((neg > 0.5).sum()), "of", len(neg))
