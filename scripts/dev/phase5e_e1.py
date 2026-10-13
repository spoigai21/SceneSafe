"""E1: add grid-checked tags (beyond weapons) to 5b."""
from phase5e_common import *
items = []
for _, L, sc, dl, *_ in POOLS:
    for i in L:
        s = sc[i]
        items += [(s, t, r5.GRIDS / f"{s}.jpg", dl[i]) for t in TAGS if t in first[s] | strip[s]]
p = verify_p(items, D / "verify_5d_dev.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
def grid(tags, th):
    return lambda s: {t for t in tags if t in first[s] | strip[s] and p.get(f"{s}|{t}", 0) >= th}
evaluate("5b", lambda s: v5b[s])
evaluate("5d (5b + grid weapons 0.7)", lambda s: t5d[s])
for extra in (("horror",), ("horror", "violence"), ("horror", "smoking", "alcohol"), ("horror", "violence", "smoking", "alcohol")):
    for th in (0.3, 0.5, 0.7):
        g = grid(("weapons",) + extra, th); gw = grid(("weapons",), 0.7)
        evaluate(f"5d + grid {'+'.join(extra)} @{th}", lambda s, g=g, gw=gw: v5b[s] | gw(s) | g(s))
