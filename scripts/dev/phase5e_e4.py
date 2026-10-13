"""E4: violence from shootouts. On scenes tagged weapons, (a) ask 'is a gun fired / someone shot at', (b) gunshot audio."""
from phase5e_common import *
A = json.load(open(SP / "audio_dev.json"))
gun = lambda s: max(A[s].get(x, 0) for x in ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"])
pc = json.load(open(D / "verify_5d_dev.json"))
gw = lambda s: {t for t in ("weapons",) if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
g3 = lambda s: {t for t in ("horror", "smoking", "alcohol") if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
base = lambda s: v5b[s] | gw(s) | g3(s)
FIRE = {"fired": "a gun or other weapon is being fired, or someone is being shot at"}
PROMPT = GRID_OBVIOUS_PROMPT
items = []
for _, L, sc, dl, *_ in POOLS:
    for i in L:
        s = sc[i]
        if "weapons" in base(s):
            items.append((s, "fired", r5.GRIDS / f"{s}.jpg", dl[i]))
pf = verify_p(items, SP / "verify_fired.json", PROMPT, FIRE)
evaluate("E1 best", base)
for th in (0.3, 0.5, 0.7):
    evaluate(f"E1 + fired question @{th} -> violence",
             lambda s, th=th: base(s) | ({"violence"} if "weapons" in base(s) and pf.get(f"{s}|fired", 0) >= th else set()))
for th in (0.05, 0.1, 0.2, 0.4):
    evaluate(f"E1 + weapons & gunshot audio @{th} -> violence",
             lambda s, th=th: base(s) | ({"violence"} if "weapons" in base(s) and gun(s) >= th else set()))
print("weapons-flagged scenes:", len(items))
for _, L, sc, *_ in POOLS:
    for i, lab in L.items():
        s = sc[i]
        if "weapons" in base(s):
            print(f"  {s:40s} label {sorted(lab)}  fired {pf[f'{s}|fired']:.2f}  gun-audio {gun(s):.2f}")
