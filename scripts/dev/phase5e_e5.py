"""E5: the 8B model asked violence (obvious definition, grid, P(yes)) on every labelled scene."""
from phase5e_common import *
from scenesafe.match import cards as cd
cd.CARD_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
items = [(sc[i], "violence", r5.GRIDS / f"{sc[i]}.jpg", dl[i]) for _, L, sc, dl, *_ in POOLS for i in L]
p = verify_p(items, SP / "verify_violence_8b.json", GRID_OBVIOUS_PROMPT, TAGS_OBVIOUS)
p4 = json.load(open(SP / "verify_all_dev.json"))
pos = []; neg8 = []; neg4 = []
for _, L, sc, *_ in POOLS:
    for i, lab in L.items():
        s = sc[i]
        if "violence" in lab: pos.append((s, round(p4[f"{s}|violence"], 2), round(p[f"{s}|violence"], 2)))
        else: neg8.append(p[f"{s}|violence"]); neg4.append(p4[f"{s}|violence"])
print("labelled violence (scene, 4B, 8B):"); [print("  ", x) for x in pos]
for th in (0.3, 0.5, 0.7, 0.9):
    print(f"  @{th}: caught 4B {sum(a >= th for _, a, _ in pos)}/{len(pos)} (false {sum(x >= th for x in neg4)}), "
          f"8B {sum(b >= th for _, _, b in pos)}/{len(pos)} (false {sum(x >= th for x in neg8)}) of {len(neg8)} non-violent")
pc = json.load(open(D / "verify_5d_dev.json"))
gw = lambda s: {t for t in ("weapons",) if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
g3 = lambda s: {t for t in ("horror", "smoking", "alcohol") if t in first[s] | strip[s] and pc.get(f"{s}|{t}", 0) >= 0.7}
base = lambda s: v5b[s] | gw(s) | g3(s)
evaluate("E1 best", base, False)
for th in (0.5, 0.7, 0.9):
    evaluate(f"E1 + 8B violence on every scene @{th}", lambda s, th=th: base(s) | ({"violence"} if p[f"{s}|violence"] >= th else set()))
