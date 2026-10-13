"""E3: how well do audio tags separate labelled violence / horror / weapons / accidents scenes?"""
from phase5e_common import *
A = json.load(open(SP / "audio_dev.json"))
GROUPS = {"gun": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"],
          "hit": ["Slap, smack", "Whack, thwack", "Thump, thud", "Smash, crash"],
          "scream": ["Screaming", "Yell", "Battle cry"], "explosion": ["Explosion"], "fire": ["Fire", "Crackle"]}
g = lambda s, k: max(A[s].get(x, 0) for x in GROUPS[k])
from sklearn.metrics import roc_auc_score
for tag in ("violence", "weapons", "horror", "accidents", "death"):
    y, feats = [], {k: [] for k in GROUPS}
    for _, L, sc, *_ in POOLS:
        for i, lab in L.items():
            y.append(tag in lab)
            for k in GROUPS: feats[k].append(g(sc[i], k))
    print(tag, sum(y), "labelled; AUC " + ", ".join(f"{k} {roc_auc_score(y, v):.2f}" for k, v in feats.items()))
