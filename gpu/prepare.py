"""Phase 6, NVIDIA lane: write gpu/sample.json and copy the sample's images into gpu/img/, so the rented machine needs
nothing but the bundle, the 3 films and the models. Runs on the laptop (needs the Phase 5 data).

The sample is the one the laptop was timed on (scripts/run_phase6.py, seed 0): the same 3 films and Whisper windows, the
same 60 scenes and their strip and grid images, the same 2 strip checks per scene, and every card text for the
embedding step. Image creation needs the films, and only 3 are shipped, so it is timed on a second sample: 60 scenes of
those 3 films, picked with the same seed (the laptop timed it on scenes from all 19 films; stated in the results).

    PYTHONPATH=.:scripts .venv/bin/python gpu/prepare.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_phase6 as r6                                     # noqa: E402
from run_phase4f import FRAMES                              # noqa: E402
from run_phase5 import GRIDS                                # noqa: E402
from scenesafe.match import cards as cd                    # noqa: E402

OUT = ROOT / "gpu"
IMG = OUT / "img"


def main(seed: int = 0):
    corpus, units, names, film, cards, pick = r6.scene_sample(seed)
    ids = r6.pick_films(seed)
    for sub in ("strips", "grids"):
        (IMG / sub).mkdir(parents=True, exist_ok=True)
    scenes = []
    for u in pick:
        n = names[u]
        shutil.copyfile(FRAMES / f"{n}.jpg", IMG / "strips" / f"{n}.jpg")
        shutil.copyfile(GRIDS / f"{n}.jpg", IMG / "grids" / f"{n}.jpg")
        scenes.append({"name": n, "film": film[u], "start": float(units.start[u]), "end": float(units.end[u]),
                       "dialogue": cards[n]["dialogue"], "strip": f"gpu/img/strips/{n}.jpg",
                       "grid": f"gpu/img/grids/{n}.jpg"})
    in_films = [u for u in range(len(names)) if film[u] in ids]
    img_pick = sorted(np.random.default_rng(seed).choice(in_films, r6.N_SCENES, replace=False).tolist())
    image_scenes = [{"name": names[u], "film": film[u], "start": float(units.start[u]), "end": float(units.end[u])}
                    for u in img_pick]
    films = {}
    for m in ("manifest_4d.json", "manifest_4h.json"):
        for t in json.loads((ROOT / "corpus" / m).read_text())["titles"]:
            if t["id"] in ids:
                films[t["id"]] = {k: t[k] for k in ("url", "bytes", "md5", "duration_s")}
    assert sorted(films) == ids, f"films missing from the manifests: {set(ids) - set(films)}"
    sample = {"seed": seed, "films": films, "whisper_window_s": list(r6.WINDOW), "runs": r6.RUNS,
              "scenes": scenes, "strip_check_tags": list(r6.STRIP_TAGS), "image_scenes": image_scenes,
              "card_texts": [cd.card_text(cards[n]["card"]) for n in names]}
    (OUT / "sample.json").write_text(json.dumps(sample, indent=1) + "\n")
    print(f"sample.json: {len(scenes)} scenes, {len(image_scenes)} image scenes from {ids}, "
          f"{len(sample['card_texts'])} card texts")


if __name__ == "__main__":
    main()
