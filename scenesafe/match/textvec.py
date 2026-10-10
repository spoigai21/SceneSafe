"""CLIP text embeddings computed in a child process: torch and faiss each bundle an OpenMP runtime and the
process aborts if both load (BUGS.md #7). Cached next to the list of texts they were made from."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scenesafe.common.results import REPO_ROOT

EMBED = """
import json, sys, numpy as np
from scenesafe.embed.clip import ClipEmbedder
texts = json.load(open(sys.argv[1]))
np.save(sys.argv[2], ClipEmbedder("mps").embed_text(texts).numpy().astype(np.float32))
"""


def embed_texts(texts: list[str], cache: Path) -> np.ndarray:
    """Unit-normalised CLIP text embeddings, one row per text. Recomputed whenever the texts change."""
    key = cache.with_suffix(".json")
    if not (cache.exists() and key.exists() and json.loads(key.read_text()) == texts):
        cache.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(texts))
        subprocess.run([sys.executable, "-c", EMBED, str(key), str(cache)], check=True, cwd=REPO_ROOT)
    out = np.load(cache)
    assert len(out) == len(texts)
    return out
