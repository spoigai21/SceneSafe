"""FAISS indexes over unit vectors (inner product = cosine, since every vector is unit-normalised).

FAISS and PyTorch each bundle an OpenMP runtime and the process aborts on the first parallel call if both are
loaded, so nothing that imports torch may import this module. Result handling lives in ranking.py, which is
free of both.
"""

import math
from dataclasses import dataclass

import faiss
import numpy as np

KINDS = ("flat", "ivf", "ivfpq")


@dataclass
class IndexSpec:
    kind: str             # flat | ivf | ivfpq
    nlist: int = 0        # IVF: number of groups (k-means clusters)
    m: int = 0            # PQ: sub-vectors per embedding (bytes per code at 8 bits)

    @property
    def name(self) -> str:
        return {"flat": "flat", "ivf": f"ivf{self.nlist}",
                "ivfpq": f"ivf{self.nlist}-pq{self.m}"}[self.kind]


def default_nlist(n: int) -> int:
    """About 4 * sqrt(n) groups, rounded to a power of two (the usual FAISS starting point), capped so k-means has
    the 39 training points per group FAISS asks for."""
    return min(2 ** round(math.log2(4 * math.sqrt(n))), 2 ** int(math.log2(n / 39)))


def build(spec: IndexSpec, vectors: np.ndarray, seed: int = 0) -> faiss.Index:
    d = vectors.shape[1]
    if spec.kind == "flat":
        index = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
        if spec.kind == "ivf":
            index = faiss.IndexIVFFlat(quantizer, d, spec.nlist, faiss.METRIC_INNER_PRODUCT)
        elif spec.kind == "ivfpq":
            index = faiss.IndexIVFPQ(quantizer, d, spec.nlist, spec.m, 8, faiss.METRIC_INNER_PRODUCT)
            index.pq.cp.seed = seed
        else:
            raise ValueError(spec.kind)
        index.cp.seed = seed
        index.train(vectors)
    index.add(vectors)
    return index


def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = nprobe


def size_bytes(index: faiss.Index) -> int:
    return int(faiss.serialize_index(index).size)
