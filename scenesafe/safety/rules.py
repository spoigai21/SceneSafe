"""Rule sets over content tags, and a greedy book allocation (EVAL-PROTOCOL.md, "Phase 5 amendment").

A break sits between two consecutive scenes of one film and is checked against both of them."""

import numpy as np
import yaml

from scenesafe.common.results import REPO_ROOT

SAFETY = REPO_ROOT / "configs" / "safety"
RULE_SETS = ("lenient", "standard", "strict")


def rule_set(name: str) -> dict:
    return yaml.safe_load((SAFETY / f"{name}.yaml").read_text())


def adjacency() -> dict[str, list[str]]:
    return yaml.safe_load((SAFETY / "adjacency.yaml").read_text())["adjacency"]


def breaks(films: list[str]) -> list[tuple[int, int]]:
    """films: the film of each scene, scenes in time order within each film. Returns (s, s + 1) for each break."""
    return [(s, s + 1) for s in range(len(films) - 1) if films[s] == films[s + 1]]


def blocked_for_all(tags: list[set], block: set) -> np.ndarray:
    """Per scene: does it carry any tag in block?"""
    return np.array([bool(t & block) for t in tags])


def sellable(tags: list[set], brk: list[tuple[int, int]], ads, block: set) -> np.ndarray:
    """(breaks × ads) bool: True where no rule removes the break for that ad. Rules: category avoidance (block),
    adjacency by the ad's category, and the ad's own avoid list; each checked against both neighbouring scenes."""
    adj = adjacency()
    near = [tags[a] | tags[b] for a, b in brk]
    out = np.ones((len(brk), len(ads)), bool)
    for j, ad in enumerate(ads):
        avoid = block | set(adj.get(ad.category, [])) | set(ad.avoid)
        out[:, j] = [not (n & avoid) for n in near]
    return out


def allocate(ok: np.ndarray, score: np.ndarray, demand: int, seed: int = 0) -> np.ndarray:
    """Greedy rounds: each round, ads in a fixed random order each take their best-scoring free break that is
    allowed (ok, breaks × ads, already restricted to relevant breaks). Returns breaks sold per ad."""
    n_brk, n_ads = ok.shape
    order = np.random.default_rng(seed).permutation(n_ads)
    ranked = [[b for b in np.argsort(-score[:, j], kind="stable") if ok[b, j]] for j in range(n_ads)]
    pos, sold, free = [0] * n_ads, np.zeros(n_ads, int), np.ones(n_brk, bool)
    active = [j for j in order if ranked[j]]
    while active:
        nxt = []
        for j in active:
            lst = ranked[j]
            while pos[j] < len(lst) and not free[lst[pos[j]]]:
                pos[j] += 1
            if pos[j] < len(lst):
                free[lst[pos[j]]] = False
                sold[j] += 1
                if sold[j] < demand:
                    nxt.append(j)
        active = nxt
    return sold
