"""The simulated ad catalogue and the Phase 4 film split (both frozen in EVAL-PROTOCOL.md)."""

from dataclasses import dataclass

import yaml

from scenesafe.common.results import REPO_ROOT
from scenesafe.corpus import titles

ADS = REPO_ROOT / "configs" / "ads.yaml"
SPLIT = REPO_ROOT / "configs" / "phase4_split.yaml"


@dataclass(frozen=True)
class Ad:
    id: str
    category: str
    brief: str
    genres: tuple[str, ...]
    avoid: tuple[str, ...]


def ads() -> list[Ad]:
    return [Ad(a["id"], a["category"], a["brief"], tuple(a["genres"]), tuple(a["avoid"]))
            for a in yaml.safe_load(ADS.read_text())["ads"]]


def test_films() -> list[str]:
    split = yaml.safe_load(SPLIT.read_text())["test"]
    return sorted(i for ids in split.values() for i in ids)


def practice_films() -> list[str]:
    test = set(test_films())
    return [t["id"] for t in titles() if t["id"] not in test]


def genre_of() -> dict[str, str]:
    return {t["id"]: t["genre"] for t in titles()}
