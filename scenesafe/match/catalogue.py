"""The simulated ad catalogue, the Phase 4d label guide and the Phase 4 film split (all frozen in EVAL-PROTOCOL.md)."""

from dataclasses import dataclass

import yaml

from scenesafe.common.results import REPO_ROOT
from scenesafe.corpus import titles

ADS = REPO_ROOT / "configs" / "ads.yaml"
ADS_EXTRA = REPO_ROOT / "configs" / "ads_extra.yaml"       # Phase 4d's 40 more ads
GUIDE = REPO_ROOT / "configs" / "label_guide.yaml"
THEMES = REPO_ROOT / "configs" / "themes.yaml"            # Phase 4e
SPLIT = REPO_ROOT / "configs" / "phase4_split.yaml"


@dataclass(frozen=True)
class Ad:
    id: str
    category: str
    brief: str
    genres: tuple[str, ...]
    avoid: tuple[str, ...]


def ads(path=ADS) -> list[Ad]:
    return [Ad(a["id"], a["category"], a["brief"], tuple(a["genres"]), tuple(a["avoid"]))
            for a in yaml.safe_load(path.read_text())["ads"]]


def ads_4d() -> list[Ad]:
    """The 60 ads of Phase 4d: the original 20, then the 40 extra."""
    return ads() + ads(ADS_EXTRA)


def label_guide() -> dict[str, dict[str, str]]:
    """Ad id -> {"fits": ..., "not": ...} (Phase 4d)."""
    return yaml.safe_load(GUIDE.read_text())["guide"]


def themes() -> dict[str, dict]:
    """Ad id -> {"theme": what counts as a thematic fit, "query": 3 phrases for the theme system} (Phase 4e)."""
    return yaml.safe_load(THEMES.read_text())["themes"]


def test_films() -> list[str]:
    split = yaml.safe_load(SPLIT.read_text())["test"]
    return sorted(i for ids in split.values() for i in ids)


def practice_films() -> list[str]:
    test = set(test_films())
    return [t["id"] for t in titles() if t["id"] not in test]


def genre_of() -> dict[str, str]:
    return {t["id"]: t["genre"] for t in titles()}
