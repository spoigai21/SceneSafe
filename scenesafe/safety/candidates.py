"""Phase 5c: more candidate tags for the verifier to check. A scene-tag pair is a candidate if the first-pass detector
gave it, if the scene card mentions it (keywords), or if CLIP ranks the scene in the top 5% for that tag. Everything
here is cheap; the verifier decides."""

import re

import numpy as np

CLIP_SHARE = 0.05

KEYWORDS = {
    "violence": r"\b(fight|fighting|punch|brawl|struggl|attack|shoot|shooting|shot|hit|hitting|grappl|wrestl|beat|"
                r"gunfight|battle|duel)",
    "weapons": r"\b(gun|guns|rifle|pistol|revolver|knife|sword|weapon|shotgun|bayonet|spear|dagger|cannon|firearm)",
    "accidents": r"\b(crash|fire|flames|explosion|explod|burning|sinking|wreck|smoke|collision|disaster|falling)",
    "death": r"\b(dead|body|corpse|funeral|grave|coffin|killed|killing|murder|death|dying|skeleton)",
    "horror": r"\b(monster|creature|ghost|skeleton|skull|scream|zombie|vampire|demon|alien)",
    "alcohol": r"\b(whisk|beer|wine|liquor|bottle|drinking|drunk|saloon|bar|cocktail|champagne|brandy|gin)",
    "smoking": r"\b(cigarette|cigar|pipe|smoking|smoke)",
    "sexual": r"\b(nude|naked|topless|undress)",
    "drugs": r"\b(drug|opium|cocaine|heroin|syringe)",
}

CLIP_PROMPTS = {
    "violence": ["people fighting", "a man punching another man", "a fistfight", "people shooting at each other",
                 "a violent struggle"],
    "weapons": ["a person holding a gun", "a man pointing a pistol", "a man with a rifle", "a person holding a knife",
                "a sword fight"],
    "accidents": ["a car crash", "a fire burning", "an explosion", "a ship sinking", "a burning building"],
    "death": ["a dead body lying on the ground", "a corpse", "a funeral", "a grave in a cemetery", "a coffin"],
    "horror": ["a monster", "a scary creature", "a ghost", "a skeleton", "a woman screaming in terror"],
    "alcohol": ["people drinking alcohol", "a man drinking whiskey", "a bar with bottles of liquor", "a drunk man",
                "people toasting with wine"],
    "smoking": ["a man smoking a cigarette", "a person smoking a pipe", "a man smoking a cigar", "cigarette smoke",
                "a woman smoking"],
    "sexual": ["a naked person", "nudity", "a couple in bed"],
    "drugs": ["drug use", "a syringe"],
}


def keyword_tags(card: str) -> set[str]:
    low = card.lower()
    return {t for t, p in KEYWORDS.items() if re.search(p, low)}


def clip_tags(scores: np.ndarray, tags: list[str], share: float = CLIP_SHARE) -> list[set[str]]:
    """scores: scenes × tags (best prompt per tag). A scene is a candidate for a tag in that tag's top share."""
    cut = np.quantile(scores, 1 - share, axis=0)
    return [{t for j, t in enumerate(tags) if row[j] >= cut[j]} for row in scores]
