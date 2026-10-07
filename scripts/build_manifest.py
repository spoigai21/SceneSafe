"""Select the corpus from the Internet Archive's public-domain feature films and write corpus/manifest.json.

Rules: public-domain licence, 60-120 min, one H.264 MP4 between 360p and 720p, a genre from a fixed
list (first match in GENRES order), at most PER_GENRE titles per genre, no exploitation or documentaries.
Popularity rankings change over time, so the committed manifest (not this script) is the source of truth.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import truststore

truststore.inject_into_ssl()  # use the OS certificate store (works behind TLS-inspecting proxies)

SEARCH_ROWS = 600
PER_GENRE = 5
MIN_S, MAX_S = 60 * 60, 120 * 60
MIN_H, MAX_H = 360, 720

# Controlled genre vocabulary: label -> subject keywords. Order sets precedence.
GENRES = {
    "horror": ["horror", "vampire", "zombie", "monster"],
    "scifi": ["sci-fi", "scifi", "science fiction", "space"],
    "western": ["western"],
    "crime": ["noir", "film-noir", "crime", "mystery", "thriller", "suspense"],
    "comedy": ["comedy"],
    "adventure": ["adventure", "action", "war"],
    "drama": ["drama", "romance"],
}
EXCLUDE = re.compile(r"exploitation|nudis|burlesque|stripper|sex|documentary|grindhouse|double feature|short|silent",
                     re.IGNORECASE)

# Excluded by hand, with the reason.
EXCLUDE_IDS = {
    "ChildBride": "exploitation film containing a nude scene with a child actor",
}

OUT = Path(__file__).resolve().parents[1] / "corpus" / "manifest.json"


def get_json(url: str) -> dict:
    with urlopen(url, timeout=60) as r:
        return json.load(r)


def search() -> list[dict]:
    q = urlencode([("q", "collection:feature_films AND mediatype:movies AND licenseurl:*publicdomain*"),
                   ("fl[]", "identifier"), ("fl[]", "title"), ("fl[]", "subject"), ("fl[]", "year"),
                   ("sort[]", "downloads desc"), ("rows", SEARCH_ROWS), ("output", "json")])
    return get_json(f"https://archive.org/advancedsearch.php?{q}")["response"]["docs"]


def field(doc: dict, key: str) -> str:
    v = doc.get(key, "")
    return v if isinstance(v, str) else "; ".join(map(str, v))


def subjects(doc: dict) -> str:
    return field(doc, "subject")


def genre_of(subject_text: str) -> str | None:
    low = subject_text.lower()
    for genre, words in GENRES.items():
        if any(w in low for w in words):
            return genre
    return None


def pick_file(identifier: str) -> dict | None:
    meta = get_json(f"https://archive.org/metadata/{identifier}")
    best = None
    for f in meta.get("files", []):
        if not f["name"].lower().endswith(".mp4") or f.get("format") not in ("h.264", "MPEG4", "h.264 IA"):
            continue
        try:
            h, length, size = int(f["height"]), float(f["length"]), int(f["size"])
        except (KeyError, ValueError):
            continue
        if not (MIN_H <= h <= MAX_H and MIN_S <= length <= MAX_S):
            continue
        if best is None or h > best["height"]:
            best = {"file": f["name"], "bytes": size, "width": int(f.get("width", 0)), "height": h,
                    "duration_s": length, "md5": f.get("md5")}
    return best


def main():
    candidates = []
    for doc in search():
        text = subjects(doc) + " " + field(doc, "title")
        genre = genre_of(subjects(doc))
        if genre and not EXCLUDE.search(text) and doc["identifier"] not in EXCLUDE_IDS:
            candidates.append((doc, genre))

    with ThreadPoolExecutor(8) as pool:
        files = list(pool.map(lambda c: pick_file(c[0]["identifier"]), candidates))

    counts, chosen, seen_titles = {g: 0 for g in GENRES}, [], set()
    for (doc, genre), f in zip(candidates, files):
        title_key = re.sub(r"\W", "", field(doc, "title").lower())
        if f is None or counts[genre] >= PER_GENRE or title_key in seen_titles:
            continue
        counts[genre] += 1
        seen_titles.add(title_key)
        chosen.append({"id": doc["identifier"], "title": field(doc, "title"), "year": doc.get("year"),
                       "genre": genre, "subjects": subjects(doc),
                       "url": f"https://archive.org/download/{doc['identifier']}/{f['file']}", **f})

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"rules": __doc__.strip(), "excluded_by_hand": EXCLUDE_IDS, "titles": chosen}, indent=2) + "\n")
    hours = sum(t["duration_s"] for t in chosen) / 3600
    gb = sum(t["bytes"] for t in chosen) / 1e9
    print(f"{len(chosen)} titles, {hours:.1f} h, {gb:.1f} GB, per genre {counts}")


if __name__ == "__main__":
    main()
