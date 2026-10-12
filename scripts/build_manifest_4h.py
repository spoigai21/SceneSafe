"""Select the Phase 4h confirmation films and write corpus/manifest_4h.json (EVAL-PROTOCOL.md, "Phase 4h
amendment").

Same rules as build_manifest.py (public domain, 60-120 min, one H.264 MP4 between 360p and 720p, the same genre
vocabulary and exclusions), applied to a fresh search. Titles already in corpus/manifest.json or
corpus/manifest_4d.json (by id or by title) are skipped, and the first PER_GENRE remaining titles per genre, in
download-count order, are taken. The committed manifest, not this script, is the source of truth.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

from scripts.build_manifest import EXCLUDE, EXCLUDE_IDS, GENRES, field, genre_of, pick_file, search, subjects
from scenesafe.corpus import MANIFEST

PER_GENRE = 1
OUT = MANIFEST.with_name("manifest_4h.json")
USED = (MANIFEST, MANIFEST.with_name("manifest_4d.json"))


def key(title: str) -> str:
    return re.sub(r"\W", "", title.lower())


def main():
    existing = [t for m in USED for t in json.loads(m.read_text())["titles"]]
    seen_ids = {t["id"] for t in existing}
    seen_titles = {key(t["title"]) for t in existing}
    candidates = []
    for doc in search():
        text = subjects(doc) + " " + field(doc, "title")
        genre = genre_of(subjects(doc))
        if (genre and not EXCLUDE.search(text) and doc["identifier"] not in EXCLUDE_IDS
                and doc["identifier"] not in seen_ids and key(field(doc, "title")) not in seen_titles):
            candidates.append((doc, genre))

    with ThreadPoolExecutor(8) as pool:
        files = list(pool.map(lambda c: pick_file(c[0]["identifier"]), candidates))

    counts, chosen = {g: 0 for g in GENRES}, []
    for (doc, genre), f in zip(candidates, files):
        tk = key(field(doc, "title"))
        if f is None or counts[genre] >= PER_GENRE or tk in seen_titles:
            continue
        counts[genre] += 1
        seen_titles.add(tk)
        chosen.append({"id": doc["identifier"], "title": field(doc, "title"), "year": doc.get("year"),
                       "genre": genre, "subjects": subjects(doc),
                       "url": f"https://archive.org/download/{doc['identifier']}/{f['file']}", **f})

    OUT.write_text(json.dumps({"rules": __doc__.strip(), "excluded_by_hand": EXCLUDE_IDS, "titles": chosen}, indent=2) + "\n")
    hours = sum(t["duration_s"] for t in chosen) / 3600
    gb = sum(t["bytes"] for t in chosen) / 1e9
    print(f"{len(chosen)} titles, {hours:.1f} h, {gb:.1f} GB, per genre {counts}")
    for t in chosen:
        print(f"  {t['genre']:9s} {t['id']}  ({t['title']}, {t.get('year')})")


if __name__ == "__main__":
    main()
