"""Download every title in corpus/manifest.json to data/corpus/<id>.mp4 and verify size and MD5.

Skips files that are already present and verified, so it can be re-run after an interruption.
"""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.json"
DEST = ROOT / "data" / "corpus"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, title: dict) -> bool:
    return (path.exists() and path.stat().st_size == title["bytes"]
            and (title.get("md5") is None or md5(path) == title["md5"]))


def fetch(title: dict) -> str:
    path = DEST / f"{title['id']}.mp4"
    if verified(path, title):
        return f"ok (cached) {title['id']}"
    # curl uses the OS certificate store and resumes partial downloads with -C -
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "--retry-delay", "15",
                    "-C", "-", "-o", str(path), title["url"]], check=True)
    if not verified(path, title):
        raise RuntimeError(f"verification failed: {title['id']}")
    return f"ok {title['id']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", help="only these title ids")
    ap.add_argument("--manifest", default=str(MANIFEST), help="e.g. corpus/manifest_4d.json")
    args = ap.parse_args()

    titles = json.loads(Path(args.manifest).read_text())["titles"]
    if args.ids:
        titles = [t for t in titles if t["id"] in set(args.ids)]
    DEST.mkdir(parents=True, exist_ok=True)
    def safe_fetch(title):
        try:
            return fetch(title)
        except (subprocess.CalledProcessError, RuntimeError) as e:  # one bad title must not stop the rest
            return f"FAILED {title['id']}: {e}"

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(safe_fetch, titles))
    for msg in results:
        print(msg, flush=True)
    failed = [m for m in results if m.startswith("FAILED")]
    if failed:
        raise SystemExit(f"{len(failed)} title(s) failed; re-run to resume")


if __name__ == "__main__":
    main()
