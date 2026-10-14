#!/usr/bin/env bash
# Phase 6, NVIDIA lane: build the bundle for the rented machine (run on the laptop, from the repo root).
#   bash gpu/bundle.sh               small bundle (code, sample, images); the machine downloads the films itself
#   bash gpu/bundle.sh --with-films  also ships the 3 films (about 1.5 GB), if the Internet Archive is slow
# Output: $OUT (default ~/gpu_bundle.tar), outside the repo so it is never committed by accident.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-$HOME/gpu_bundle.tar}"
PYTHONPATH=.:scripts .venv/bin/python gpu/prepare.py
files=(scenesafe gpu/run_gpu.py gpu/setup.sh gpu/RUNBOOK.md gpu/sample.json gpu/img)
if [ "${1:-}" = "--with-films" ]; then
  while read -r tid; do files+=("data/corpus/$tid.mp4"); done < <(python3 -c \
    "import json; print('\n'.join(json.load(open('gpu/sample.json'))['films']))")
fi
COPYFILE_DISABLE=1 tar --exclude '__pycache__' --exclude '.DS_Store' -cf "$OUT" "${files[@]}"
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
