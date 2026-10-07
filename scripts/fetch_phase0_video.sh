#!/usr/bin/env bash
# Download the public-domain film used in Phase 0 and verify its size.
set -euo pipefail

URL="https://archive.org/download/Night_Of_The_Living_Dead_raw_HD_WS/NightOfTheLivingDeadRemasteredHD.mp4"
OUT="data/raw/notld_1968_1024x768.mp4"
EXPECTED_BYTES=1902907166

mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || curl -L --fail -o "$OUT" "$URL"

actual=$(wc -c < "$OUT" | tr -d ' ')
if [ "$actual" != "$EXPECTED_BYTES" ]; then
    echo "size mismatch: got $actual bytes, expected $EXPECTED_BYTES" >&2
    exit 1
fi
echo "ok: $OUT ($actual bytes)"
