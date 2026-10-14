#!/usr/bin/env bash
# Phase 6, NVIDIA lane: set up a fresh Lambda Cloud 1x A10 instance (Ubuntu + Lambda Stack) to run gpu/run_gpu.py.
# Run from the unpacked bundle directory: bash gpu/setup.sh
#   1. downloads the 3 films from the Internet Archive (in the background, checked by size and MD5)
#   2. makes a Python 3.11 venv with uv and installs pinned packages (vLLM picks the PyTorch build for this driver)
#   3. pre-downloads every model, so no download is timed
# Writes gpu/env.sh, which the run commands source.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
VENV="$HOME/ss-venv"

echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== films (background)"
mkdir -p data/corpus
python3 - <<'EOF' > /tmp/films.txt
import json
for tid, f in json.load(open("gpu/sample.json"))["films"].items():
    print(tid, f["url"], f["bytes"], f["md5"])
EOF
if ls data/corpus/*.mp4 >/dev/null 2>&1; then echo "films already present (shipped in the bundle)"; fi
while read -r tid url bytes md5; do
  out="data/corpus/$tid.mp4"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$bytes" ]; then continue; fi
  (curl -sSL --fail --retry 5 --retry-all-errors --retry-delay 10 -C - -o "$out" "$url" && echo "  downloaded $tid") &
done < /tmp/films.txt

echo "== uv + venv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 "$VENV"
# One resolve for everything. vLLM 0.31.0 pins torch 2.13.0 and transformers <5.18; --torch-backend=auto picks the
# CUDA build of torch that this machine's driver supports.
uv pip install --python "$VENV/bin/python" --torch-backend=auto \
  "vllm==0.31.0" "transformers>=5.10.4,<5.18" "faster-whisper==1.2.1" "ctranslate2==4.8.3" \
  "open_clip_torch==3.3.0" "sentence-transformers==6.1.0" "av==18.1.0" "pyyaml" "pillow" "numpy"
# CTranslate2 (faster-whisper) loads CUDA 12 cuBLAS and cuDNN 9 at run time; keep them apart from torch's own libs.
uv pip install --python "$VENV/bin/python" --target "$HOME/ct2libs" "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"

cat > gpu/env.sh <<EOF
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$HOME/ct2libs/nvidia/cublas/lib:$HOME/ct2libs/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"
EOF
set +u; source gpu/env.sh; set -u

echo "== models (pre-download, not timed)"
python - <<'EOF'
from huggingface_hub import snapshot_download
import open_clip
from faster_whisper import download_model
from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED
from scenesafe.match.dialogue import TEXT_MODEL
snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
snapshot_download(TEXT_MODEL)
download_model("large-v3-turbo")
open_clip.create_model_and_transforms(DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED)
print("models ready")
EOF

echo "== waiting for films"
wait
while read -r tid url bytes md5; do
  out="data/corpus/$tid.mp4"
  [ "$(stat -c %s "$out")" = "$bytes" ] || { echo "SIZE MISMATCH $tid"; exit 1; }
  [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ] || { echo "MD5 MISMATCH $tid"; exit 1; }
  echo "  ok $tid"
done < /tmp/films.txt
echo "SETUP OK. Next: source gpu/env.sh && python gpu/run_gpu.py smoke"
