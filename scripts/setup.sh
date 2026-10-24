#!/usr/bin/env bash
# Minimal setup: system deps, uv venv with vllm==0.32.0, model weights (INT8 W8A8).
#   scripts/setup.sh [--download-int8 | --convert]      (default: --convert)
#   --convert        download the official BF16 weights (CMKL/MANGO1.5-Qwen3.5-9B) and build the INT8 checkpoint locally
#                    with scripts/quant_w8a8_int8.py (CPU only, ~3 min, ~5 GB RAM)
#   --download-int8  download our pre-built INT8 checkpoint from Hugging Face instead (see docs/README.md for the link)
# Env: GM_VENV (default <repo>/.venv-vllm), GM_MODELS (default <repo>/models), GM_HF_INT8 (HF repo id of the INT8 weights),
#      HF_TOKEN only if the repo is private (never stored by this script).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${GM_VENV:-$ROOT/.venv-vllm}"
MODELS="${GM_MODELS:-$ROOT/models}"
HF_INT8="${GM_HF_INT8:-Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8}"
BASE_REPO=CMKL/MANGO1.5-Qwen3.5-9B
VLLM_VERSION=0.32.0
PY_VERSION=3.12
MODE=${1:---convert}

# 1. system dependencies (Ubuntu 24.04; skipped when not root and sudo is missing)
SUDO=""; [ "$(id -u)" != 0 ] && command -v sudo >/dev/null && SUDO=sudo
if command -v apt-get >/dev/null; then
  $SUDO apt-get update -y && $SUDO apt-get install -y git curl build-essential pkg-config ca-certificates python3
fi
nvidia-smi -L || echo "WARNING: no NVIDIA GPU visible (needed to serve, not to convert)"

# 2. uv + venv + vLLM
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$VENV/bin/python" ] || uv venv --python "$PY_VERSION" "$VENV"
uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto \
  || uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
uv pip install --python "$VENV/bin/python" -q safetensors numpy compressed-tensors huggingface_hub httpx nvidia-ml-py
"$VENV/bin/python" -I -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'

# 3. weights
mkdir -p "$MODELS"
HF="$VENV/bin/hf"
case "$MODE" in
  --download-int8)
    case "$HF_INT8" in *"{{"*) echo "HF INT8 repo not set: export GM_HF_INT8=<user>/<repo> or use --convert"; exit 1 ;; esac
    "$HF" download "$HF_INT8" --local-dir "$MODELS/w8a8-int8" ;;
  --convert)
    "$HF" download "$BASE_REPO" --local-dir "$MODELS/bf16-cmkl"
    "$VENV/bin/python" "$ROOT/scripts/quant_w8a8_int8.py" "$MODELS/bf16-cmkl" "$MODELS/w8a8-int8" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "done. Serve with: scripts/serve_final.sh start   (model dir $MODELS/w8a8-int8)"
