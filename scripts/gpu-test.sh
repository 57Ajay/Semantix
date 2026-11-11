#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SEMANTIX_SOURCE:-git+https://github.com/57Ajay/Semantix}"
BACKEND="${UV_TORCH_BACKEND:-auto}"

if ! command -v uv >/dev/null 2>&1; then
  if command -v pip >/dev/null 2>&1; then
    pip install --quiet uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

echo "== Hardware"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "nvidia-smi not found (expected on Apple Silicon and CPU-only machines)"
fi

echo "== Installing semantix (torch backend: $BACKEND)"
uv tool install --force --refresh-package semantix --torch-backend "$BACKEND" "$SOURCE"
SEMANTIX="$(uv tool dir --bin)/semantix"
PYTHON="$(uv tool dir)/semantix/bin/python"

echo "== PyTorch"
"$PYTHON" - <<'PY'
import torch

mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available(), "| Apple Metal available:", mps)
PY

if command -v nvidia-smi >/dev/null 2>&1 && ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo
  echo "A GPU is present but PyTorch cannot use it. The installed build probably does not match the"
  echo "driver. Set an explicit backend, for example UV_TORCH_BACKEND=cu126, and run this script again."
  echo "Continuing on the CPU."
fi

echo "== Benchmark"
"$SEMANTIX" benchmark "$@"

echo
echo "== Report (copy everything between the lines)"
echo "-----BEGIN SEMANTIX REPORT-----"
cat semantix-benchmark-*.json
echo "-----END SEMANTIX REPORT-----"
