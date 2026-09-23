#!/usr/bin/env bash
set -euo pipefail

PYTHON="${P31_HF_PYTHON:-/home/yueqi/gpu-atlas/framework-runtimes/hf-live/bin/python}"
[[ -x "$PYTHON" ]] || {
  echo "P31 Hugging Face Python is unavailable: $PYTHON" >&2
  exit 2
}
exec "$PYTHON" "$@"
