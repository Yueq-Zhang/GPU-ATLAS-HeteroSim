#!/usr/bin/env bash
set -euo pipefail

profile="${1:-all}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${HETEROSIM_FRAMEWORK_ROOT:-${HOME}/.local/share/gpu-atlas/framework-runtimes}"
python_bin="${HETEROSIM_FRAMEWORK_PYTHON:-/usr/bin/python3}"
uv_bin="${HETEROSIM_UV_BIN:-${HOME}/.local/bin/uv}"
index_url="${HETEROSIM_PYPI_INDEX:-https://pypi.org/simple}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is missing at ${uv_bin}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python is missing at ${python_bin}" >&2
  exit 2
fi

mkdir -p "${runtime_root}" "${runtime_root}/cache/uv" "${runtime_root}/cache/huggingface"
export UV_CACHE_DIR="${runtime_root}/cache/uv"
export HF_HOME="${runtime_root}/cache/huggingface"

install_huggingface() {
  local env_dir="${runtime_root}/hf-live"
  "${uv_bin}" venv --python "${python_bin}" --seed "${env_dir}"
  "${uv_bin}" pip install --python "${env_dir}/bin/python" \
    --index-url "${index_url}" --torch-backend=auto \
    "torch==2.14.0" \
    "transformers==5.16.1" \
    "accelerate==1.14.0" \
    "huggingface-hub==1.30.0" \
    "sentencepiece==0.2.1" \
    "protobuf==6.32.1"
  "${uv_bin}" pip freeze --python "${env_dir}/bin/python" > "${env_dir}/requirements.lock"
}

install_vllm() {
  local env_dir="${runtime_root}/vllm-live"
  "${uv_bin}" venv --python "${python_bin}" --seed "${env_dir}"
  "${uv_bin}" pip install --python "${env_dir}/bin/python" \
    --index-url "${index_url}" --torch-backend=auto \
    "vllm==0.29.0"
  "${uv_bin}" pip freeze --python "${env_dir}/bin/python" > "${env_dir}/requirements.lock"
}

install_tensorrt_llm() {
  local env_dir="${runtime_root}/trtllm-live"
  local wheel="https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.2.1-cp310-cp310-linux_x86_64.whl#sha256=66d7e8405b3021d163cdc761c997aa6dcc3bb9c15aefc2029323748840c3288a"
  "${uv_bin}" venv --python "${python_bin}" --seed "${env_dir}"
  "${uv_bin}" pip install --python "${env_dir}/bin/python" \
    --index-url "${index_url}" \
    --extra-index-url https://pypi.nvidia.com \
    "${wheel}"
  "${uv_bin}" pip freeze --python "${env_dir}/bin/python" > "${env_dir}/requirements.lock"
  if ! "${env_dir}/bin/python" -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())' \
    >/dev/null 2>&1; then
    if ! HETEROSIM_FRAMEWORK_ROOT="${runtime_root}" \
      bash "${script_dir}/run_p29_profile.sh" tensorrt_llm \
      -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())' \
      >/dev/null 2>&1; then
      HETEROSIM_FRAMEWORK_ROOT="${runtime_root}" \
        bash "${script_dir}/install_p29_user_mpi_runtime.sh"
    fi
    HETEROSIM_FRAMEWORK_ROOT="${runtime_root}" \
      bash "${script_dir}/run_p29_profile.sh" tensorrt_llm \
      -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())' \
      >/dev/null
  fi
}

case "${profile}" in
  huggingface) install_huggingface ;;
  vllm) install_vllm ;;
  tensorrt_llm) install_tensorrt_llm ;;
  all)
    install_huggingface
    install_vllm
    install_tensorrt_llm
    ;;
  *)
    echo "usage: $0 [huggingface|vllm|tensorrt_llm|all]" >&2
    exit 2
    ;;
esac

echo "Installed ${profile} runtime profile under ${runtime_root}"
