#!/usr/bin/env bash
set -euo pipefail

profile="${1:-}"
if [[ -z "${profile}" || $# -lt 2 ]]; then
  echo "usage: $0 <huggingface|vllm|tensorrt_llm> <python-script> [arguments...]" >&2
  exit 2
fi
shift

runtime_root="${HETEROSIM_FRAMEWORK_ROOT:-${HOME}/.local/share/gpu-atlas/framework-runtimes}"
case "${profile}" in
  huggingface) env_dir="${runtime_root}/hf-live" ;;
  vllm) env_dir="${runtime_root}/vllm-live" ;;
  tensorrt_llm) env_dir="${runtime_root}/trtllm-live" ;;
  *)
    echo "unsupported P29 profile: ${profile}" >&2
    exit 2
    ;;
esac

python_bin="${env_dir}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "runtime profile is not installed: ${env_dir}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-${runtime_root}/cache/huggingface}"
export PATH="${env_dir}/bin:${PATH}"

if [[ "${profile}" == "tensorrt_llm" ]]; then
  mpi_root="${HETEROSIM_MPI_ROOT:-${runtime_root}/mpi-runtime}"
  if [[ -d "${mpi_root}/usr/lib/x86_64-linux-gnu" ]]; then
    export OPAL_PREFIX="${mpi_root}/usr"
    export PMIX_PREFIX="${mpi_root}/usr"
    export PMIX_MCA_mca_base_component_path="${mpi_root}/usr/lib/x86_64-linux-gnu/pmix2/lib/pmix"
    export LD_LIBRARY_PATH="${mpi_root}/usr/lib/x86_64-linux-gnu:${mpi_root}/usr/lib/libpsm1${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="${mpi_root}/usr/bin:${PATH}"
    export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader,tcp}"
    export OMPI_MCA_mtl="${OMPI_MCA_mtl:-^psm,ofi}"
  fi
fi

exec "${python_bin}" "$@"
