#!/usr/bin/env bash
set -euo pipefail

# TensorRT-LLM imports mpi4py even for a one-GPU process.  On a machine where
# sudo is unavailable, extract Ubuntu's Open MPI runtime into the isolated P29
# prefix instead of changing the host operating system.
runtime_root="${HETEROSIM_FRAMEWORK_ROOT:-${HOME}/.local/share/gpu-atlas/framework-runtimes}"
mpi_root="${HETEROSIM_MPI_ROOT:-${runtime_root}/mpi-runtime}"

for command in apt-cache apt-get dpkg-deb; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required to install the user-space MPI runtime" >&2
    exit 2
  fi
done

download_dir="$(mktemp -d)"
trap 'rm -rf -- "${download_dir}"' EXIT

mapfile -t packages < <(
  {
    printf '%s\n' openmpi-bin openmpi-common libopenmpi3
    apt-cache depends --recurse --no-recommends --no-suggests \
      --no-conflicts --no-breaks --no-replaces --no-enhances \
      openmpi-bin libopenmpi3 2>/dev/null \
      | sed -n 's/^[[:space:]]*Depends:[[:space:]]*//p'
  } | sed '/^</d' | sort -u
)

mkdir -p "${mpi_root}"
(
  cd "${download_dir}"
  for package in "${packages[@]}"; do
    if apt-cache show "${package}" >/dev/null 2>&1; then
      apt-get download "${package}" >/dev/null
    fi
  done
  for archive in ./*.deb; do
    dpkg-deb -x "${archive}" "${mpi_root}"
  done
)

libmpi="$(find "${mpi_root}" -type f -name 'libmpi.so.*' -print -quit)"
if [[ -z "${libmpi}" ]]; then
  echo "Open MPI extraction completed without producing libmpi.so" >&2
  exit 1
fi
printf '%s\n' "${packages[@]}" > "${mpi_root}/packages.lock"
echo "Installed user-space Open MPI under ${mpi_root}"
