#!/bin/bash
# Build the GPU DMRG code: bash build.sh [dmrg_envck|dmrg_cap]  ->  bin/<app>
# Needs CUDA (tested with 12.1; nvcc, cuBLAS, cuSOLVER, cuRAND) and cuTENSOR 2.x.
# Set CUDA_HOME and CUTENSOR_ROOT if they are not found automatically.
set -euo pipefail
cd "$(dirname "$0")"
APP=${1:-dmrg_envck}
CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
: "${CUTENSOR_ROOT:?set CUTENSOR_ROOT to the cuTENSOR install (contains include/ and lib/)}"
ARCH=${ARCH:-sm_90}   # H100/H200; use sm_80 for A100
mkdir -p bin
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$ARCH" ${CCBIN:+-ccbin "$CCBIN"} -Isrc -I"$CUTENSOR_ROOT/include" \
    "apps/$APP.cu" src/*.cu \
    -lcublas -lcusolver -lcurand -L"$CUTENSOR_ROOT/lib" -lcutensor \
    -Xlinker -rpath="$CUTENSOR_ROOT/lib" -Xlinker -rpath="$CUDA_HOME/lib64" \
    -o "bin/$APP"
echo "built bin/$APP"
