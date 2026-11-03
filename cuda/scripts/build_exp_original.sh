#!/bin/bash
# Usage (from DMRG root or anywhere): bash own_experimental/scripts/build_exp.sh <app>
#   builds own_experimental/apps/<app>.cu + main src/*.cu + own_experimental/src/*.cu -> own_experimental/bin/<app>
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_HOME=${CUDA_HOME:-/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p}
CUTENSOR_ROOT=${CUTENSOR_ROOT:-$HOME/opt/cutensor_whl/cutensor}
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p own_experimental/bin
# main src minus files overridden by an own_experimental/src copy of the same name
MAIN_SRC=$(for f in src/*.cu; do [ -e own_experimental/src/$(basename $f) ] || echo $f; done)
nvcc -O3 -std=c++17 -Iown_experimental/src -Isrc -I"$CUTENSOR_ROOT/include" \
     "own_experimental/apps/$1.cu" $MAIN_SRC own_experimental/src/*.cu \
     -lcublas -lcusolver -lcurand -L"$CUTENSOR_ROOT/lib" -lcutensor \
     -Xlinker -rpath="$CUTENSOR_ROOT/lib" -Xlinker -rpath="$CUDA_HOME/lib64" \
     -o "own_experimental/bin/$1"
echo "built own_experimental/bin/$1"
