#!/bin/bash
# Compile-only check of every translation unit with the real flags, on a machine without a GPU
# (Kaggle CPU session: CUDA comes from pip wheels).  Prints every error; exit 1 if any.
set -u
if [ -x /usr/local/cuda/bin/nvcc ]; then export PATH=/usr/local/cuda/bin:$PATH; fi
if ! command -v nvcc >/dev/null; then
  # CPU-only image: install just the compiler + headers from NVIDIA's apt repo (~2 min)
  (cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
     && dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && apt-get update -qq >/dev/null 2>&1 \
     && apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 cuda-nvml-dev-12-8 >/dev/null 2>&1)
  export PATH=/usr/local/cuda-12.8/bin:$PATH
fi
INC=""
nvcc --version | tail -1
FLAGS="-std=c++17 -O3 -arch=sm_75 -Werror=all-warnings --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra,-Werror,-march=native $INC"
rc=0
for f in src/gpu/*.cu src/kernels/*.cu bench/*.cu; do
  out=$(nvcc $FLAGS -c "$f" -o /tmp/$(basename "$f").o 2>&1) || { echo "=== $f"; echo "$out" | grep -v "^$" | head -30; rc=1; }
done
for f in src/*.cpp src/cpu/*.cpp; do
  out=$(g++ -std=c++17 -O3 -march=native -Wall -Wextra -Werror -c "$f" -o /tmp/x.o 2>&1) || { echo "=== $f"; echo "$out" | head -30; rc=1; }
done
echo "compile_check rc=$rc"
exit $rc
