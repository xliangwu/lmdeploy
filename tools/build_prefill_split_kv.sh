#!/usr/bin/env bash
# 作用：在独立构建容器中编译 SM86 TurboMind，打包用于继承 mentor 镜像的原生模块。
# 示例：bash tools/build_prefill_split_kv.sh
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_root/build-splitkv}"
site_root="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# 沿用镜像已安装的 CUDA 与 pybind11 依赖，避免误选系统旧版本。
export CPATH="$site_root/nvidia/cublas/include:$site_root/nvidia/curand/include:$site_root/nvidia/nvtx/include${CPATH:+:$CPATH}"
cmake -S "$repo_root" \
  -B "$build_dir" \
  -G Ninja \
  -DBUILD_TEST=ON \
  -DBUILD_PY_FFI=ON \
  -DBUILD_MULTI_GPU=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCC_THREADS=1 \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCUDA_cublas_LIBRARY="$site_root/nvidia/cublas/lib/libcublas.so.12" \
  -DCUDA_cublasLt_LIBRARY="$site_root/nvidia/cublas/lib/libcublasLt.so.12" \
  -DCMAKE_CXX_FLAGS="-I$site_root/nvidia/cublas/include" \
  -DCMAKE_CUDA_FLAGS="-I$site_root/nvidia/cublas/include"
cmake --build "$build_dir" --target _turbomind test_prefill_split_kv --parallel "${JOBS:-4}"
mkdir -p "$build_dir/image/runtime"
cp "$repo_root"/lmdeploy/turbomind/_turbomind*.so "$build_dir/image/runtime/"
# 原生依赖如存在也一并复制，不覆盖基础镜像中的 Python 预处理代码。
if [ -d "$repo_root/lmdeploy/turbomind/lib" ]; then
  cp -a "$repo_root/lmdeploy/turbomind/lib" "$build_dir/image/runtime/"
fi
printf 'Image context: %s\n' "$build_dir/image"
