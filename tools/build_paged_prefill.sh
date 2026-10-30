#!/usr/bin/env bash
# 作用：在挂载当前交接源码的独立容器中构建分页 KV 测试和原生模块，不启动或停止服务。
# 示例：DEPENDENCY_SEED=/dependency-seed JOBS=6 bash tools/build_paged_prefill.sh
# 可选参数：BUILD_DIR 指定独立输出目录；DEPENDENCY_SEED 指定已有第三方源码缓存。
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_root/build-paged-kv}"
site_root="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$build_dir"

# 只复制第三方源码，不复用旧项目目标文件；xgrammar 的配置过程会写源码目录，不能直接用只读缓存。
dependency_args=()
if [ -n "${DEPENDENCY_SEED:-}" ]; then
  for dependency in fmt catch2 repo-cutlass yaml-cpp xgrammar concurrentqueue; do
    source_dir="$build_dir/_deps/$dependency-src"
    if [ ! -d "$source_dir" ]; then
      mkdir -p "$build_dir/_deps"
      cp -a "$DEPENDENCY_SEED/$dependency-src" "$source_dir"
    fi
    dependency_args+=("-DFETCHCONTENT_SOURCE_DIR_${dependency^^}=$source_dir")
  done
fi
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
  -DCMAKE_CUDA_FLAGS="-I$site_root/nvidia/cublas/include" \
  "${dependency_args[@]}"
cmake --build "$build_dir" --target test_prefill_paged_kv test_prefill_split_kv _turbomind --parallel "${JOBS:-6}"

# 将候选模块单独打包；仅覆盖原生模块可在验收时隔离分页改动，避免混入 tokenizer 缓存。
mkdir -p "$build_dir/image"
cp "$repo_root"/lmdeploy/turbomind/_turbomind*.so "$build_dir/image/"
cp "$repo_root/docker/Dockerfile.paged-prefill" "$build_dir/image/Dockerfile"
sha256sum "$build_dir/image/"*.so
