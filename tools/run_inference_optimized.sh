#!/usr/bin/env bash
# 作用：启动已验收的短prefill优化镜像；存在同名容器时拒绝覆盖，模型和服务参数保持固定。
# 示例：bash tools/run_inference_optimized.sh
set -euo pipefail
service_name="${SERVICE_NAME:-lmdeploy-v166-opt24-final-20261029}"
if docker container inspect "$service_name" >/dev/null 2>&1; then
  echo "容器 $service_name 已存在，请先检查状态。" >&2
  exit 1
fi
docker run \
  -d \
  --name "$service_name" \
  --gpus device=0 \
  --privileged \
  --shm-size=8g \
  -p 23333:23333 \
  -e TM_PREFILL_MAX_SPLITS=8 \
  -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 \
  -v /data/resources/resource/olive.core.llm:/models:ro \
  lmdeploy_cu128:v0.18.0_sp07_verified_20261029 serve api_server /models/V166_mindspeed_mg2hf \
  --log-level WARNING \
  --model-name qwen \
  --tp 1 \
  --session-len 16384 \
  --max-batch-size 16 \
  --cache-max-entry-count 0.9 \
  --enable-prefix-caching \
  --server-port 23333
