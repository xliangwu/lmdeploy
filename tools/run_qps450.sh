#!/usr/bin/env bash
# 作用：启动已验证的 3090/V166 最终配置；不停止或覆盖任何现有服务。
# 示例：bash tools/run_qps450.sh
set -euo pipefail
service_name=lmdeploy-v166-qps450
# 已有同名容器时拒绝重复创建，避免误覆盖已完成的部署。
if docker container inspect "$service_name" >/dev/null 2>&1; then
  echo "容器 $service_name 已存在；请先检查当前状态。" >&2
  exit 1
fi
docker run \
  -d \
  --name "$service_name" \
  --gpus 'device=0' \
  --privileged \
  --shm-size=8g \
  -e TM_PREFILL_MAX_SPLITS=8 \
  -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 \
  -p 23333:23333 \
  -v /data/resources/resource/olive.core.llm:/models \
  lmdeploy_cu128:v0.18.0_sp04_qps450_20261028 serve api_server /models/V166_mindspeed_mg2hf \
  --log-level WARNING \
  --model-name qwen \
  --tp 1 \
  --session-len 16384 \
  --max-batch-size 16 \
  --cache-max-entry-count 0.9 \
  --enable-prefix-caching \
  --server-port 23333
