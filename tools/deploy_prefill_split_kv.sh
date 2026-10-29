#!/usr/bin/env bash
# 作用：将 3090 的 V166 服务切换至 split-KV 镜像；启动或输出验收失败时自动恢复旧服务。
# 示例：bash tools/deploy_prefill_split_kv.sh
set -euo pipefail
old_container=c258526a2236
new_container=lmdeploy-v166-splitkv
new_image=lmdeploy_cu128:v0.18.0_sp02_splitkv_20261028_r3
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
report_dir=/home/yutao.he/lmdeploy-profile-v166-20261027/deploy-20261028
switched=0
success=0

wait_ready() {
  # 最多等待两分钟，仅检查模型列表，不产生压测负载。
  for attempt in $(seq 1 60); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$1")" != true ]; then return 1; fi
    if curl -fsS --max-time 2 http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

rollback() {
  # 只在已经停止旧服务、且新版本未完成验收时回退；保留失败容器供诊断。
  result=$?
  if [ "$switched" = 1 ] && [ "$success" = 0 ]; then
    docker logs --tail 100 "$new_container" > "$report_dir/new-server-failure.log" 2>&1 || true
    docker stop -t 20 "$new_container" || true
    docker start "$old_container"
    docker exec -d "$old_container" sh -c 'exec /opt/py3/bin/python /tmp/lmdeploy-deep/diagnostic_entry.py --state /tmp/lmdeploy-deep/original.json --restore > /tmp/lmdeploy-deep/restored-server.log 2>&1'
    wait_ready "$old_container" && echo 'rollback_ready=true'
  fi
  exit "$result"
}
trap rollback EXIT

# 所有前提先验证，避免缺少镜像或参考结果时停止服务。
test -s "$report_dir/mentor-reference.json"
docker image inspect "$new_image" >/dev/null
# 模块能够导入不代表 kernel 注册完整；切换前实际查询两种浮点格式。
docker run --rm --gpus 'device=0' --entrypoint /opt/py3/bin/python "$new_image" -c 'from lmdeploy.turbomind import _tm; from lmdeploy.turbomind.weight_format import TrivialFormat; g=_tm.Gemm(); types=[_tm.DataType.TYPE_BF16,_tm.DataType.TYPE_FP16]; assert all(t in g.data_types(TrivialFormat(weight_dtype=t).make_data_format()) for t in types); print("native_dtypes_ok")'
test "$(docker inspect -f '{{.State.Running}}' "$old_container")" = true
if docker container inspect "$new_container" >/dev/null 2>&1; then
  echo '目标容器名已存在，请先核对其状态，拒绝覆盖。' >&2
  exit 1
fi
docker exec "$old_container" test -s /tmp/lmdeploy-deep/original.json
date -Is
docker stop -t 45 "$old_container"
switched=1
docker run \
  -d \
  --name "$new_container" \
  --gpus 'device=0' \
  --privileged \
  --shm-size=8g \
  -e TM_PREFILL_MAX_SPLITS=8 \
  -p 23333:23333 \
  -v /data/resources/resource/olive.core.llm:/models \
  "$new_image" serve api_server /models/V166_mindspeed_mg2hf \
  --log-level WARNING \
  --model-name qwen \
  --tp 1 \
  --session-len 16384 \
  --max-batch-size 16 \
  --cache-max-entry-count 0.9 \
  --enable-prefix-caching \
  --server-port 23333
wait_ready "$new_container"
python3 "$repo_root/tools/verify_prefill_split_kv.py" \
  --mode compare \
  --dataset /data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt \
  --reference "$report_dir/mentor-reference.json"
success=1
echo 'deployment_verified=true'
docker inspect "$new_container" --format 'container={{.Id}} image={{.Image}} running={{.State.Running}}'
date -Is
