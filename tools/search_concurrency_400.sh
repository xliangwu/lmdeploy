#!/usr/bin/env bash
# 作用：在指定已验证镜像上搜索400ms完整请求延迟内的客户端并发与batch16/32，退出时恢复原sp04。
# 示例：bash tools/search_concurrency_400.sh --execute
# 可用 REPORT_DIR 指定新证据目录，VARIANT=sp04 使用原版，RUN_SUFFIX 指定唯一容器后缀。
# SLA_METRIC=p95/p99/max 指定延迟口径；不改变context预算、不采集profiler。
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
report_dir="${REPORT_DIR:-$repo_root/../artifacts/concurrency-400-20261029}"
export REPORT_DIR="$report_dir" SLA_METRIC="${SLA_METRIC:-p95}" SLA_MS=400 FD_SAMPLE_SECONDS=5
candidate_image=sha256:fa987b96aa211ec01b736a1d8e60fef3e6a954aa9d02113262301b24c6fe23a1
baseline_image=sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b
native_sha=3e1ac6ae9569d2fb482714acdb9a8ef17cbdc60df5a697f89521f5e63372116a
variant="${VARIANT:-paged}"
run_suffix="${RUN_SUFFIX:-20261029}"
paged_enabled=1
[[ "$variant" == paged || "$variant" == sp04 ]]
[[ "$run_suffix" =~ ^[a-zA-Z0-9_-]+$ ]]
old_container=lmdeploy-v166-qps450
dataset=/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt
previous_reference="$repo_root/../artifacts/paged-kv/authorized-acceptance/service-acceptance/reference.json"
current_container=""
current_batch=0
touched=0
created_containers=()
declare -A batch_starts=()

# 默认仅说明计划，避免查看脚本时意外切换服务。
if [[ "${1:-}" != --execute || "$#" != 1 ]]; then
  echo '计划：指定已验证镜像、nofile65536；最多5个探索点+赢家3分钟复核，结束恢复sp04。'
  exit 0
fi

wait_ready() {
  # 参数为容器名；只检查该容器内部端点，防止认错占用同端口的服务。
  local container="$1" attempt
  for ((attempt=0; attempt<90; ++attempt)); do
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]] || return 1
    if docker exec "$container" curl -fsS --max-time 2 http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

manifest() {
  # 参数为容器名和输出路径，仅保存本任务相关环境，排除无关密钥。
  docker inspect "$1" | jq '.[0] | {name:.Name,image:.Image,path:.Path,args:.Args,
    env:(.Config.Env|map(select(test("^(TM_|LMDEPLOY_)")))),
    ulimits:.HostConfig.Ulimits,gpu:.HostConfig.DeviceRequests,shm:.HostConfig.ShmSize,
    ports:.HostConfig.PortBindings,model_mounts:[.Mounts[]|select(.Destination=="/models")],
    running:.State.Running,started_at:.State.StartedAt}' > "$2"
}

restore() {
  # 任一错误或正常退出都停止本轮容器并恢复原sp04；保留原容器和本轮全部失败点证据。
  local status=$? name restored=0 stopped=1
  trap - EXIT
  trap '' INT TERM
  set +e
  for name in "${created_containers[@]}"; do
    docker logs "$name" > "$report_dir/$name-final-server.log" 2>&1
    docker stop -t 30 "$name" >/dev/null || status=1
    [[ "$(docker inspect -f '{{.State.Running}}' "$name")" == false ]] || stopped=0
  done
  if [[ "$touched" == 1 ]]; then
    if docker start "$old_container" >/dev/null && wait_ready "$old_container"; then
      manifest "$old_container" "$report_dir/restored-sp04.json"
      if jq -e --arg image "$baseline_image" '.running and .image==$image' "$report_dir/restored-sp04.json" >/dev/null; then restored=1; fi
    fi
  fi
  [[ "$stopped" == 1 && "$restored" == 1 ]] || status=1
  jq -n \
    --argjson exit_code "$status" \
    --argjson candidate_stopped "$stopped" \
    --argjson restored_sp04 "$restored" \
    '{exit_code:$exit_code,candidate_stopped:$candidate_stopped,restored_sp04:$restored_sp04}' > "$report_dir/restoration.json"
  echo "search_exit=$status candidate_stopped=$stopped restored_sp04=$restored"
  exit "$status"
}

point() {
  # 参数为标签、客户端并发、持续时间；batch由当前容器决定，结果中额外记录以便公平比较。
  local label="$1" concurrency="$2" duration="$3"
  LABEL="$label" CONCURRENCY="$concurrency" DURATION="$duration" CONTAINER="$current_container" \
    bash "$repo_root/tools/benchmark_concurrency_point.sh"
  jq --argjson batch "$current_batch" --arg variant "$variant" '. + {server_batch:$batch,variant:$variant}' "$report_dir/$label.result.json" > "$report_dir/$label-configured.json"
  jq -c . "$report_dir/$label-configured.json" >> "$report_dir/search-results.jsonl"
  # 等待两倍keepalive时间，再记录主服务FD，核对连接是否在压测结束后正常释放。
  sleep 10
  docker exec "$current_container" sh -c 'set -- /proc/1/fd/*; printf "%s\n" "$#"' > "$report_dir/$label-idle-fd.txt"
  # 每个点检查完整服务日志，入口资源耗尽或CUDA错误出现时不继续提高压力。
  docker logs "$current_container" > "$report_dir/$label-server.log" 2>&1
  if rg -q 'OSError: \[Errno 24\]|CUDA error|illegal memory access|out of memory' "$report_dir/$label-server.log"; then
    echo "检测到服务资源或CUDA异常，停止搜索：$label" >&2
    return 1
  fi
}

start_batch() {
  # 参数为服务batch；按明确版本选择原版/分页，提高nofile避免上轮1024连接上限。
  local batch="$1" name="lmdeploy-v166-concurrency400-b$1-$run_suffix" start_label
  batch_starts[$batch]=$(( ${batch_starts[$batch]:-0} + 1 ))
  start_label="b$batch-start${batch_starts[$batch]}"
  if [[ -n "$current_container" ]]; then docker stop -t 30 "$current_container" >/dev/null; fi
  if docker container inspect "$name" >/dev/null 2>&1; then
    # 同一轮返回先前batch时复用自己的容器；预检保证启动前不存在同名容器。
    docker start "$name" >/dev/null
  else
    created_containers+=("$name")
    docker run \
      -d \
      --name "$name" \
      --gpus device=0 \
      --privileged \
      --shm-size 8g \
      --ulimit nofile=65536:65536 \
      -p 23333:23333 \
      -e TM_PREFILL_MAX_SPLITS=8 \
      -e TM_PREFILL_PAGED_KV="$paged_enabled" \
      -e TM_PREFILL_QUERY_TILE=64 \
      -e TM_PREFILL_LOG_DISPATCH=1 \
      -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 \
      -e LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1 \
      -e LMDEPLOY_QWEN2_ENCODE_CACHE=0 \
      -v /data/resources/resource/olive.core.llm:/models:ro \
      "$candidate_image" serve api_server /models/V166_mindspeed_mg2hf \
      --log-level WARNING \
      --model-name qwen \
      --tp 1 \
      --session-len 16384 \
      --max-batch-size "$batch" \
      --cache-max-entry-count 0.9 \
      --enable-prefix-caching \
      --server-port 23333 > "$report_dir/b$batch-container-id.txt"
  fi
  current_container="$name"
  current_batch="$batch"
  wait_ready "$name"
  manifest "$name" "$report_dir/$start_label-manifest.json"
  docker exec "$name" cat /proc/1/limits > "$report_dir/$start_label-process-limits.txt"
  docker exec "$name" /opt/py3/bin/lmdeploy serve api_server --help > "$report_dir/$start_label-api-help.txt"
  docker exec "$name" sha256sum /opt/py3/lib/python3.12/site-packages/lmdeploy/turbomind/_turbomind.cpython-312-x86_64-linux-gnu.so > "$report_dir/$start_label-native-sha256.txt"
  [[ "$(awk '{print $1}' "$report_dir/$start_label-native-sha256.txt")" == "$native_sha" ]]
  # 旧参考是串行采集，按相同串行条件比较；并发批次的输出变动另存，不用串行参考冒充同批次对照。
  cp "$previous_reference" "$report_dir/$start_label-reference.json"
  python3 "$repo_root/tools/verify_prefill_split_kv.py" \
    --mode compare \
    --dataset "$dataset" \
    --reference "$report_dir/$start_label-reference.json" \
    --count 512 \
    --concurrency 1 > "$report_dir/$start_label-correctness.json"
  jq -e '.requests==512 and .errors==[] and .output_mismatches==[]' "$report_dir/$start_label-correctness.json" >/dev/null
  LABEL="$start_label-warmup" CONCURRENCY=16 ITERATIONS=5000 CONTAINER="$name" \
    bash "$repo_root/tools/benchmark_concurrency_point.sh"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$report_dir/$start_label-gpu.csv"
  docker logs "$name" > "$report_dir/$start_label-warmup-server.log" 2>&1
  # 新batch的预热也必须没有资源异常，不能等第一轮正式压测后才发现已经失效。
  if rg -q 'OSError: \[Errno 24\]|CUDA error|illegal memory access|out of memory' "$report_dir/$start_label-warmup-server.log"; then
    echo "预热出现资源或CUDA异常：$start_label" >&2
    return 1
  fi
}

# 预检固定镜像、已有正确性证据、端口归属和独占压测，避免干扰其他任务。
mkdir -p "$report_dir"
exec 9>"$repo_root/../artifacts/concurrency-search.lock"
flock -n 9
test ! -e "$report_dir/search-results.jsonl"
ulimit -n 65536
{ ulimit -Sn; ulimit -Hn; } > "$report_dir/client-inherited-nofile.txt"
for batch in 16 32; do
  if docker container inspect "lmdeploy-v166-concurrency400-b$batch-$run_suffix" >/dev/null 2>&1; then
    echo '本轮容器已存在，拒绝覆盖。' >&2; exit 1
  fi
done
for process in k6-sse k6 nsys ncu; do
  while read -r pid; do
    # 仅可显式豁免已人工只读核验的远端压测PID；不停止他人的任务，也不忽略未知负载。
    if [[ "$process" == k6-sse && "$pid" == "${EXCLUDED_REMOTE_LOAD_PID:-}" ]]; then
      ps -p "$pid" -o user,pid,ppid,pcpu,pmem,lstart,args > "$report_dir/existing-remote-load.txt"
      continue
    fi
    echo "发现其他负载：$process pid=$pid" >&2
    exit 1
  done < <(pgrep -x "$process" || true)
done
manifest "$old_container" "$report_dir/sp04-before.json"
jq -e --arg image "$baseline_image" '.running and .image==$image' "$report_dir/sp04-before.json" >/dev/null
if [[ "$variant" == sp04 ]]; then
  # 原版容量测试直接固定原镜像及当前模块哈希，不把分页候选的正确性状态套到原版上。
  candidate_image="$baseline_image"
  paged_enabled=0
  native_sha="$(docker exec "$old_container" sha256sum /opt/py3/lib/python3.12/site-packages/lmdeploy/turbomind/_turbomind.cpython-312-x86_64-linux-gnu.so | awk '{print $1}')"
else
  jq -e '.correctness_and_sla_passed' "$repo_root/../artifacts/paged-kv/authorized-acceptance/service-validation.json" >/dev/null
  docker image inspect "$candidate_image" | jq -e --arg native "$native_sha" '.[0].Config.Labels["ai.aispeech.paged-prefill.native-sha256"]==$native' >/dev/null
fi
sha256sum "$repo_root/tools/benchmark_concurrency_point.sh" "$repo_root/tools/search_concurrency_400.sh" \
  /home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js "$dataset" > "$report_dir/input-sha256.txt"
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
touched=1
docker stop -t 30 "$old_container" >/dev/null
start_batch 16
point b16-c32 32 90s
point b16-c64 64 90s
start_batch 32
point b32-c64 64 90s

# 在相同并发64下优先选择通过SLA且QPS较高的batch；两者均超时则选p95较低的一侧向下探测。
selected_batch="$(jq -s '(.|map(select(.sla_passed))) as $passing |
  if ($passing|length)>0 then ($passing|max_by(.qps)).server_batch
  else (sort_by(.complete_p95_ms,-.qps)|first).server_batch end' \
  "$report_dir/b16-c64-configured.json" "$report_dir/b32-c64-configured.json")"
if [[ "$selected_batch" != "$current_batch" ]]; then
  # 每次start_batch均使用新代号保存预热/正确性证据，避免回到旧batch时覆盖首次记录。
  start_batch "$selected_batch"
fi
if jq -e '.sla_passed' "$report_dir/b$selected_batch-c64-configured.json" >/dev/null; then
  point "b$selected_batch-c80" 80 90s
  if jq -e '.sla_passed' "$report_dir/b$selected_batch-c80-configured.json" >/dev/null; then next=96; else next=72; fi
else
  point "b$selected_batch-c48" 48 90s
  if jq -e '.sla_passed' "$report_dir/b$selected_batch-c48-configured.json" >/dev/null; then next=56; else next=40; fi
fi
point "b$selected_batch-c$next" "$next" 90s

# 只从通过SLA的探索点选最高QPS，再用3分钟复核；边界不通过时不推荐该点。
jq -s 'map(select(.sla_passed)) | if length==0 then error("没有通过400ms的配置") else max_by(.qps) end' \
  "$report_dir/search-results.jsonl" > "$report_dir/exploration-winner.json"
winner_batch="$(jq -r '.server_batch' "$report_dir/exploration-winner.json")"
winner_concurrency="$(jq -r '.concurrency' "$report_dir/exploration-winner.json")"
if [[ "$winner_batch" != "$current_batch" ]]; then
  echo '最佳点位于另一batch，需要保留记录后恢复该batch。' >&2
  start_batch "$winner_batch"
fi
point "b$winner_batch-c$winner_concurrency-confirm" "$winner_concurrency" 3m
cp "$report_dir/b$winner_batch-c$winner_concurrency-confirm-configured.json" "$report_dir/final-confirmation.json"
jq -e '.sla_passed' "$report_dir/final-confirmation.json" >/dev/null
if [[ "$variant" == paged ]]; then
  rg -q 'paged_prefill_dispatch dtype=bf16 query_tile=64' "$report_dir/b$winner_batch-c$winner_concurrency-confirm-server.log"
fi
cat "$report_dir/final-confirmation.json"
