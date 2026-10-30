#!/usr/bin/env bash
# 作用：复用已验证且停止的 sp04 batch32 容器，预热后对指定较高并发复核3分钟，结束时恢复原sp04。
# 示例：REPORT_DIR=/home/yutao.he/lmdeploy-handoff-20261029/artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm CONCURRENCY=48 bash tools/confirm_sp04_concurrency.sh --execute
# REPORT_DIR必须是搜索报告下的新子目录；本脚本不创建容器、不修改镜像、启动参数或连接行为。
# CONTAINER可覆盖默认本轮batch32容器，但必须匹配父目录已记录的容器ID、模块哈希及正确性证据。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_image=sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b
native_path=/opt/py3/lib/python3.12/site-packages/lmdeploy/turbomind/_turbomind.cpython-312-x86_64-linux-gnu.so
old_container=lmdeploy-v166-qps450
target_container="${CONTAINER:-lmdeploy-v166-concurrency400-b32-20261029r2}"
expected_args='["serve","api_server","/models/V166_mindspeed_mg2hf","--log-level","WARNING","--model-name","qwen","--tp","1","--session-len","16384","--max-batch-size","32","--cache-max-entry-count","0.9","--enable-prefix-caching","--server-port","23333"]'
touched=0
target_attempted=0
run_started_at=

# 未明确执行时只显示计划，不读取必需环境或影响正在运行的搜索。
if [[ "${1:-}" != --execute || "$#" != 1 ]]; then
  echo '计划：核验已停止sp04/batch32容器和已有正确性→切换→5000条预热→指定并发3分钟→恢复原sp04。'
  exit 0
fi
: "${REPORT_DIR:?必须设置新的 REPORT_DIR 子目录}"
: "${CONCURRENCY:?必须设置 CONCURRENCY}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]
[[ "$target_container" != "$old_container" ]]
report_dir="${REPORT_DIR%/}"
evidence_dir="$(dirname -- "$report_dir")"
point_label="b32-c$CONCURRENCY-confirm"
export REPORT_DIR="$report_dir" SLA_METRIC=p95 SLA_MS=400 FD_SAMPLE_SECONDS=5

wait_ready() {
  # 参数为容器名；使用容器内API验证当前实例，最多等待180秒启动完成。
  local name="$1" attempt
  for ((attempt=0; attempt<90; ++attempt)); do
    [[ "$(docker inspect -f '{{.State.Running}}' "$name")" == true ]] || return 1
    if docker exec "$name" curl -fsS --max-time 2 http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

manifest() {
  # 参数为容器名、文件路径；严格白名单过滤环境，只记录与模型、分页和资源相关的配置。
  docker inspect "$1" | jq '.[0] | {id:.Id,name:.Name,image:.Image,path:.Path,args:.Args,
    env:(.Config.Env | map(capture("^(?<key>[^=]+)=(?<value>.*)$")) |
      map(select(.key as $key | ["TM_PREFILL_MAX_SPLITS","TM_PREFILL_PAGED_KV","TM_PREFILL_QUERY_TILE",
        "TM_PREFILL_LOG_DISPATCH","LMDEPLOY_CONTEXT_TOKEN_BUDGET","LMDEPLOY_QWEN2_BYTELEVEL_TAIL",
        "LMDEPLOY_QWEN2_ENCODE_CACHE"] | index($key))) | from_entries),
    ulimits:.HostConfig.Ulimits,gpu:.HostConfig.DeviceRequests,shm:.HostConfig.ShmSize,
    privileged:.HostConfig.Privileged,ports:.HostConfig.PortBindings,
    model_mounts:[.Mounts[]|select(.Destination=="/models")|{Type,Source,Destination,RW}],
    running:.State.Running,started_at:.State.StartedAt}' > "$2"
}

check_target() {
  # 参数为清单和预期running值；完整启动参数、分页关闭和资源约束必须同时满足。
  local file="$1" running="$2"
  jq -e \
    --arg image "$baseline_image" \
    --argjson args "$expected_args" \
    --argjson running "$running" \
    '.image==$image and .path=="/opt/py3/bin/lmdeploy" and .args==$args and .running==$running
      and .env.TM_PREFILL_PAGED_KV=="0" and .env.TM_PREFILL_MAX_SPLITS=="8"
      and .env.TM_PREFILL_QUERY_TILE=="64" and .env.LMDEPLOY_CONTEXT_TOKEN_BUDGET=="24576"
      and .env.LMDEPLOY_QWEN2_BYTELEVEL_TAIL=="1" and .env.LMDEPLOY_QWEN2_ENCODE_CACHE=="0"
      and any(.ulimits[]; .Name=="nofile" and .Soft==65536 and .Hard==65536)
      and .shm==8589934592 and .privileged==true and any(.gpu[]; .DeviceIDs==["0"])
      and any(.ports["23333/tcp"][]; .HostPort=="23333")
      and (.model_mounts|length)==1 and any(.model_mounts[];
        .Type=="bind" and .Source=="/data/resources/resource/olive.core.llm" and .Destination=="/models" and .RW==false)' \
    "$file" >/dev/null
}

restore() {
  # 正常、SLA失败或异常退出均先停止目标再恢复原sp04；恢复结果单独落盘，不覆盖性能失败证据。
  local status=$? target_stopped=1 restored=0 restored_hash=
  trap - EXIT
  trap '' INT TERM
  set +e
  if [[ "$target_attempted" == 1 ]]; then
    docker logs "$target_container" > "$report_dir/target-full-server.log" 2>&1
    if [[ -n "$run_started_at" ]]; then
      docker logs --since "$run_started_at" "$target_container" > "$report_dir/target-current-server.log" 2>&1
    fi
    docker stop -t 30 "$target_container" >/dev/null || status=1
    [[ "$(docker inspect -f '{{.State.Running}}' "$target_container")" == false ]] || target_stopped=0
  fi
  if [[ "$touched" == 1 && "$target_stopped" == 1 ]]; then
    if docker start "$old_container" >/dev/null && wait_ready "$old_container"; then
      manifest "$old_container" "$report_dir/restored-sp04.json"
      restored_hash="$(docker exec "$old_container" sha256sum "$native_path" | awk '{print $1}')"
      if [[ "$restored_hash" == "$native_sha" ]] && jq -se '
        .[1].running and ((.[0]|del(.running,.started_at))==(.[1]|del(.running,.started_at)))' \
        "$report_dir/sp04-before.json" "$report_dir/restored-sp04.json" >/dev/null; then restored=1; fi
    fi
  fi
  [[ "$target_stopped" == 1 && "$restored" == 1 ]] || status=1
  jq -n \
    --argjson exit_code "$status" \
    --argjson target_stopped "$target_stopped" \
    --argjson restored_sp04 "$restored" \
    '{exit_code:$exit_code,target_stopped:$target_stopped,restored_sp04:$restored_sp04}' > "$report_dir/restoration.json"
  echo "confirmation_exit=$status target_stopped=$target_stopped restored_sp04=$restored"
  exit "$status"
}

check_server_log() {
  # 参数为日志路径；任何EMFILE、CUDA或OOM证据都保留原日志并判为本点资源失败。
  if rg -qi 'OSError: \[Errno 24\]|too many open files|EMFILE|CUDA error|illegal memory access|out of memory|OutOfMemoryError' "$1"; then
    echo "检测到服务资源或CUDA异常：$1" >&2
    return 1
  fi
}

# 与搜索共用锁，必须等待主搜索结束且原sp04已恢复；只检查自己拥有的负载进程。
for command in docker jq rg sha256sum tar awk flock pgrep id date; do command -v "$command" >/dev/null; done
exec 9>"$repo_root/../artifacts/concurrency-search.lock"
flock -n 9
for process in k6-sse k6 nsys ncu; do
  if pgrep -u "$(id -u)" -x "$process" >/dev/null; then
    echo "存在自己的压测或profiler进程：$process" >&2
    exit 1
  fi
done
test ! -e "$report_dir"
test -s "$evidence_dir/b32-start1-correctness.json"
test -s "$evidence_dir/b32-start1-native-sha256.txt"
test -s "$evidence_dir/b32-container-id.txt"
jq -e '.requests==512 and .errors==[] and .output_mismatches==[]' \
  "$evidence_dir/b32-start1-correctness.json" >/dev/null
[[ "$(docker inspect -f '{{.Id}}' "$target_container")" == "$(cat "$evidence_dir/b32-container-id.txt")" ]]
mkdir -p "$report_dir"
# 固定实际执行的辅助脚本版本，避免后续修改脚本后误认成本轮配置。
sha256sum "$repo_root/tools/confirm_sp04_concurrency.sh" "$repo_root/tools/benchmark_concurrency_point.sh" > "$report_dir/tool-sha256.txt"
manifest "$old_container" "$report_dir/sp04-before.json"
manifest "$target_container" "$report_dir/target-before.json"
jq -e \
  --arg image "$baseline_image" \
  '.running and .image==$image and (.env.TM_PREFILL_PAGED_KV//"0")=="0"' \
  "$report_dir/sp04-before.json" >/dev/null
check_target "$report_dir/target-before.json" false
native_sha="$(docker exec "$old_container" sha256sum "$native_path" | awk '{print $1}')"
target_sha="$(docker cp "$target_container:$native_path" - | tar -xOf - | sha256sum | awk '{print $1}')"
prior_sha="$(awk '{print $1}' "$evidence_dir/b32-start1-native-sha256.txt")"
[[ "$native_sha" =~ ^[0-9a-f]{64}$ && "$native_sha" == "$target_sha" && "$native_sha" == "$prior_sha" ]]
printf '%s  original_sp04\n%s  stopped_batch32\n%s  prior_verified_batch32\n' \
  "$native_sha" "$target_sha" "$prior_sha" > "$report_dir/native-sha256.txt"
cp "$evidence_dir/b32-start1-correctness.json" "$report_dir/reused-correctness.json"
sha256sum "$evidence_dir/b32-start1-correctness.json" > "$report_dir/reused-correctness-sha256.txt"
# 与探索保持相同客户端描述符上限；不修改k6连接复用行为或目标容器资源配置。
ulimit -n 65536
{ ulimit -Sn; ulimit -Hn; } > "$report_dir/client-inherited-nofile.txt"

# 在第一条停服命令之前注册恢复；start失败或后续检查失败也会回到原sp04。
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
touched=1
docker stop -t 30 "$old_container" >/dev/null
target_attempted=1
docker start "$target_container" >/dev/null
wait_ready "$target_container"
run_started_at="$(docker inspect -f '{{.State.StartedAt}}' "$target_container")"
manifest "$target_container" "$report_dir/target-running.json"
check_target "$report_dir/target-running.json" true
docker exec "$target_container" sha256sum "$native_path" > "$report_dir/target-running-native-sha256.txt"
[[ "$(awk '{print $1}' "$report_dir/target-running-native-sha256.txt")" == "$native_sha" ]]
docker exec "$target_container" cat /proc/1/limits > "$report_dir/target-process-limits.txt"

LABEL=b32-confirm-warmup CONCURRENCY=16 ITERATIONS=5000 CONTAINER="$target_container" \
  bash "$repo_root/tools/benchmark_concurrency_point.sh"
docker logs --since "$run_started_at" "$target_container" > "$report_dir/warmup-server.log" 2>&1
check_server_log "$report_dir/warmup-server.log"

# 正式点显式清空ITERATIONS，确保不继承父环境的预热设置；SLA未过仍保留完整结果并正常恢复。
point_exit=0
LABEL="$point_label" CONCURRENCY="$CONCURRENCY" DURATION=3m ITERATIONS= CONTAINER="$target_container" \
  bash "$repo_root/tools/benchmark_concurrency_point.sh" || point_exit=$?
sleep 10
docker exec "$target_container" sh -c 'set -- /proc/1/fd/*; printf "%s\n" "$#"' > "$report_dir/idle-fd.txt"
docker logs --since "$run_started_at" "$target_container" > "$report_dir/target-current-server.log" 2>&1
server_log_passed=true
if ! check_server_log "$report_dir/target-current-server.log"; then server_log_passed=false; point_exit=2; fi
if [[ -s "$report_dir/$point_label.result.json" ]]; then
  jq \
    --argjson server_log_passed "$server_log_passed" \
    --argjson point_exit "$point_exit" \
    '. + {server_batch:32,variant:"sp04",purpose:"higher_concurrency_confirmation",
      server_log_passed:$server_log_passed,point_exit:$point_exit} |
      if $server_log_passed then . else .status="server_resource_failed" end' \
    "$report_dir/$point_label.result.json" > "$report_dir/final-confirmation.json"
  cat "$report_dir/final-confirmation.json"
else
  echo '正式点未产生有效结果；不能把缺失报告解释为通过。' >&2
  point_exit=2
fi
exit "$point_exit"
