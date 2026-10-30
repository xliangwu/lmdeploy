#!/usr/bin/env bash
# 作用：对当前服务执行一个固定并发点的 k6 验收，独立判断完整请求 SLA，并保存完成数、失败数与资源快照。
# 本脚本不启动、停止或修改服务；ITERATIONS 非空表示顺序预热，否则按 DURATION 执行正式随机负载。
# 正式示例：REPORT_DIR=/tmp/lmdeploy-points LABEL=c16 CONCURRENCY=16 bash tools/benchmark_concurrency_point.sh
# 预热示例：REPORT_DIR=/tmp/lmdeploy-points LABEL=warmup CONCURRENCY=16 ITERATIONS=5000 bash tools/benchmark_concurrency_point.sh
# 监控示例：REPORT_DIR=/tmp/lmdeploy-points LABEL=c24 CONCURRENCY=24 CONTAINER=lmdeploy-v166-qps450 FD_SAMPLE_SECONDS=15 bash tools/benchmark_concurrency_point.sh
set -euo pipefail

: "${REPORT_DIR:?必须设置 REPORT_DIR}"
: "${LABEL:?必须设置 LABEL}"
: "${CONCURRENCY:?必须设置 CONCURRENCY}"
duration="${DURATION:-90s}"
iterations="${ITERATIONS:-}"
sla_metric="${SLA_METRIC:-p95}"
sla_ms="${SLA_MS:-400}"
container="${CONTAINER:-}"
sample_seconds="${FD_SAMPLE_SECONDS:-0}"
report_dir="$REPORT_DIR"
label="$LABEL"
concurrency="$CONCURRENCY"
load_script=/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js
dataset=/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt
k6=/data/users/xueliang.wu/me-k6-tools/k6-sse
phase=formal
sampler_pid=
raw_exit=0
client_nofile_soft="$(ulimit -Sn)"
client_nofile_hard="$(ulimit -Hn)"

# 标签作为文件名使用，拒绝路径穿越和覆盖既有点；数值参数只接受明确合法形式。
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'LABEL 只能使用字母、数字、点、下划线或连字符。' >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo 'CONCURRENCY 必须是正整数。' >&2; exit 2; }
[[ "$sample_seconds" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'FD_SAMPLE_SECONDS 必须是非负整数。' >&2; exit 2; }
[[ "$sla_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'SLA_MS 必须是正数。' >&2; exit 2; }
case "$sla_metric" in p95|p99|max) ;; *) echo 'SLA_METRIC 仅支持 p95、p99、max。' >&2; exit 2 ;; esac
if [ -n "$iterations" ]; then
  [[ "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo 'ITERATIONS 必须是正整数。' >&2; exit 2; }
  phase=warmup
fi
for command in jq rg sha256sum date flock tail; do command -v "$command" >/dev/null; done
if [ -n "$container" ]; then command -v docker >/dev/null; fi
jq -ne --arg value "$sla_ms" '$value|tonumber|.>0' >/dev/null
test -x "$k6"
test -s "$load_script"
test -s "$dataset"
mkdir -p "$report_dir"
exec 9>"$report_dir/concurrency-point.lock"
flock -n 9 || { echo '此报告目录已有并发点正在执行。' >&2; exit 2; }
for suffix in json log result.json exit-code.txt; do
  test ! -e "$report_dir/$label.$suffix" || { echo "拒绝覆盖已有文件：$report_dir/$label.$suffix" >&2; exit 2; }
done

stop_sampler() {
  # 仅结束本脚本创建的监控子进程；不会向服务、容器或其他压测发送信号。
  if [ -n "$sampler_pid" ]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=
  fi
}
trap stop_sampler EXIT

snapshot_container() {
  # 参数为采样阶段；同时保存当前宿主 init PID、容器内各进程 FD 和 docker stats 内存，避免复用旧 PID。
  local stage="$1" identity descriptors memory timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  identity="$(docker inspect \
    --format '{{json .State}}' \
    "$container")" || return 1
  jq -e '.Running and .Pid>0' <<<"$identity" >/dev/null || return 1
  descriptors="$(docker exec "$container" sh -c '
    # 统计当前容器进程的 FD，排除负责采样的 shell；已退出进程直接忽略。
    for proc_path in /proc/[0-9]*; do
      proc_pid=${proc_path##*/}
      [ "$proc_pid" = "$$" ] && continue
      [ -d "$proc_path/fd" ] || continue
      [ -r "$proc_path/fd" ] || exit 1
      set -- "$proc_path"/fd/*
      if [ -e "$1" ] || [ -L "$1" ]; then fd_count=$#; else fd_count=0; fi
      printf "%s %s\n" "$proc_pid" "$fd_count"
    done
  ' | jq -Rn '[inputs | split(" ") | {pid:(.[0]|tonumber), fd_count:(.[1]|tonumber)}]')" || return 1
  memory="$(docker stats \
    --no-stream \
    --format '{{json .}}' \
    "$container" | jq -c '{memory_usage:.MemUsage,memory_percent:.MemPerc,pids:.PIDs}')" || return 1
  jq -cn \
    --arg stage "$stage" \
    --arg timestamp "$timestamp" \
    --arg container "$container" \
    --argjson identity "$identity" \
    --argjson descriptors "$descriptors" \
    --argjson memory "$memory" \
    '{stage:$stage,timestamp:$timestamp,container:$container,host_init_pid:$identity.Pid,
      started_at:$identity.StartedAt,fd_total:([$descriptors[].fd_count]|add//0),processes:$descriptors,
      memory:$memory}'
}

sample_resources() {
  # 可选后台监控只采样一个指定容器；单次失败落盘，不伪造缺失的 FD 数值。
  local sampler_sleep=
  # 结束采样时一并回收自己的 sleep，避免残留子进程继续持有锁或等待整个采样周期。
  trap 'if [ -n "$sampler_sleep" ]; then kill "$sampler_sleep" 2>/dev/null || true; fi; exit 0' TERM INT
  while true; do
    sleep "$sample_seconds" & sampler_sleep=$!
    wait "$sampler_sleep" || return 0
    sampler_sleep=
    if ! snapshot_container running >> "$report_dir/$label.resources.jsonl"; then
      printf '%s\n' '资源采样失败；请检查容器是否重启或退出。' >> "$report_dir/$label.monitor-errors.log"
      return 1
    fi
  done
}

load_once() {
  # 只运行指定并发点；正式点强制清空 WARMUP，预热按顺序遍历原始数据集。
  local -a load_args
  load_args=(-u "$concurrency")
  if [ "$phase" = warmup ]; then
    load_args+=(-i "$iterations" -e WARMUP=1)
  else
    load_args+=(-d "$duration" -e WARMUP=)
  fi
  "$k6" run "$load_script" \
    "${load_args[@]}" \
    --summary-trend-stats='avg,min,med,max,p(90),p(95),p(99)' \
    --summary-export="$report_dir/$label.json" \
    > "$report_dir/$label.log" 2>&1 || raw_exit=$?
  printf '%s\n' "$raw_exit" > "$report_dir/$label.exit-code.txt"
}

record_result() {
  # 将终态日志与 summary 交叉校验；QPS 使用成功完成 Counter 的真实总运行速率，不除以名义压测时长。
  local terminal_line completed=0 interrupted=0 terminal_ok=false summary='{}' summary_ok=false expected=0
  local before=null after=null monitor_ok=true
  terminal_line="$(LC_ALL=C rg 'complete and [0-9]+ interrupted iterations' "$report_dir/$label.log" | tail -n 1 || true)"
  if [[ "$terminal_line" =~ [[:space:]]0+/[0-9]+[[:space:]]VUs, ]] &&
     [[ "$terminal_line" =~ ([0-9]+)[[:space:]]complete[[:space:]]and[[:space:]]([0-9]+)[[:space:]]interrupted[[:space:]]iterations ]]; then
    completed="${BASH_REMATCH[1]}"
    interrupted="${BASH_REMATCH[2]}"
    terminal_ok=true
  fi
  if [ -s "$report_dir/$label.json" ] && summary="$(jq -ce 'select(.metrics|type=="object")' "$report_dir/$label.json")"; then
    summary_ok=true
  else
    summary='{}'
  fi
  if [ "$phase" = warmup ]; then expected="$iterations"; fi
  if [ -n "$container" ]; then
    before="$(cat "$report_dir/$label.before.json")"
    if [ -s "$report_dir/$label.after.json" ]; then after="$(cat "$report_dir/$label.after.json")"; else monitor_ok=false; fi
    if [ -s "$report_dir/$label.monitor-errors.log" ]; then monitor_ok=false; fi
  fi
  jq -n \
    --arg point_label "$label" \
    --arg phase "$phase" \
    --arg duration "$duration" \
    --arg started_at "$started_at" \
    --arg ended_at "$ended_at" \
    --arg client_nofile_soft "$client_nofile_soft" \
    --arg client_nofile_hard "$client_nofile_hard" \
    --arg sla_metric "$sla_metric" \
    --argjson sla_ms "$sla_ms" \
    --argjson concurrency "$concurrency" \
    --argjson expected "$expected" \
    --argjson completed "$completed" \
    --argjson interrupted "$interrupted" \
    --argjson terminal_ok "$terminal_ok" \
    --argjson summary_ok "$summary_ok" \
    --argjson summary "$summary" \
    --argjson raw_exit "$raw_exit" \
    --argjson before "$before" \
    --argjson after "$after" \
    --argjson monitor_ok "$monitor_ok" \
    '($summary.metrics // {} | with_entries(.value=(.value.values // .value))) as $m |
      ($m.stream_done.value // $m.stream_done.rate) as $done |
      ($m.request_complete_ms // {}) as $latency |
      (if $sla_metric=="p95" then "p(95)" elif $sla_metric=="p99" then "p(99)" else "max" end) as $key |
      {label:$point_label,phase:$phase,concurrency:$concurrency,duration:$duration,expected_iterations:$expected,
       started_at:$started_at,ended_at:$ended_at,raw_k6_exit:$raw_exit,
       client_nofile:{soft:$client_nofile_soft,hard:$client_nofile_hard,source:"inherited shell limits; unchanged"},
       terminal_present:$terminal_ok,summary_valid:$summary_ok,completed:$completed,interrupted:$interrupted,
       iterations:($m.iterations.count//null),successful:($m.requests_successful.count//0),failed:($m.requests_failed.count//0),
       done:$done,qps:($m.requests_successful.rate//null),qps_basis:"requests_successful.rate over total k6 run",
       complete_p90_ms:$latency["p(90)"],complete_p95_ms:$latency["p(95)"],complete_p99_ms:$latency["p(99)"],
       complete_max_ms:$latency.max,latency_ms:$latency,
       input_tokens:($m.llm_prompt_tokens//{}),output_tokens:($m.llm_completion_tokens//{}),
       input_tokens_total:($m.llm_prompt_tokens_total_counter.count//null),
       output_tokens_total:($m.llm_completion_tokens_total_counter.count//null),
       sla_metric:$sla_metric,sla_ms:$sla_ms,sla_value_ms:$latency[$key],
       resources_before:$before,resources_after:$after,monitor_passed:$monitor_ok,
       container_identity_stable:($before==null or ($after!=null and $before.host_init_pid==$after.host_init_pid
          and $before.started_at==$after.started_at)),
       transport_passed:(($raw_exit==0 or $raw_exit==99) and $terminal_ok and $summary_ok and $completed>0
          and $interrupted==0 and $m.iterations.count==$completed and $m.requests_successful.count==$completed
          and ($m.requests_failed.count//0)==0 and $done==1 and ($expected==0 or $completed==$expected)
          and ($m.requests_successful.rate|type)=="number" and $m.requests_successful.rate>0)} |
      .sla_passed=(.transport_passed and (.sla_value_ms|type)=="number" and .sla_value_ms<=$sla_ms) |
      .status=(if (.transport_passed|not) then "transport_failed"
               elif (.container_identity_stable|not) or (.monitor_passed|not) then "monitor_failed"
               elif .sla_passed then "passed" else "sla_failed" end)' \
    > "$report_dir/$label.result.json"
  jq -c . "$report_dir/$label.result.json" | tee -a "$report_dir/results.jsonl"
}

# 保存原始工作负载校验值，不修改输入、模型参数、随机选样或旧脚本内置阈值。
sha256sum "$load_script" "$dataset" > "$report_dir/$label.workload-sha256.txt"
if [ -n "$container" ]; then
  snapshot_container before > "$report_dir/$label.before.json"
  if [ "$sample_seconds" -gt 0 ]; then sample_resources 9>&- & sampler_pid=$!; fi
fi
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
load_once
ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stop_sampler
if [ -n "$container" ]; then
  if ! snapshot_container after > "$report_dir/$label.after.json"; then
    printf '%s\n' '结束资源采样失败。' >> "$report_dir/$label.monitor-errors.log"
  fi
fi
record_result

# SLA 失败仍保留本点且退出 0，供上层选择并发；任何传输、终态或监控异常都要求停止后续点。
if ! jq -e '.transport_passed and .container_identity_stable and .monitor_passed' "$report_dir/$label.result.json" >/dev/null; then
  if [ "$raw_exit" != 0 ] && [ "$raw_exit" != 99 ]; then exit "$raw_exit"; fi
  exit 2
fi
exit 0
