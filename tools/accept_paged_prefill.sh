#!/usr/bin/env bash
# 作用：获准临时切换后，在同一张 GPU 对 sp04 与仅替换原生模块的分页候选做固定 16/32 验收，最后恢复 sp04。
# 示例：bash tools/accept_paged_prefill.sh --execute
# 注意：会重启/暂时停止 sp04；仅在分页算子验证通过且用户允许服务切换后执行，不采集 profiler。
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${ARTIFACT_ROOT:-$repo_root/../artifacts/paged-kv}"
report_dir="$artifact_root/service-acceptance"
old_container=lmdeploy-v166-qps450
new_container=lmdeploy-v166-paged-acceptance-20261029
probe_container="$new_container-native-check"
new_image=lmdeploy_cu128:paged-kv-candidate-20261029
baseline_tag=lmdeploy_cu128:v0.18.0_sp04_qps450_20261028
baseline_image=sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b
native_path=/opt/py3/lib/python3.12/site-packages/lmdeploy/turbomind/_turbomind.cpython-312-x86_64-linux-gnu.so
load_script=/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js
dataset=/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt
k6=/data/users/xueliang.wu/me-k6-tools/k6-sse
expected_args='["serve","api_server","/models/V166_mindspeed_mg2hf","--log-level","WARNING","--model-name","qwen","--tp","1","--session-len","16384","--max-batch-size","16","--cache-max-entry-count","0.9","--enable-prefix-caching","--server-port","23333"]'
touched=0
candidate_attempted=0
probe_attempted=0

# 无执行参数时仅说明影响；本脚本不会把 --execute 本身当作用户批准的证据。
if [ "${1:-}" != --execute ] || [ "$#" != 1 ]; then
  echo '执行计划：sp04重启 → 512条输出参考 → 5000条预热 → c16/c32各3分钟 → 分页候选同流程 → 恢复sp04。'
  echo '用户允许临时服务切换且kernel-validation.json通过后，才可传入 --execute。'
  exit 0
fi

wait_ready() {
  # 参数为容器名；在目标容器内部检查 API，避免残留候选占端口时误认 sp04 已恢复。
  local container="$1" attempt
  for ((attempt=0; attempt<60; ++attempt)); do
    [ "$(docker inspect -f '{{.State.Running}}' "$container")" = true ] || return 1
    if docker exec "$container" curl -fsS --max-time 2 http://127.0.0.1:23333/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

write_manifest() {
  # 参数为容器、版本、输出文件；仅保存公平性所需配置，禁止落盘完整 Env 中的不相关密钥。
  local container="$1" variant="$2" output="$3"
  docker inspect "$container" | jq \
    --arg variant "$variant" \
    '.[0] | {variant:$variant, container:.Name, image:.Image, path:.Path, args:.Args,
      env:(.Config.Env | map(capture("^(?<key>[^=]+)=(?<value>.*)$"))
        | map(select(.key as $k | ["TM_PREFILL_MAX_SPLITS","TM_PREFILL_PAGED_KV","TM_PREFILL_QUERY_TILE",
          "TM_PREFILL_LOG_DISPATCH","LMDEPLOY_CONTEXT_TOKEN_BUDGET","LMDEPLOY_QWEN2_BYTELEVEL_TAIL",
          "LMDEPLOY_QWEN2_ENCODE_CACHE"] | index($k))) | from_entries),
      gpu_requests:.HostConfig.DeviceRequests, shm_size:.HostConfig.ShmSize,
      privileged:.HostConfig.Privileged, port_bindings:.HostConfig.PortBindings,
      model_mounts:[.Mounts[] | select(.Destination=="/models") | {Source,Destination,RW}],
      running:.State.Running, started_at:.State.StartedAt}' > "$output"
}

check_manifest() {
  # 参数为清单、精确镜像 ID、分页开关值；关键启动参数或环境漂移时拒绝比较。
  local manifest="$1" image_id="$2" paged="$3"
  jq -e \
    --arg image "$image_id" \
    --arg paged "$paged" \
    --argjson args "$expected_args" \
    '.image==$image and .path=="/opt/py3/bin/lmdeploy" and .args==$args and .running
      and .env.TM_PREFILL_MAX_SPLITS=="8" and .env.LMDEPLOY_CONTEXT_TOKEN_BUDGET=="24576"
      and .env.LMDEPLOY_QWEN2_BYTELEVEL_TAIL=="1" and (.env.LMDEPLOY_QWEN2_ENCODE_CACHE // "0")=="0"
      and (.env.TM_PREFILL_PAGED_KV // "0")==$paged and (.env.TM_PREFILL_QUERY_TILE // "64")=="64"
      and .shm_size==8589934592 and .privileged==true and any(.gpu_requests[]; .DeviceIDs==["0"])
      and any(.port_bindings["23333/tcp"][]; .HostPort=="23333")
      and any(.model_mounts[]; .Source=="/data/resources/resource/olive.core.llm")' "$manifest" >/dev/null
}

restore() {
  # 退出前先停本轮尝试创建的候选，再恢复并核验 sp04；信号打断 docker run 也会按容器名清理。
  local result=$? candidate_stopped=1 restored=0
  trap - EXIT
  trap '' INT TERM
  set +e
  if [ "$probe_attempted" = 1 ] && docker container inspect "$probe_container" >/dev/null 2>&1; then
    docker rm "$probe_container" >/dev/null || result=1
  fi
  if [ "$candidate_attempted" = 1 ] && docker container inspect "$new_container" >/dev/null 2>&1; then
    docker logs "$new_container" > "$report_dir/candidate-server.log" 2>&1
    docker stop -t 20 "$new_container" >/dev/null || result=1
    if [ "$(docker inspect -f '{{.State.Running}}' "$new_container")" != false ]; then
      candidate_stopped=0
      result=1
    fi
  fi
  if [ "$touched" = 1 ]; then
    if docker start "$old_container" >/dev/null && wait_ready "$old_container"; then
      write_manifest "$old_container" sp04 "$report_dir/restored-sp04.json"
      if check_manifest "$report_dir/restored-sp04.json" "$baseline_image" 0; then restored=1; else result=1; fi
    else
      result=1
    fi
  fi
  jq -n --argjson exit_code "$result" --argjson sp04_touched "$touched" \
    --argjson candidate_stopped "$candidate_stopped" --argjson restored_sp04 "$restored" \
    '{exit_code:$exit_code,sp04_touched:$sp04_touched,candidate_stopped:$candidate_stopped,restored_sp04:$restored_sp04}' \
    > "$report_dir/restoration.json"
  echo "service_acceptance_exit=$result candidate_stopped=$candidate_stopped restored_sp04=$restored"
  exit "$result"
}

load_once() {
  # 参数为阶段标签及固定负载参数；99 只代表阈值失败，后续仍须检查请求成功和中断情况。
  local label="$1" result=0
  shift
  "$k6" run "$load_script" "$@" \
    --summary-trend-stats='avg,min,med,max,p(90),p(95),p(99)' \
    --summary-export="$report_dir/$label.json" > "$report_dir/$label.log" 2>&1 || result=$?
  printf '%s\n' "$result" > "$report_dir/$label-exit-code.txt"
  if [ "$result" != 0 ] && [ "$result" != 99 ]; then return "$result"; fi
}

record_result() {
  # 参数为版本、并发、阶段；终态日志和 JSON 互相校验，不能漏算被强制中断的请求。
  local variant="$1" concurrency="$2" phase="$3" label terminal_line completed interrupted expected=0
  label="$variant-c$concurrency"
  if [ "$phase" = warmup ]; then label="$variant-warmup"; expected=5000; fi
  terminal_line="$(LC_ALL=C rg 'complete and [0-9]+ interrupted iterations' "$report_dir/$label.log" | tail -n 1)"
  [[ "$terminal_line" =~ [[:space:]]0+/[0-9]+[[:space:]]VUs, ]] || return 1
  [[ "$terminal_line" =~ ([0-9]+)[[:space:]]complete[[:space:]]and[[:space:]]([0-9]+)[[:space:]]interrupted[[:space:]]iterations ]] || return 1
  completed="${BASH_REMATCH[1]}"
  interrupted="${BASH_REMATCH[2]}"
  jq --arg variant "$variant" --arg phase "$phase" --argjson concurrency "$concurrency" \
    --argjson completed "$completed" --argjson interrupted "$interrupted" --argjson expected "$expected" \
    '(.metrics | with_entries(.value=(.value.values // .value))) as $m |
      {variant:$variant, phase:$phase, concurrency:$concurrency, qps:$m.requests_successful.rate,
       requests:$m.iterations.count, failures:($m.requests_failed.count // 0), interrupted:$interrupted,
       complete_p95_ms:$m.request_complete_ms["p(95)"], complete_p99_ms:$m.request_complete_ms["p(99)"],
       stream_done:($m.stream_done.value // $m.stream_done.rate), input_p50:$m.llm_prompt_tokens.med,
       input_p95:$m.llm_prompt_tokens["p(95)"], output_mean:$m.llm_completion_tokens.avg,
       transport_passed:($completed>0 and $interrupted==0 and $m.iterations.count==$completed
         and $m.requests_successful.count==$completed and ($m.requests_failed.count // 0)==0
         and ($m.stream_done.value // $m.stream_done.rate)==1 and ($expected==0 or $completed==$expected))} |
      . + {sla_passed:(.transport_passed and (.complete_p95_ms|type)=="number" and .complete_p95_ms<450)}' \
    "$report_dir/$label.json" > "$report_dir/$label-result.json"
  if [ "$phase" = warmup ]; then
    jq -e '.transport_passed' "$report_dir/$label-result.json" >/dev/null
  else
    jq -c . "$report_dir/$label-result.json" >> "$report_dir/results.jsonl"
  fi
}

# 先检查依赖、固定负载和镜像身份；全程加锁，避免两个验收同时操作同一容器。
for command in docker curl jq rg sha256sum nvidia-smi python3 flock awk; do command -v "$command" >/dev/null; done
test -x "$k6"
test -s "$dataset"
test -s "$load_script"
test "$(awk 'NF {n++} END {print n}' "$dataset")" = 5000
mkdir -p "$artifact_root"
exec 9>"$artifact_root/service-acceptance.lock"
flock -n 9
test ! -e "$report_dir"
jq -e '.passed == true and (.native_sha256|type)=="string" and (.native_sha256|test("^[0-9a-f]{64}$"))' "$artifact_root/kernel-validation.json" >/dev/null
validated_sha="$(jq -r '.native_sha256' "$artifact_root/kernel-validation.json")"
test "$(docker image inspect "$baseline_tag" --format '{{.Id}}')" = "$baseline_image"
candidate_image_id="$(docker image inspect "$new_image" --format '{{.Id}}')"
baseline_layers="$(docker image inspect "$baseline_image" --format '{{json .RootFS.Layers}}')"
docker image inspect "$candidate_image_id" | jq -e --arg base "$baseline_image" --arg native "$validated_sha" --argjson layers "$baseline_layers" \
  '.[0] | .Config.Labels["ai.aispeech.paged-prefill.base-image-id"]==$base
    and .Config.Labels["ai.aispeech.paged-prefill.native-sha256"]==$native and .RootFS.Layers[0:($layers|length)]==$layers' >/dev/null
for container in "$new_container" "$probe_container"; do
  if docker container inspect "$container" >/dev/null 2>&1; then echo "容器 $container 已存在，拒绝覆盖。" >&2; exit 1; fi
done
if pgrep -x k6-sse >/dev/null || pgrep -x k6 >/dev/null || pgrep -x nsys >/dev/null || pgrep -x ncu >/dev/null; then
  echo '存在其他压测或profiler，拒绝不公平比较。' >&2; exit 1
fi
mkdir -p "$report_dir"
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_manifest "$old_container" sp04 "$report_dir/sp04-preflight-manifest.json"
check_manifest "$report_dir/sp04-preflight-manifest.json" "$baseline_image" 0

# 只创建不启动检查容器，读取镜像内真实模块 hash；不会加载模型或申请 GPU。
probe_attempted=1
docker create --name "$probe_container" --entrypoint /bin/true "$candidate_image_id" > "$report_dir/native-probe-id.txt"
docker cp "$probe_container:$native_path" "$report_dir/candidate-native.so"
actual_sha="$(sha256sum "$report_dir/candidate-native.so" | awk '{print $1}')"
test "$actual_sha" = "$validated_sha"
docker rm "$probe_container" >/dev/null
probe_attempted=0
printf '%s\n' "$actual_sha" > "$report_dir/candidate-native-sha256.txt"
sha256sum "$load_script" "$dataset" > "$report_dir/workload-sha256.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free --format=csv > "$report_dir/gpu.csv"

# sp04 对照候选仅替换原生模块；Python/tokenizer、预算、batch、split 和 query tile 均保持一致。
for variant in sp04 paged; do
  sha256sum --check "$report_dir/workload-sha256.txt" >/dev/null
  if [ "$variant" = sp04 ]; then
    touched=1
    docker restart -t 30 "$old_container" >/dev/null
    current="$old_container"; mode=capture; expected_image="$baseline_image"; expected_paged=0
  else
    docker stop -t 30 "$old_container" >/dev/null
    candidate_attempted=1
    docker run -d --name "$new_container" --gpus device=0 --privileged --shm-size 8g -p 23333:23333 \
      -e TM_PREFILL_MAX_SPLITS=8 -e TM_PREFILL_PAGED_KV=1 -e TM_PREFILL_QUERY_TILE=64 -e TM_PREFILL_LOG_DISPATCH=1 \
      -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 -e LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1 -e LMDEPLOY_QWEN2_ENCODE_CACHE=0 \
      -v /data/resources/resource/olive.core.llm:/models:ro "$candidate_image_id" serve api_server /models/V166_mindspeed_mg2hf \
      --log-level WARNING --model-name qwen --tp 1 --session-len 16384 --max-batch-size 16 --cache-max-entry-count 0.9 \
      --enable-prefix-caching --server-port 23333 > "$report_dir/candidate-container-id.txt"
    current="$new_container"; mode=compare; expected_image="$candidate_image_id"; expected_paged=1
  fi
  wait_ready "$current"
  write_manifest "$current" "$variant" "$report_dir/$variant-manifest.json"
  check_manifest "$report_dir/$variant-manifest.json" "$expected_image" "$expected_paged"
  python3 "$repo_root/tools/verify_prefill_split_kv.py" --mode "$mode" --dataset "$dataset" --reference "$report_dir/reference.json" --count 512 --concurrency 1 > "$report_dir/$variant-correctness.json"
  jq -e '.requests==512 and .errors==[] and ((.output_mismatches // [])==[])' "$report_dir/$variant-correctness.json" >/dev/null
  load_once "$variant-warmup" -u 16 -i 5000 -e WARMUP=1
  record_result "$variant" 16 warmup
  # 显式清空继承的 WARMUP，正式点保留原脚本随机抽样，只测已有 16/32 两档。
  for concurrency in 16 32; do
    load_once "$variant-c$concurrency" -u "$concurrency" -d 3m -e WARMUP=
    record_result "$variant" "$concurrency" formal
  done
  docker logs "$current" > "$report_dir/$variant-server.log" 2>&1
done

# 正确输出不代表分页实际执行；必须同时保留实际 BF16、64 行分页 kernel 的分派证据。
rg -q 'paged_prefill_dispatch dtype=bf16 query_tile=64' "$report_dir/paged-server.log"
jq -s -e 'length == 4 and all(.[]; .sla_passed)' "$report_dir/results.jsonl" >/dev/null
cat "$report_dir/results.jsonl"
