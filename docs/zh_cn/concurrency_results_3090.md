# RTX 3090：sp04 不同并发结果与复现命令

发布范围及服务器证据路径约定见[完整源码入口](v166_3090_optimized.md)和[修改方法文档](optimization_changes_3090.md)。`perf/v166-3090-optimized` 收录本页、CSV、源码和测试脚本；原始日志、大报告、模型、私有输入及仓库外 k6 工作负载仍保留在原服务器。以下命令按该环境记录，源码发布没有新增负载测试或更换服务。

记录日期：2026-10-29，Asia/Hong_Kong。本轮使用**原版 sp04，分页 KV 关闭**。在已测配置中，服务 batch16 / 客户端并发32 的3分钟确认结果为 **178.3463 QPS、完整请求 p95 220ms**；最高已通过的客户端并发为 **48**，对应服务 batch32，结果为 **176.1040 QPS、p95 328ms**。两者均满足本轮 **完整请求 p95 ≤ 400ms、零请求失败、零中断、SSE 正常结束** 的要求。

并发48比32多50%的在途请求，确认点QPS约低1.26%，延迟更高；这次短测没有统计显著性结论。48只是**已测档位中的最高通过值**，未测试所有整数并发，也未证明长期容量上限。p95 SLA不代表每条请求都在400ms以内。

## 七个正式点

数据来自[原始汇总](../../../artifacts/concurrency-400-resumed-20261029/summary.json)、[探索及并发32确认记录](../../../artifacts/concurrency-400-resumed-20261029/search-results.jsonl)和[并发48确认记录](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/final-confirmation.json)。[CSV表](concurrency_results_3090.csv)保留原始精度、p90/max、输入输出分布及每点证据路径；下表QPS显示四位小数。

| 阶段 / 点 | 服务 batch | 客户端并发 | 名义时长 | QPS | 完整 p95 ms | 完整 p99 ms | 成功完成请求 | 失败 | 中断 | p95≤400 SLA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 探索 b16-c32 | 16 | 32 | 90s | 180.1311 | 215 | 467 | 16,235 | 0 | 0 | 通过 |
| 探索 b16-c64 | 16 | 64 | 90s | 174.4375 | 678 | 868.49 | 15,752 | 0 | 0 | 不通过 |
| 探索 b32-c64 | 32 | 64 | 90s | 181.0228 | 545 | 604 | 16,332 | 0 | 0 | 不通过 |
| 探索 b32-c48 | 32 | 48 | 90s | 175.2525 | 328 | 622 | 15,808 | 0 | 0 | 通过 |
| 探索 b32-c56 | 32 | 56 | 90s | 170.6186 | 593.25 | 665 | 15,396 | 0 | 0 | 不通过 |
| 确认 b16-c32-confirm | 16 | 32 | 3m | 178.3463 | 220 | 396.76 | 32,125 | 0 | 0 | 通过 |
| 确认 b32-c48-confirm | 32 | 48 | 3m | 176.1040 | 328 | 606 | 31,776 | 0 | 0 | 通过 |

五个90秒探索点和两个3分钟确认点合计 **143,424 条正式请求，失败0、中断0，各点 `stream_done=1`**。另有四次5000条预热，均正常完成，未计入143,424。全部超出SLA的点保留；不能选取最高的181.0228 QPS却忽略其545ms p95。

QPS直接取k6 `requests_successful.rate`，使用包含正常请求收尾的真实运行时间，**没有用成功数除以名义90秒或180秒**。p95/p99/max取 `request_complete_ms`，计时覆盖整个请求迭代直到SSE完成，并在异常时通过 `finally` 记录耗时；不是首帧延迟。各点核验日志最终 `0/N VUs`、completed/interrupted、summary迭代数、成功/失败计数和DONE比例，避免漏算中断请求。

| 点 | 判定结果 | 原始 k6 summary | k6 完整日志 |
| --- | --- | --- | --- |
| b16-c32 | [结果](../../../artifacts/concurrency-400-resumed-20261029/b16-c32.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b16-c32.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b16-c32.log) |
| b16-c64 | [结果](../../../artifacts/concurrency-400-resumed-20261029/b16-c64.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b16-c64.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b16-c64.log) |
| b32-c64 | [结果](../../../artifacts/concurrency-400-resumed-20261029/b32-c64.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b32-c64.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b32-c64.log) |
| b32-c48 | [结果](../../../artifacts/concurrency-400-resumed-20261029/b32-c48.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b32-c48.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b32-c48.log) |
| b32-c56 | [结果](../../../artifacts/concurrency-400-resumed-20261029/b32-c56.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b32-c56.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b32-c56.log) |
| b16-c32-confirm | [结果](../../../artifacts/concurrency-400-resumed-20261029/b16-c32-confirm.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/b16-c32-confirm.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/b16-c32-confirm.log) |
| b32-c48-confirm | [结果](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/b32-c48-confirm.result.json) | [JSON](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/b32-c48-confirm.json) | [日志](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/b32-c48-confirm.log) |

## 固定配置、工作负载与判定方式

服务配置由[batch16清单](../../../artifacts/concurrency-400-resumed-20261029/b16-start2-manifest.json)、[batch32清单](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/target-running.json)和[模块校验](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/native-sha256.txt)确认。

| 项目 | 本轮固定值 |
| --- | --- |
| GPU / 模型 | GPU0，RTX 3090；`/models/V166_mindspeed_mg2hf`，V166/Qwen2 BF16，TP1 |
| 镜像 | `sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b`，原sp04 |
| 原生模块 SHA-256 | `92b2aab612a4c50341bc05bb1635c46802d98091940606eda1c751591e5f2347` |
| 启动入口 / 端口 | `/opt/py3/bin/lmdeploy serve api_server`；23333；模型名 `qwen` |
| session / KV缓存 | `session-len=16384`、`cache-max-entry-count=0.9`、前缀缓存开启 |
| context / split / query tile | `LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576`、`TM_PREFILL_MAX_SPLITS=8`、`TM_PREFILL_QUERY_TILE=64` |
| 分页 / 编码缓存 | `TM_PREFILL_PAGED_KV=0`、`LMDEPLOY_QWEN2_ENCODE_CACHE=0` |
| tokenizer / 日志 | `LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1`、`TM_PREFILL_LOG_DISPATCH=1`、`--log-level WARNING` |
| 容器资源 | `--gpus device=0`、`--privileged`、`--shm-size 8g`、`nofile=65536:65536` |
| 模型挂载 | `/data/resources/resource/olive.core.llm:/models:ro` |
| 客户端 | k6-sse；shell继承 `nofile=65536:65536`，连接行为保持原样 |
| 请求 | 流式SSE、`max_tokens=3`、`temperature=0`、`top_k=1`、`top_p=1`、`repetition_penalty=1`、返回usage |
| 预热 / 正式采样 | 每次重启顺序遍历同一5000条；正式阶段从同一数据池随机抽样 |
| SLA | 完整请求 `p95≤400ms`，成功数=完成数=迭代数，失败=0，中断=0，DONE比例=1 |

工作负载文件为 `/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js`，SHA-256为 `4621d1deeba247deda83d1bea996bf19cbf53f88ab9b73f8c67b147ad94bce3d`。数据为 `/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt`，SHA-256为 `abdf163be67a84181b30b8b32291d1ec2726b0d78477f6dca669419bc53fd663`。详见[输入与脚本校验](../../../artifacts/concurrency-400-resumed-20261029/input-sha256.txt)。原始输入未复制进文档。

| 3分钟确认点 | 输入 token 均值 | 输入 min / median / p95 / max | 输出 token 均值 | 输出 min / median / p95 / max |
| --- | ---: | --- | ---: | --- |
| batch16 / 并发32 | 7,340.665 | 556 / 4,594 / 12,262 / 13,182 | 2.3973 | 1 / 3 / 3 / 3 |
| batch32 / 并发48 | 7,360.002 | 556 / 4,595 / 12,264 / 13,182 | 2.3951 | 1 / 3 / 3 / 3 |

**原k6脚本仍内置 `p(95)<450`，本轮没有改它。** [单点runner](../../tools/benchmark_concurrency_point.sh)读取原始summary后独立按 `p95≤400` 判定；允许k6退出99表示其内置阈值失败，再核验传输是否完整，其他非零退出视为失败。SLA失败仍输出JSON并退出0供搜索选点，传输/中断/资源监控异常退出非零。因此不能仅凭k6退出0、原450阈值或runner退出0断言本轮SLA通过，必须检查结果JSON的 `transport_passed`、`sla_passed` 和 `status`。

## FD与当前恢复状态

本轮各正式点结束并等待10秒后，服务FD均回到47。3分钟并发32/48确认的FD采样峰值分别为1026/1051，全部正式点最高采到1057，已超过原服务1024的soft limit。采样间隔为5秒等待加采集耗时，峰值不是连续监控的真实最大值。原始样本分别见[并发32资源记录](../../../artifacts/concurrency-400-resumed-20261029/b16-c32-confirm.resources.jsonl)、[并发48资源记录](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/b32-c48-confirm.resources.jsonl)。本轮已保存服务日志未匹配EMFILE、CUDA/OOM等异常。

两次流程均已恢复原 `lmdeploy-v166-qps450`，结束时临时batch16/32容器停止。最后恢复记录见[恢复JSON](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/restoration.json)、[原服务恢复清单](../../../artifacts/concurrency-400-resumed-20261029/higher-concurrency-confirm/restored-sp04.json)及[独立复核](../../../artifacts/concurrency-400-resumed-20261029/independent-review.md)。记录中的原服务最后启动时间为 **2026-10-29 17:23:20 +08:00**，仍为batch16、原镜像和原模块，实际nofile恢复为 **1024 / 524288**。

**65536 nofile和batch32均未永久应用。** 本文是截至上述恢复记录的实验结论，不能描述为当前原服务已经具备临时测试配置。若采用容量结果，应同时保留对应已验证的nofile条件。

本轮各次batch启动的512条输出回归与旧参考同为串行，检查通过；batch32最终确认复用同一个已验证容器并再次核验镜像、配置、容器ID和模块哈希。相关证据为[batch16回归](../../../artifacts/concurrency-400-resumed-20261029/b16-start2-correctness.json)、[batch32回归](../../../artifacts/concurrency-400-resumed-20261029/b32-start1-correctness.json)。该检查不构成分页候选在任意并发批次下的输出等价证明。

## 两种已测服务配置

以下命令仅供复现，文档整理未执行。**两组均占用GPU0和宿主23333端口；执行前需先停止占用该GPU/端口的已有服务，每次只启动一组。** 结束测试后应停止测试容器并恢复原sp04。名称为新复现容器名，避免覆盖本轮保留的容器与证据。

服务batch16，对应推荐吞吐点客户端并发32：

```bash
docker run \
  -d \
  --name lmdeploy-sp04-b16-doc-reproduce \
  --gpus device=0 \
  --privileged \
  --shm-size 8g \
  --ulimit nofile=65536:65536 \
  -p 23333:23333 \
  -e TM_PREFILL_MAX_SPLITS=8 \
  -e TM_PREFILL_PAGED_KV=0 \
  -e TM_PREFILL_QUERY_TILE=64 \
  -e TM_PREFILL_LOG_DISPATCH=1 \
  -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 \
  -e LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1 \
  -e LMDEPLOY_QWEN2_ENCODE_CACHE=0 \
  -v /data/resources/resource/olive.core.llm:/models:ro \
  sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b serve api_server /models/V166_mindspeed_mg2hf \
  --log-level WARNING \
  --model-name qwen \
  --tp 1 \
  --session-len 16384 \
  --max-batch-size 16 \
  --cache-max-entry-count 0.9 \
  --enable-prefix-caching \
  --server-port 23333
```

服务batch32，对应最高已通过客户端并发48：

```bash
docker run \
  -d \
  --name lmdeploy-sp04-b32-doc-reproduce \
  --gpus device=0 \
  --privileged \
  --shm-size 8g \
  --ulimit nofile=65536:65536 \
  -p 23333:23333 \
  -e TM_PREFILL_MAX_SPLITS=8 \
  -e TM_PREFILL_PAGED_KV=0 \
  -e TM_PREFILL_QUERY_TILE=64 \
  -e TM_PREFILL_LOG_DISPATCH=1 \
  -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576 \
  -e LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1 \
  -e LMDEPLOY_QWEN2_ENCODE_CACHE=0 \
  -v /data/resources/resource/olive.core.llm:/models:ro \
  sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b serve api_server /models/V166_mindspeed_mg2hf \
  --log-level WARNING \
  --model-name qwen \
  --tp 1 \
  --session-len 16384 \
  --max-batch-size 32 \
  --cache-max-entry-count 0.9 \
  --enable-prefix-caching \
  --server-port 23333
```

## 同负载预热与正式点命令

每次服务重启后先确认API就绪，创建全新输出目录并设置与本轮相同的客户端nofile。在同一shell中执行后续命令：

```bash
curl -fsS http://127.0.0.1:23333/v1/models
export REPRO_REPORT="$(mktemp -d /home/yutao.he/lmdeploy-handoff-20261029/artifacts/concurrency-manual-XXXXXXXX)"
ulimit -n 65536
```

推荐使用runner，它会保存原始summary/log、检查终态和DONE、独立判断400ms SLA，并追加 `results.jsonl`。以下以batch16服务为例预热5000条；测试batch32时将 `CONTAINER` 换为 `lmdeploy-sp04-b32-doc-reproduce`，并在新的报告目录中同样预热：

```bash
export REPORT_DIR="$REPRO_REPORT"
export LABEL=warmup
export CONCURRENCY=16
export ITERATIONS=5000
export CONTAINER=lmdeploy-sp04-b16-doc-reproduce
export FD_SAMPLE_SECONDS=5
bash /home/yutao.he/lmdeploy-handoff-20261029/source/tools/benchmark_concurrency_point.sh
```

batch16 / 客户端并发32正式3分钟：

```bash
export REPORT_DIR="$REPRO_REPORT"
export LABEL=b16-c32-confirm
export CONCURRENCY=32
export DURATION=3m
export ITERATIONS=
export WARMUP=
export SLA_METRIC=p95
export SLA_MS=400
export CONTAINER=lmdeploy-sp04-b16-doc-reproduce
export FD_SAMPLE_SECONDS=5
bash /home/yutao.he/lmdeploy-handoff-20261029/source/tools/benchmark_concurrency_point.sh
```

batch32服务启动、同样预热后，客户端并发48正式3分钟：

```bash
export REPORT_DIR="$REPRO_REPORT"
export LABEL=b32-c48-confirm
export CONCURRENCY=48
export DURATION=3m
export ITERATIONS=
export WARMUP=
export SLA_METRIC=p95
export SLA_MS=400
export CONTAINER=lmdeploy-sp04-b32-doc-reproduce
export FD_SAMPLE_SECONDS=5
bash /home/yutao.he/lmdeploy-handoff-20261029/source/tools/benchmark_concurrency_point.sh
```

直接调用原k6的等价命令如下。它们保留原450ms内置阈值，**不包含runner的400ms独立判定、FD监控与终态交叉核验**；正式复现优先使用上面的runner。对同一次预热或同一个正式点，选择一种执行方式即可，不要重复运行两套。

顺序预热5000条，固定并发16：

```bash
/data/users/xueliang.wu/me-k6-tools/k6-sse run /home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js \
  -u 16 \
  -i 5000 \
  -e WARMUP=1 \
  --summary-trend-stats='avg,min,med,max,p(90),p(95),p(99)' \
  --summary-export="$REPRO_REPORT/warmup-direct.json" \
  > "$REPRO_REPORT/warmup-direct.log" 2>&1
```

batch16对应并发32，正式随机负载3分钟，显式清空继承的WARMUP：

```bash
/data/users/xueliang.wu/me-k6-tools/k6-sse run /home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js \
  -u 32 \
  -d 3m \
  -e WARMUP= \
  --summary-trend-stats='avg,min,med,max,p(90),p(95),p(99)' \
  --summary-export="$REPRO_REPORT/b16-c32-direct.json" \
  > "$REPRO_REPORT/b16-c32-direct.log" 2>&1
```

batch32对应并发48，正式随机负载3分钟：

```bash
/data/users/xueliang.wu/me-k6-tools/k6-sse run /home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js \
  -u 48 \
  -d 3m \
  -e WARMUP= \
  --summary-trend-stats='avg,min,med,max,p(90),p(95),p(99)' \
  --summary-export="$REPRO_REPORT/b32-c48-direct.json" \
  > "$REPRO_REPORT/b32-c48-direct.log" 2>&1
```

## 重现整轮搜索与较高并发确认

[search_concurrency_400.sh](../../tools/search_concurrency_400.sh)会切换服务：固定sp04镜像、预算和工作负载，运行五个90秒探索点，再对最高合格QPS点确认3分钟，最后恢复原sp04。执行前保持原sp04运行，停止其他使用同GPU/端口的手动复现容器。每次使用新的 `REPORT_DIR` 和 `RUN_SUFFIX`，避免覆盖已有JSON、日志及容器。以下生成新的唯一代号；此命令仅展示，不是本次文档整理中新增执行的测试：

```bash
export REPRO_SUFFIX="$(date +%Y%m%dT%H%M%S)-$$"
export REPRO_SEARCH_DIR="/home/yutao.he/lmdeploy-handoff-20261029/artifacts/concurrency-400-reproduce-$REPRO_SUFFIX"
export REPORT_DIR="$REPRO_SEARCH_DIR"
export VARIANT=sp04
export RUN_SUFFIX="$REPRO_SUFFIX"
export SLA_METRIC=p95
export ITERATIONS=
export WARMUP=
bash /home/yutao.he/lmdeploy-handoff-20261029/source/tools/search_concurrency_400.sh --execute
```

以上主流程只自动复核最高QPS点。对于本轮原始证据中的较高通过并发48，另用[confirm_sp04_concurrency.sh](../../tools/confirm_sp04_concurrency.sh)复用同轮已停止的batch32容器：先核验父目录的512条正确性证据、容器ID、原生模块哈希和配置，再预热5000条、测试3分钟、恢复原sp04。下面示例面向前一个命令新建的报告与容器；只有新一轮并发48探索仍通过时才将其作为合格容量确认：

```bash
export REPORT_DIR="$REPRO_SEARCH_DIR/higher-concurrency-confirm"
export CONCURRENCY=48
export CONTAINER="lmdeploy-v166-concurrency400-b32-$REPRO_SUFFIX"
export ITERATIONS=
export WARMUP=
bash /home/yutao.he/lmdeploy-handoff-20261029/source/tools/confirm_sp04_concurrency.sh --execute
```

本轮实际已完成的报告根目录为 `/home/yutao.he/lmdeploy-handoff-20261029/artifacts/concurrency-400-resumed-20261029`，容器后缀为 `20261029r2`。重现命令使用新目录，**不应把新结果覆盖回本轮原始证据**。

## 前一次分页 A/B：单独保留，不并入本轮

前一次[authorized-acceptance结果](../../../artifacts/paged-kv/authorized-acceptance/service-acceptance/results.jsonl)均为服务batch16、每点3分钟，分别测sp04与分页候选的客户端并发16/32：

| 版本 | 客户端并发 | QPS | 完整 p95 ms | 完整 p99 ms | 成功完成请求 | 失败 / 中断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sp04 | 16 | 160.5434 | 132 | 157 | 28,913 | 0 / 0 |
| sp04 | 32 | 174.0047 | 219 | 500.55 | 31,346 | 0 / 0 |
| paged | 16 | 165.8965 | 129 | 151 | 29,877 | 0 / 0 |
| paged | 32 | 174.3354 | 220 | 478 | 31,404 | 0 / 0 |

该组结果使用当时450ms验收口径，且两版服务日志都出现过文件描述符耗尽异常，见[前次独立复核](../../../artifacts/paged-kv/authorized-acceptance/service-acceptance/independent-review.md)。请求计数完整不等于服务日志无异常，不能忽略FD条件直接与本轮nofile65536结果合并。分页候选后续并发输出变化也需要独立受控对照；本轮143,424条正式请求与容量结论仅属于原sp04，不代表已部署分页优化或分页路径吞吐收益。
