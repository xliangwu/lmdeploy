# RTX 3090 / V166：既有 Nsight 证据与统计口径

本文整理 2026-10-29 已完成的 **一次 Nsight Systems 和一次 Nsight Compute 采集**，分析对象为 `lmdeploy_cu128:v0.18.0_sp04_qps450_20261028`。本次文档整理没有重新采集、运行负载或更换服务。这里的结果描述被采集的 sp04，不证明后续分页 KV 候选已通过正确性或取得部署收益。

配套文档：[修改方法与版本记录](optimization_changes_3090.md)、[最新并发结果与命令](concurrency_results_3090.md)。最新容量验收采用完整请求 p95≤400ms；下文450ms描述历史验收口径。

文件索引中的绝对路径及仓库外链接指向原测试服务器。`perf/v166-3090-optimized` 收录优化源码、候选、测试、构建/验收脚本及本文；Nsight 原始报告、仓库外分析工具和原始日志保留在服务器，不随 Git 上传。完整发布边界与构建入口见[完整源码说明](v166_3090_optimized.md)；本次源码发布未重新采集 Nsight。

模型为 V166/Qwen2、BF16、28 层、12 个 query heads、2 个 KV heads、head_dim 128，单 RTX 3090、TP1。历史服务参数为 session length 16384、max batch size 16、prefix caching 开启、`cache-max-entry-count=0.9`、`TM_PREFILL_MAX_SPLITS=8`、`LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576`。热32表示客户端并发32，服务 max batch size 仍为16。

最直接的结构证据是：热16缓存命中约99.64%，但每层仍运行 `flattenKV_v2`，占累计 kernel 时间8.09%。这支持优先消除历史 KV 的重复物化；该占比不能直接换算成完整请求加速8.09%。完整请求 p95<450ms、零失败及 QPS 的验收必须使用无 profiler 的服务负载结果。

## 1. 原始证据的有效性边界

Systems 的 `DIAGNOSTIC_EVENT` 明确保留了以下告警：

- `Not all NVTX events might have been collected.`
- `Not all CUDA events might have been collected.`
- `Not all OS runtime libraries events might have been collected.`

因此，kernel 数量、区间并集和 CPU/NVTX 事件统计都是**已记录事件的统计**，不能断言捕获完整。日志未见 GPU counter overflow 告警，也不能据此取消其他事件缺失告警。此次使用 `cuda-sw` 软件插桩 CUDA trace，而非无扰动采样。

NCU 的 full 指标采集进行了 kernel replay，日志包含 `Backing up device memory in system memory. Kernel replay might be slow...`，并提示未控制 GPU cache 和时钟。实际只保存4个辅助 kernel，不能把采集时的请求延迟作为正式性能，不能用这些 kernel 的 occupancy/stall 外推 attention 或 GEMM。

## 2. 时间窗口、分母与裁切算法

最终分析使用现有 `analyze_campaign.py` 对 SQLite 的只读查询。原目录与交接包的 `systems-analysis.json` SHA-256 一致：`106fe1bca6870a341b3b23661a2479a9efcdbb71743d631774c93a94d8e6faea`。

| 阶段 | NVTX phase 跨度（秒） | phase 内首末有效 kernel 跨度（秒） | 两端各去1秒后的窗口（秒） |
| --- | ---: | ---: | ---: |
| cold16 | 15.802895488 | 10.044360775 | 8.044360775 |
| warmup | 50.525683041 | 43.995493996 | 41.995493996 |
| hot16 | 36.493659968 | 30.064526736 | 28.064526736 |
| hot32 | 36.251970534 | 30.111154202 | 28.111154202 |

热16的 `36.49 → 30.06 → 28.06 秒` 是同一份报告的范围修正。phase 标记包含 k6 启动、抓取 `/metrics` 等外部空档；先定位 kernel 活跃跨度，再去掉两端过渡区。**窗口内部等待没有被删除**，不能理解成只选 GPU 忙碌时刻计算均值。按完整 phase 计算的 busy 与最终94.57%使用不同分母，不能据其差异声称性能提升；本文不沿用旧概述中未精确定位窗口的整 phase busy 约数。

首尾各1秒是用于避开负载建立与收尾过渡的经验缓冲，不是经验证的唯一合理阈值，也不保证窗口内已完全稳定。该窗口适合描述持续负载期间的 GPU 状态；裁剪可能提高利用率均值，因此74.55%的 SM Active 必须与窗口一起报告，不能称为全程平均。完整请求 QPS、p95 和失败数没有采用这一裁剪规则。

具体算法如下：

1. 从 `NVTX_EVENTS` 读取 `phase/*` 的起止时间。在 phase 内选择 `start >= phase_begin AND end <= phase_end AND end >= start` 的 kernel，求最早 start 和最晚 end。
2. `lo = first_start + 1e9`、`hi = last_end - 1e9`，分析窗口长度为 `hi - lo`，时间单位为纳秒。
3. 统计完整落在 `[lo, hi]` 内的 kernel；跨边界的 kernel 被排除，未裁切成片段。NVTX、CUDA API 和 OS runtime 区间同样要求完整落在窗口内。
4. kernel busy 是跨 stream 的 kernel 区间并集长度除以窗口长度。类别占比是该类别 kernel 时长之和除以所有 kernel 时长之和，两者分母不同。
5. gap 是相邻并集区间之间的内部空隙；`>1ms gap` 为长度严格大于1ms的内部空隙之和除以窗口。`100%-busy` 还包括窗口首尾没有被完整 kernel 覆盖的小缝，因此不与内部 gap 总和严格等价。
6. overlap 是各 stream 独立区间并集长度之和减去全局并集长度，再除以窗口。它描述已发生的累计重叠量，不是保证可以隐藏的优化时间。
7. `GPU_METRICS` 使用 `timestamp BETWEEN lo AND hi` 的1kHz样本，按样本算术平均；没有按 kernel 时长加权，也没有丢弃内部空闲样本。

CPU 叶节点采样是全报告的 `COMPOSITE_EVENTS` 联结 `SAMPLING_CALLCHAINS`，取 `stackDepth=0`，不采用上述裁切窗口。CPU 插桩 JSON 按调用开始时的 phase 归类；服务 `/metrics` 是阶段前后计数器差分。这三类 CPU/服务证据与 GPU 窗口需要分别解释。

## 3. GPU 活动与忙碌程度

以下均为百分数；`GPU Active`、SM、Tensor、warp、DRAM 行为窗口内样本均值，其余行按上一节算法计算。

| 指标 | 冷16 | 热16 | 热32 |
| --- | ---: | ---: | ---: |
| GPU Active | 99.95% | 97.52% | 97.03% |
| kernel 区间并集 / 窗口 | 98.43% | 94.57% | 94.19% |
| SMs Active | 75.58% | **74.55%** | **74.66%** |
| SM Issue | 7.92% | 7.69% | 7.76% |
| Tensor Active | 28.89% | 21.99% | 22.72% |
| Compute Warps in Flight | 12.76% | 13.67% | 13.16% |
| DRAM Read Bandwidth | 26.60% | 40.57% | 39.68% |
| DRAM Write Bandwidth | 5.60% | 5.81% | 6.14% |
| `100%-kernel busy` | 1.57% | 5.43% | 5.81% |
| >1ms 内部 gap / 窗口 | 0.05% | 2.42% | 2.97% |
| stream 并集和减全局并集 / 窗口 | 5.92% | 8.80% | 9.12% |
| 窗口内完整 kernel 数 | 139,162 | 1,003,101 | 964,089 |
| 每项 GPU metric 的样本数 | 8,044 | 28,065 | 28,112 |

SM Active 描述存在活跃 warp 的周期，SM Issue 描述指令发射活动，Tensor Active 描述 Tensor 管线活动；这些数值都不等于实际 FLOPS / 理论峰值 FLOPS。`Compute Warps in Flight` 是设备采样指标，不是某个 kernel 的 NCU achieved occupancy。DRAM 行是工具给出的吞吐百分比，不是读写字节数，也不是物理 KV 搬运量的直接计数。

GPU 大部分时间有 kernel 在运行，但 issue 和 Tensor 活动仍较低，说明“设备忙”与“计算单元充分执行有效计算”是不同事实。不能仅凭上述采样确定 attention 的 stall 原因，也不能把本轮或其他并发测试中 `nvidia-smi utilization.gpu=100%` 当成这里的 SM Active、Tensor Active 或新采集证据。

此前记忆中的“60多”与本次“74”来自不同报告：2026-10-28 r3 的 `followup-report.md` 比较旧 clean16 **50.80%** 与 new8 **67.32%**，另一次 threads8 为69.30%；对象为 `v0.18.0_sp02_splitkv_20261028_r3`，new8 使用上下文预算16384。本文的 **74.55% / 74.66%** 来自2026-10-29 sp04 的 hot16 / hot32，预算24576，且采用两端各去1秒的窗口。r3旧记录仅用于辨识历史来源；模型请求、缓存与窗口未完全配对，不能把这几个数拼成严格的单因素收益曲线，也不能把74.55%归因于后来的分页候选。

## 4. CPU、线程池与服务阶段

### 4.1 同 GPU 裁切窗口的 NVTX 墙钟统计

本表使用最终 `systems-analysis.json`，时间为每次调用平均毫秒，括号为完整落在窗口内的事件数。旧 `DIAGNOSIS.md` 的 CPU 摘要有少量数值差异，本表以可复核的最终 JSON 为准；没有把不同范围的均值合并成一个时间序列。

| NVTX 事件（省略 `diag/` 前缀） | 热16，毫秒（次数） | 热32，毫秒（次数） | 含义 |
| --- | ---: | ---: | --- |
| `thread_execute/HFChatTemplate.messages2prompt` | 0.263（4,239） | 0.295（4,505） | worker 内模板渲染 |
| `thread_execute/Tokenizer.encode` | 14.183（4,237） | 14.254（4,503） | worker 实际执行 encode 的外层区间 |
| `encode` | 14.169（4,237） | 14.241（4,503） | tokenizer 包装器内层区间 |
| `thread_wait/Tokenizer.encode` | 0.507（4,239） | 1.045（4,503） | 提交线程任务到 worker 开始 |
| `detokenize` | 0.065（11,646） | 0.072（9,924） | 每次增量反分词调用，非每请求 |
| `preprocess` | 19.910（4,237） | 38.811（4,501） | 异步预处理，包含 await 等待 |
| `engine_first_wait` | 40.190（4,234） | 52.922（4,496） | 进入引擎迭代到首次输出 |

`engine_next_wait` 包含迭代结束探测，不能当作每输出 token 的 decode 延迟。NVTX 事件会跨请求重叠，encode 也嵌套在 preprocess 内；表中各行不能相加成请求完成时间。

### 4.2 整个 phase 的 Python 插桩统计

`cpu-stages-systems.json` 使用 `perf_counter_ns` 测墙钟、`thread_time_ns` 测当前线程 CPU 时间，调用按开始时 phase 归类。它不裁切为上面的28秒窗口。

| 指标 | 热16 | 热32 | 样本数（热16 / 热32） |
| --- | ---: | ---: | ---: |
| `encode` 墙钟平均 | 14.072ms | 14.258ms | 4,556 / 4,836 |
| `encode` 当前线程 CPU 平均 | 12.976ms | 13.380ms | 4,556 / 4,836 |
| 事件循环额外延迟平均 | 2.748ms | 10.025ms | 2,862 / 1,811 |

事件循环使用10ms心跳，记录 `max(0, 实际间隔 - 10ms)`，正常睡眠没有计为 CPU 忙碌。编码约13ms线程 CPU 与0.5–1ms线程池等待表明，减少重复编码计算有直接依据；单纯扩大线程池并不能消除这段计算。

全报告共有328,976个 CPU 采样事件，叶节点模块主要为 tokenizers 81,568、libcuda 81,156、libc 75,852、Python 47,158，另有 Nsight 注入库13,241。它们用于定位热点，含诊断工具自身开销，不能解释成热16请求各阶段的时间百分比。

### 4.3 阶段 `/metrics` 差分

热16计4,556个请求、热32计4,836个请求。下列均值来自服务阶段计数器差分，非 GPU 裁切窗口。

| 服务指标，平均毫秒 | 热16 | 热32 |
| --- | ---: | ---: |
| queue | 8.777 | 12.022 |
| inference | 51.711 | 60.385 |
| prefill | 30.418 | 40.138 |
| decode | 21.292 | 20.248 |
| TTFT | 40.102 | 53.081 |
| backend e2e | 61.385 | 73.319 |
| 每新增输出 token 的后端时间 | 12.012 | 15.460 |

最后一行分别有7,962 / 5,874个输出计数，表示 LMDeploy 后端输出区间除以新增 token 数，不是客户端 SSE 事件到达间隔。`RequestStats` 在 `AsyncEngine.generate` 中创建，backend e2e 没有覆盖全部 HTTP 与 preprocess，因此完整请求 p95 的450ms门槛应使用测量副本中的 `request_complete_ms`：覆盖请求函数、完整 SSE 消费和原脚本用例处理。服务均值和 NVTX 均值不能人为相加替代这一指标。

热16/热32的输入平均为7,344.65 / 7,391.72 token，命中缓存平均为7,318.22 / 7,365.43 token，token 加权命中率为99.640% / 99.644%。高命中减少新 token 投影，但新 query 仍要关注长历史 KV，且旧生产路径仍将历史 KV 展平成连续布局。

## 5. Kernel 归因与代码含义

分类使用 kernel 名称规则：`AttentionCtaMap` 为 prefill、`DecodingCtaMap` 为 decode、`flattenKV` 为展平、包含 gemm/gemv 为 GEMM、包含 norm/silu/bias/reduce/processkv 为逐元素或归并。它是粗粒度名称归因，不是完整算子语义树。

| 类别 | 热16累计 kernel 时间占比 | 热32累计 kernel 时间占比 | 解释 |
| --- | ---: | ---: | --- |
| GEMM / GEMV | 47.61% | 45.94% | cuBLAS / CUTLASS 等矩阵计算 |
| prefill attention | 25.78% | 26.86% | 旧 `LinearIteratorFactory` 消费连续 KV |
| decode attention | 8.96% | 9.31% | `DecodingCtaMap`，使用分页 KV |
| KV flatten | 8.09% | 8.54% | 每层 `invokeFlattenKV_v2_` 搬运历史 KV |
| 逐元素 / 归并 | 9.21% | 8.99% | norm、ProcessKV、归并等 |
| 其他 | 0.35% | 0.36% | 上述名称规则未覆盖项 |

热16累计 kernel 时长分母为29.008917273秒，热32为29.040484772秒；它们因跨 stream 重叠而大于28.0645 / 28.1112秒的墙钟窗口。每类占比不等于请求墙钟占比。

旧 prefill kernel 的模板为 SM80 MMA16816、CTA_Q=64、head_dim128，在 SM86 设备执行，常见 grid `(1,2,72)`、block `(128,1,1)`、动态共享内存32,768字节、253 registers/thread。热16平均132.42µs，热32平均141.62µs。这里的寄存器数来自 Systems kernel launch 元数据，不能替代 NCU 的硬件 occupancy/stall 证据。

`flattenKV_v2` 热16平均41.54µs、热32平均45.02µs。生产调用位于 `unified_attention_layer.cc`：写入新增 KV 的 `invokeProcessKV_v2_` 后，为线性 prefill 消费者准备完整历史 KV 中间缓冲。按2个 KV heads、head_dim128、BF16、28层与平均约7,345 token推算，一请求需物化约210.6MB KV，计读写约421.2MB张量流量。该形状推导不等于实际 DRAM 字节，片上缓存会影响真实显存流量。

热16 `cudaLaunchKernel` 平均6.83µs（938,326次），`cudaStreamSynchronize` 平均10.92ms（2,217次）；热32分别6.06µs和11.07ms。同步调用包含等待 GPU 的时间，累计 API 时间不能全算为可消除的 CPU 开销。

这些证据支持审查页表、缓存生命周期和生产者/消费者布局，完成 prefill 分页直读。能否删除 flatten、数值是否一致、寄存器与溢出是否恶化，以及完整请求是否受益，仍需分页路径本身的针对性验证；旧线性测试和旧 Nsight 报告都不能代替这一步。

## 6. Nsight Compute 实际覆盖

`ncu-summary.json` 与原始报告实际保存以下4个 launch，均为 grid `(1,1,1)`：

| ID | Kernel | Block | Duration | Registers/thread | Achieved occupancy |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | ClearTokenMaskKernel | `(128,1,1)` | 1.66µs | 16 | 8.18% |
| 1 | BatchPrefixSumKernel | `(256,1,1)` | 2.21µs | 29 | 16.26% |
| 2 | embeddingLookupKernel | `(192,1,1)` | 2.30µs | 18 | 12.63% |
| 3 | RMSNorm | `(512,1,1)` | 2.72µs | 36 | 30.44% |

这些是小网格辅助 kernel 的观测，不能代表整个模型或 attention/GEMM 的占用率。历史脚本注释原本希望覆盖 attention/GEMM/归并/KV，但实际结果没有达到这一覆盖目标；`--launch-count 32` 是上限，也不保证报告必有32个 kernel。

**本次没有 attention 或 GEMM 的 achieved occupancy、eligible warps、硬件 stall 证据。** 不用旧版报告、理论 occupancy、Systems 寄存器数或设备 warp 指标补齐缺项，也不因此重新采集。

## 7. 诊断插桩与历史采集命令

`profile_entry.py` 在诊断进程中包装已安装 LMDeploy 函数，不改写生产源码文件：同步调用记录非栈式 NVTX range 与墙钟/当前线程 CPU；`asyncio.to_thread` 分开记录排队和 worker 执行；`AsyncEngine.preprocess` 记录包含 await 的异步区间；`TurboMindInstance.async_stream_infer` 分开记录首次/后续输出等待并排除 consumer 处理时间。控制服务仅监听容器 `127.0.0.1:49555`，phase 路径切换 NVTX 区间，start/stop 调用 CUDA profiler API，snapshot 写 CPU JSON。

下面是 `run_ns_campaign.sh` 的**历史命令记录，仅用于核对已有报告，不执行**。该脚本会停止正常服务、启动诊断容器并产生负载，不能当作离线分析入口；现有 `campaign.started` 是一次性保护，也不得删除后重跑。

历史命令使用的变量与数组：

```bash
# 历史采集记录：仅供阅读，不执行本代码块或 run_ns_campaign.sh。
R=/home/yutao.he/lmdeploy-sp04-ns-20261029
T=/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools
IMAGE=lmdeploy_cu128:v0.18.0_sp04_qps450_20261028
SYS=lmdeploy-v166-sp04-nsys-20261029
NCU=lmdeploy-v166-sp04-ncu-20261029
NS=/tools/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
COMMON=(
  --gpus device=0
  --privileged
  --shm-size=8g
  -p 23333:23333
  -e TM_PREFILL_MAX_SPLITS=8
  -e LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576
  -v /data/resources/resource/olive.core.llm:/models:ro
  -v "$R:/diagnostic"
  -v "$T:/tools:ro"
)
ARGS=(
  serve api_server /models/V166_mindspeed_mg2hf
  --log-level WARNING
  --model-name qwen
  --tp 1
  --session-len 16384
  --max-batch-size 16
  --cache-max-entry-count 0.9
  --enable-prefix-caching
  --server-port 23333
)
```

Systems 历史启动与开始采集命令：

```bash
# 历史采集记录，禁止作为本轮执行命令。
docker run \
  -d \
  --name "$SYS" \
  "${COMMON[@]}" \
  -e PROFILE_KIND=systems \
  --entrypoint "$NS" \
  "$IMAGE" launch \
  --session-new=sp04once \
  --trace=cuda-sw,nvtx,osrt,cublas \
  --python-sampling=true \
  --python-sampling-frequency=100 \
  --cuda-graph-trace=node \
  --cuda-flush-interval=1000 \
  /opt/py3/bin/python /diagnostic/profile_entry.py "${ARGS[@]}"

docker exec "$SYS" "$NS" start \
  --session=sp04once \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --sampling-period=3500000 \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=1000 \
  --discard-environment=true \
  --output=/diagnostic/sp04-once
```

这里 Python 采样100Hz，GPU metric 采样1kHz，原生 CPU 采样配置为参考周期3,500,000，并采集进程树上下文切换。历史负载顺序为冷16持续10秒、16并发5000条预热、热16持续30秒、热32持续30秒，然后 snapshot 和 `nsys stop --session=sp04once`。phase 跨度大于负载设定时长的原因见第2节。

Compute 历史命令：

```bash
# 历史采集记录，禁止作为本轮执行命令。
docker run \
  -d \
  --name "$NCU" \
  "${COMMON[@]}" \
  -e PROFILE_KIND=compute \
  --entrypoint /tools/ncu/ncu \
  "$IMAGE" \
  --target-processes all \
  --profile-from-start off \
  --set full \
  --kernel-name-base demangled \
  --kernel-id '::regex:.*:1' \
  --launch-count 32 \
  --replay-mode kernel \
  --cache-control none \
  --clock-control none \
  -o /diagnostic/sp04-kernels \
  /opt/py3/bin/python /diagnostic/profile_entry.py "${ARGS[@]}"
```

Compute 在5000条预热后开启 `compute16` phase 和 CUDA profiler API，历史诊断负载为16并发20秒，然后 stop/snapshot。Systems 与 Compute 分开运行，full replay 对采集负载有明显扰动。

## 8. 现有报告的离线查看与分析

以下命令只读取已有报告；导出和分析命令仅写新产物。整理本文时仅查询了已有 JSON、日志、SQLite元数据、文件元数据及工具帮助/版本，没有执行重新导出或全量重新分析。

查看最终 GPU 汇总和 NCU 覆盖列表：

```bash
jq '.efficiency' /home/yutao.he/lmdeploy-handoff-20261029/docs/analysis-digest.json
jq '[.[] | {id, kernel, grid, block}]' /home/yutao.he/lmdeploy-handoff-20261029/docs/ncu-summary.json
```

使用已有脚本从 SQLite 复核同一统计算法；SQLite 以 `mode=ro` 打开，只在 `/tmp` 写新 JSON，宿主已有 Python 标准库足够，无需安装依赖：

```bash
/home/yutao.he/miniconda3/bin/python /home/yutao.he/lmdeploy-sp04-ns-20261029/analyze_campaign.py /home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-once.sqlite /tmp/sp04-systems-analysis-review.json
```

如确需从原报告重建 SQLite，使用已安装 Systems 的离线导出；现有 SQLite 已可直接使用，通常无需再生成约1.3GiB文件。下面指定新路径，不覆盖原报告：

```bash
/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/nsight-systems-cli/2026.5.1/target-linux-x64/nsys export \
  --type=sqlite \
  --output=/tmp/sp04-once-review.sqlite \
  /home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-once.nsys-rep
```

只导入已存在的 Compute 报告，输出 details CSV；此命令不启动模型、不发请求、不采集 GPU：

```bash
/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/ncu/ncu \
  --import /home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-kernels.ncu-rep \
  --page details \
  --csv
```

这些导出参数已与现有工具帮助核对。查看历史采集实现时应直接阅读脚本，**不要运行 `bash run_ns_campaign.sh`**。

## 9. 工具版本与现有安装位置

Systems 版本由历史 `nsys-environment.txt` 直接确认，为 `2026.5.1.161-265138896106v0`；SQLite 元数据的导出工具版本为 `2026.5.1.161`，schema为 `3.31.0`。历史 CPU 环境检查记录 root enabled、perf_event_open 可用、进程树与系统级采样环境可用，宿主内核为 `5.14.0-611.24.1.el9_7.x86_64`、paranoid为2。这些是历史环境检查结果，不是对任意机器的保证。

Compute 现有安装的静态 HTML 文档标明 `v2025.1.1`，同一路径的只读 `--version` 查询为 `2025.1.1.0`、build `35528883`。历史脚本绑定的工具根目录与当前核验目录相同；没有把 Systems 的2026.5版本误写成 Compute 版本。

| 主机绝对路径 | 容器对应位置 / 用途 | 存在性与大小（字节） |
| --- | --- | ---: |
| [nsys](/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/nsight-systems-cli/2026.5.1/target-linux-x64/nsys) | `/tools/nsight-systems-cli/2026.5.1/target-linux-x64/nsys`；Systems程序 | 已核验，89,343,040 |
| [ncu 启动器](/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/ncu/ncu) | `/tools/ncu/ncu`；按架构选择目标程序 | 已核验，310 |
| [ncu x86-64 程序](/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/ncu/target/linux-desktop-glibc_2_11_3-x64/ncu) | 启动器实际调用的 Compute 程序 | 已核验，62,706,344 |
| [Compute 静态版本文档](/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/tools/ncu/docs/NsightCompute/index.html) | 文档页明确写有 `v2025.1.1` | 已核验，231,276 |

## 10. 证据文件索引

以下文件均在本次文档整理时检查存在性并读取文件元数据。大小为精确字节数，不代表完整性校验；原始大报告只索引，不复制进仓库。链接目标均为服务器实际绝对路径。默认先读小型 JSON/文本，必要时读取现有 SQLite 或原报告。

### 10.1 原始报告、插桩与日志

根目录：`/home/yutao.he/lmdeploy-sp04-ns-20261029`。

| 文件 | 用途 | 存在性与大小（字节） |
| --- | --- | ---: |
| [sp04-once.nsys-rep](/home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-once.nsys-rep) | 唯一一次 Systems 原报告，约304.26MiB | 已核验，319,043,223 |
| [sp04-once.sqlite](/home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-once.sqlite) | 同报告导出的事件数据库，约1.291GiB；分析只读打开 | 已核验，1,386,258,432 |
| [sp04-kernels.ncu-rep](/home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-kernels.ncu-rep) | 唯一一次 Compute 原报告，4个辅助 kernel | 已核验，3,665,955 |
| [profile_entry.py](/home/yutao.he/lmdeploy-sp04-ns-20261029/profile_entry.py) | 历史诊断 NVTX、线程 CPU、心跳和控制端点插桩 | 已核验，7,192 |
| [run_ns_campaign.sh](/home/yutao.he/lmdeploy-sp04-ns-20261029/run_ns_campaign.sh) | 历史采集/负载/恢复服务脚本；只读，不执行 | 已核验，5,871 |
| [analyze_campaign.py](/home/yutao.he/lmdeploy-sp04-ns-20261029/analyze_campaign.py) | SQLite只读分析算法，输出汇总 JSON | 已核验，8,926 |
| [systems-analysis.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/systems-analysis.json) | 窗口、kernel、NVTX、CPU采样、原始告警汇总 | 已核验，464,519 |
| [cpu-stages-systems.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/cpu-stages-systems.json) | Systems 全 phase Python墙钟/线程CPU/心跳统计 | 已核验，10,797 |
| [cpu-stages-compute.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/cpu-stages-compute.json) | Compute期间CPU插桩；受重放扰动，不作正式延迟 | 已核验，4,790 |
| [nsys-environment.txt](/home/yutao.he/lmdeploy-sp04-ns-20261029/nsys-environment.txt) | Systems精确版本和历史采样权限检查 | 已核验，732 |
| [nsys-start.log](/home/yutao.he/lmdeploy-sp04-ns-20261029/nsys-start.log) | 采集启动记录 | 已核验，40 |
| [nsys-stop.log](/home/yutao.he/lmdeploy-sp04-ns-20261029/nsys-stop.log) | 采集停止、事件处理及报告生成记录 | 已核验，1,608 |
| [nsys-service.log](/home/yutao.he/lmdeploy-sp04-ns-20261029/nsys-service.log) | Systems诊断容器服务日志 | 已核验，1,845,553 |
| [ncu-service.log](/home/yutao.he/lmdeploy-sp04-ns-20261029/ncu-service.log) | Compute重放、主机内存备份和cache/clock警告 | 已核验，605,406 |
| [ncu-details.csv](/home/yutao.he/lmdeploy-sp04-ns-20261029/ncu-details.csv) | 已导出的4个kernel指标和规则 | 已核验，106,734 |
| [tool-help.txt](/home/yutao.he/lmdeploy-sp04-ns-20261029/tool-help.txt) | 历史工具帮助，含NCU import/page/csv参数 | 已核验，70,715 |
| [analysis.log](/home/yutao.he/lmdeploy-sp04-ns-20261029/analysis.log) | 离线分析各phase完成记录 | 已核验，96 |
| [gpu-telemetry.csv](/home/yutao.he/lmdeploy-sp04-ns-20261029/gpu-telemetry.csv) | 采集期间1秒间隔温度、功耗、时钟与GPU Util | 已核验，20,702 |
| [campaign.started](/home/yutao.he/lmdeploy-sp04-ns-20261029/campaign.started) | 一次性采集启动标记，不删除重跑 | 已核验，26 |
| [campaign.finished](/home/yutao.he/lmdeploy-sp04-ns-20261029/campaign.finished) | 历史流程完成时间标记 | 已核验，26 |

### 10.2 汇总、旧来源与负载计量

| 文件绝对路径 | 用途 | 存在性与大小（字节） |
| --- | --- | ---: |
| [交接 analysis-digest.json](/home/yutao.he/lmdeploy-handoff-20261029/docs/analysis-digest.json) | 本文GPU效率、服务指标差分与范围摘要 | 已核验，15,086 |
| [交接 systems-analysis.json](/home/yutao.he/lmdeploy-handoff-20261029/docs/systems-analysis.json) | 与原目录版本SHA-256一致 | 已核验，464,519 |
| [交接 ncu-summary.json](/home/yutao.he/lmdeploy-handoff-20261029/docs/ncu-summary.json) | 本文4个NCU kernel指标；原采集目录无同名文件 | 已核验，58,214 |
| [交接 DIAGNOSIS.md](/home/yutao.he/lmdeploy-handoff-20261029/docs/DIAGNOSIS.md) | 原诊断说明；CPU精确数值以最终JSON为准 | 已核验，8,549 |
| [交接 ROOT_CAUSE.md](/home/yutao.he/lmdeploy-handoff-20261029/docs/ROOT_CAUSE.md) | 历史KV重复物化根因与分页验证边界 | 已核验，4,311 |
| [2026-10-28 r3 followup-report.md](/home/yutao.he/lmdeploy-profile-v166-20261027/new-metrics-20261028/followup-report.md) | 50.80%→67.32%等旧阶段数值来源；不能当成本次sp04采样 | 已核验，9,555 |
| [原始 test.js](/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/test.js) | 原始随机选样与模型请求定义 | 已核验，5,357 |
| [test.measured.js](/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js) | 修复依赖/缺失参考标签、补完整SSE请求计量的副本 | 已核验，7,101 |
| [原始5000条输入数据](/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt) | 历史预热/随机负载数据；只核验元数据，不复制私有内容 | 已核验，158,950,591 |
| [k6-sse](/data/users/xueliang.wu/me-k6-tools/k6-sse) | 历史SSE负载程序；本文不执行 | 已核验，50,196,664 |
| [systems-cold16-summary.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/systems-cold16-summary.json) | 冷16诊断负载汇总；含profiler开销 | 已核验，5,476 |
| [systems-hot16-summary.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/systems-hot16-summary.json) | 热16诊断负载汇总；不替代无profiler验收 | 已核验，5,431 |
| [systems-hot32-summary.json](/home/yutao.he/lmdeploy-sp04-ns-20261029/systems-hot32-summary.json) | 热32诊断负载汇总；不替代无profiler验收 | 已核验，5,431 |
| [hot16-metrics-before.txt](/home/yutao.he/lmdeploy-sp04-ns-20261029/hot16-metrics-before.txt) | 热16前服务计数器 | 已核验，33,015 |
| [hot16-metrics-after.txt](/home/yutao.he/lmdeploy-sp04-ns-20261029/hot16-metrics-after.txt) | 热16后、热32前服务计数器 | 已核验，33,174 |
| [hot32-metrics-after.txt](/home/yutao.he/lmdeploy-sp04-ns-20261029/hot32-metrics-after.txt) | 热32后服务计数器 | 已核验，33,184 |

原始 `test.js` SHA-256为 `8661249419f14e94bae0ebc32e9380f29c1b7ef6d1c91ad7c38724b4427be310`。测量副本保持模型请求和随机选样：`max_tokens=3`、`temperature=0`、`top_k=1`、`top_p=1`、`repetition_penalty=1`；4条缺失参考标签的输入仍发送，仅跳过缺失标签的质量比较。原脚本首事件时间包含空内容/元数据事件，不能直接当作首内容 TTFT。
