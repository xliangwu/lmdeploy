# V166 / RTX 3090 优化修改方法与版本记录

实验记录：2026-10-29，Asia/Hong_Kong；完整源码发布说明更新于2026-10-30。保留原有“原因 → 方法 → 文件 → 验证与边界”，补齐分页 KV、连接上限、并发容量及 Nsight 口径。源码整理不修改运行服务。

当前目标：**完整请求 p95≤400ms、零失败、零中断，并提高 QPS/可承受并发**。历史 sp01→sp04 和分页 A/B 使用 `<450ms`，对应结果保留原门槛。适用环境为 V166、RTX 3090、单 GPU；成绩来自充分预热、长输入、最多生成 3 token 的负载。

配套文档：[完整源码入口与构建说明](v166_3090_optimized.md)、[并发结果表与完整命令](concurrency_results_3090.md)、[Nsight 指标、命令与文件索引](nsight_analysis_3090.md)、[分页 KV 详细验证](paged_prefill_kv_validation.md)。

**GitHub 发布与路径说明：**`perf/v166-3090-optimized` 收录完整仓库源码、历史优化、当前输入编码缓存/短 query/分页 KV 候选、测试、Dockerfile、辅助脚本及文档/CSV。先前 `7b234c4` 仅提交文档，这是历史发布边界。文中的仓库外 `../../../artifacts/`、`../../../docs/` 及 `/home/`、`/data/` 路径用于定位原测试服务器上的证据，GitHub 不提供这些大报告、模型、私有输入、原始日志和外部工具。候选随源码发布不表示已部署或新增验收通过；复现仍需匹配工具链、基础镜像、模型和工作负载。

## 本次采用的版本与当前状态

“当前运行版本”固定为 **sp04**。完整源码分支包含输入编码缓存、短 query 与分页候选，源码默认不开启这些候选；全量源码不等同于已运行镜像。专用候选 Dockerfile 的显式开关另见下文。

| 项目 | 本轮整理时核验的状态 |
| --- | --- |
| 工作区 / 实验时 Git 基线 | `/home/yutao.he/lmdeploy-handoff-20261029/source`；2026-10-29 以 `399e1cc5b3b8a5eba0f868136a900cb1e508d239` 加当时未提交候选完成实验；现由 `perf/v166-3090-optimized` 收录，实验产物以当时源码快照和二进制哈希定位 |
| 运行容器 / 镜像 tag | `lmdeploy-v166-qps450`；`lmdeploy_cu128:v0.18.0_sp04_qps450_20261028` |
| 镜像 ID | `sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b` |
| 运行 native SHA-256 | `92b2aab612a4c50341bc05bb1635c46802d98091940606eda1c751591e5f2347` |
| 硬件 / 模型 | GPU0 RTX 3090、驱动570.144；V166 BF16，28层，12 Q / 2 KV heads，head_dim128，TP1 |
| 服务参数 | 23333端口、session16384、batch16、cache_max_entry_count=0.9、prefix caching开启 |
| 实际显式环境 | `TM_PREFILL_MAX_SPLITS=8`、`LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576`、`LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1` |
| 未显式设置的环境 | `TM_PREFILL_PAGED_KV`、`TM_PREFILL_QUERY_TILE`、`LMDEPLOY_QWEN2_ENCODE_CACHE`；当前运行sp04无这些候选优化 |
| 实际 nofile | soft1024 / hard524288；容量测试时新容器和客户端使用65536，尚未永久应用 |
| 最后恢复 | 2026-10-29 17:23:20 +08:00；测试服务容器均已停止，两个构建容器仅运行sleep |

运行快照见 [runtime-manifest.json](../../../artifacts/documentation-20261029/runtime-manifest.json)。本轮容量复核的 `batch16/客户端32` 为178.35 QPS、p95 220ms；`batch32/客户端48` 为176.10 QPS、p95 328ms，均零失败/中断。两个结果属于原sp04加测试资源配置，不是分页收益，也不表示该资源配置已部署。

## 先看代码来源

```text
v0.18.0 发布提交 110965c7
  ├─ main 后续 26 个提交 → 7998cdb → 优化提交65e8a7f → 文档399e1cc5、7b234c4 → perf/v166-3090-optimized
  └─ mentor 异步前处理提交 ccc395d → v0.18.0_optimize

实际验收镜像：mentor sp01 → GPU 分片 r3 → CPU 优化与预算配置 sp04
```

mentor 已把模板渲染和输入 encode 改为 `await asyncio.to_thread(...)`，文件为 [multimodal.py](../../lmdeploy/serve/processors/multimodal.py)，来源提交 `ccc395d`。**这是 mentor 的已有优化，不计作我们新增的成果。**

2026-10-29 已核对：mentor 与 sp04 容器中的该文件都与 `v0.18.0_optimize` 一致。当时源码分支基于 `main`，没有这两处异步调用，已测镜像通过继承保留它。2026-10-30完整源码整理时，将 `_get_text_prompt_input` 内模板渲染和encode的两处 `asyncio.to_thread` 最小移植到当前源码，保留main的空content处理、reasoning参数和其他逻辑，没有整文件回退或合并整个mentor分支。两条基础分支的 `src/turbomind/`、`tokenizer.py` 和 `turbomind.py` 原始内容一致。

新增 [test_text_preprocess_async.py](../../tests/test_lmdeploy/test_text_preprocess_async.py)，实际导入当前源码，以受控模板/tokenizer验证5项CPU行为：模板和encode阻塞worker时事件循环仍响应、空content与参数传递、reasoning回退、无效模板拒绝及worker异常传播。5项均通过，不需要模型或GPU。此次完整源码组合没有新增服务A/B或GPU性能验收，仍不能默认等价于历史镜像的全量行为与成绩。

## 1. CPU：减少首次输出反分词的重复工作

**原因：**原实现第一次返回文字时，要转换并解码整个输入前缀。在约 7300 token 输入、约 2～3 token 输出的负载中，首次初始化开销占比较大；该函数在异步请求处理中同步调用。

```text
原来：完整输入前缀 ──────────┐
                            ├→ 解码两遍、扣除旧前缀 → 新增文字
现在：输入尾部上下文 ────────┘
      模型推理仍使用完整输入；改变的只是输出解码状态。
```

**方法：**从输入尾部 16 个 token ID 开始建立状态；过滤后不足 8 个非空 token 字符串，就把窗口扩为 32、64……。保留原来的增量解码、全局 token 位置和不完整字符等待逻辑。仅显式开启的 Transformers Qwen2 fast + ByteLevel 生效；其他 tokenizer、自定义非零前缀状态走原路径。

**边界：**短输出解释收益大小，正确性依据是限定解码器的局部字节语义。16/8 是当前实现的窗口参数；代码尚未显式验证 UTF-8 截断边界，不能仅凭窗口长度保证任意输入等价。

**文件：**[lmdeploy/tokenizer.py](../../lmdeploy/tokenizer.py)，新增 `_initial_detokenize_tokens` 和启用条件。开关：`LMDEPLOY_QWEN2_BYTELEVEL_TAIL=1`；代码默认关闭。

**证据：**首次反分词 CPU 微测平均 6.788→0.064ms；29,516 个边界测试分块零差异；512 次真实推理的生成 token 经离线逐 token 重放，共 1,260 个分块新旧解码零差异。函数级微测不能当成服务提速倍数。

## 2. GPU：启用 prefill split-KV

**原因：**前缀缓存命中较多时，新增 query 少、历史 KV 长，热点 attention 的线程块数量不足。原实现又因 prefill/decode 共用分片临时缓冲，强制 prefill 只用 1 片。

```text
原来：少量线程块各自遍历很长的 KV
现在：KV 分成多个区间 → 多个线程块并行 → 归并完整 attention 输出
      prefill 写自己的临时缓冲；decode 写原来的缓冲。
```

**方法：**复用仓库已有的分片 kernel、分片选择和 softmax 归并；新增 prefill 专用的 `partial_O`、`partial_ML`、`split_cnt`。保留原有 stream 同步，避免两条并发路径覆盖中间结果。

**文件：**[unified_attention_layer.cc](../../src/turbomind/models/llama/unified_attention_layer.cc)、[unified_attention_layer.h](../../src/turbomind/models/llama/unified_attention_layer.h)。已有的 `attention_template.h`、`reduce.cu` 被复用，未重写算法。

**边界：**完整 KV 和原有 mask 继续参与计算；分片增加临时存储、读写和归并成本。浮点累加顺序变化不保证逐 bit 相同。

**证据：**r3 诊断中常见网格 12→96 CTA，attention 平均调用 333.83→119.08μs，窗口 SM Active 50.80%→67.32%。这是不同诊断窗口的趋势证据，非严格逐形状配对，也不是最终 sp04 的新采样。

## 3. GPU 配套：修正混合批次下标，并限制分片范围

**原因：**混合批次先放 decode、后放 prefill；独立 prefill 缓冲从局部位置 0 开始，直接写全局 query 下标会错位。

```text
全局 query：       0 1 2 3 | 4 5 6
类型：              decode | prefill
prefill 局部下标：         | 0 1 2
修复：局部下标 = 全局下标 - offset_q
```

**方法：**`split_cnt` 写入下标减去 `offset_q`，与部分输出及归并一致。新增环境开关 `TM_PREFILL_MAX_SPLITS`，部署显式设为 **8**；实际分片数由已有算法决定。容量上限为 `min(4096 / prefill_query数, 配置上限)`，本路径在 query 总数不超过 2048 等条件满足时启用，否则保留单片。

**文件：**[attention_universal.h](../../src/turbomind/kernels/attention/attention_universal.h)、上述 `unified_attention_layer.cc/.h`。设置 `TM_PREFILL_MAX_SPLITS=1` 可关闭这条 prefill 分片路径。

**边界与证据：**仅覆盖受支持的常规 attention 路径。FP16/BF16 各 96 组形状/分片及 8 组双 stream 检查通过；memcheck 内存检查 0 errors。128 路上限候选曾出现标签变化，已回退；8 路通过当时的 128/512 条固定样本验收，不代表所有未来输入必然输出相同。

## 4. 调度：显式设置上下文 token 预算

**原因：**CPU 路径加速后，还需协调单轮上下文调度规模，兼顾吞吐与长尾。

**方法：**新增 `_resolve_context_token_budget`，把环境变量接入引擎已有的 `max_context_token_num`，部署选择 **24576**。参数校验失败时直接报错；未设置时保留引擎默认行为。

```text
单轮上下文预算：原配置 16384 → 24576
单请求 session-len：仍为 16384；服务 max-batch-size：仍为 16
```

**文件：**[lmdeploy/turbomind/turbomind.py](../../lmdeploy/turbomind/turbomind.py)。开关：`LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576`。

**边界：**这是与代码优化共同验收的配置改动，没有单项严格消融；不能将总收益全部归因于预算或某一个函数。

**本轮补充的实际调度语义：**[context_token_resource.h](../../src/turbomind/models/llama/context_token_resource.h) 中，完整上下文长度为 `seq_len + inflight_new_tokens`；新增输入超过1 token或请求尚未active时，按完整上下文扣预算，已有active单token decode扣零。[scheduler.cc](../../src/turbomind/engine/scheduler.cc) 遇到当前请求无法放入预算会结束本轮接纳。约7345-token的请求即使命中约7318 token，仍按约7345计费，等长例子一轮可放3条prefill，4条需29380。实际混合长度会变化，不能将该例子当动态batch实测值。将服务batch16改为32不会放宽24576这一门槛；它是已确认的约束，不是已证明的唯一瓶颈。

## 5. 构建：保留 GEMM 静态注册对象

**原因：**本次 CMake 构建曾丢失 GEMM 静态注册对象，导致原生模块可以 import、但 `Gemm.data_types()` 为空，模型无法启动。

**方法：**调整链接关系，让 kernel archives 的 `WHOLE_ARCHIVE` 经普通接口传递，并保持在 core 前面。

**文件：**[kernels/gemm/CMakeLists.txt](../../src/turbomind/kernels/gemm/CMakeLists.txt)。这是保证构建可用的修复，不单独计作性能收益。

## 6. 验证、构建与交付辅助文件

| 用途 | 文件 | 记录要点 |
| --- | --- | --- |
| CUDA 回归及注册测试目标 | [test_prefill_split_kv.cu](../../src/turbomind/kernels/attention/test_prefill_split_kv.cu)、[attention/CMakeLists.txt](../../src/turbomind/kernels/attention/CMakeLists.txt) | 数值、mask、偏移、容量边界与双 stream |
| GPU 层构建与部署 | [build_prefill_split_kv.sh](../../tools/build_prefill_split_kv.sh)、[deploy_prefill_split_kv.sh](../../tools/deploy_prefill_split_kv.sh)、[Dockerfile.prefill-split-kv](../../docker/Dockerfile.prefill-split-kv) | 继承 mentor 镜像，仅替换编译的 TurboMind 模块；当前构建面向 SM86、单 GPU |
| Python 优化层与启动 | [Dockerfile.qwen2-bytelevel](../../docker/Dockerfile.qwen2-bytelevel)、[run_qps450.sh](../../tools/run_qps450.sh) | 继承 r3，仅替换两个 Python 文件并设置最终配置 |
| 业务与解码验证 | [verify_prefill_split_kv.py](../../tools/verify_prefill_split_kv.py)、[verify_bytelevel_detokenize.py](../../tools/verify_bytelevel_detokenize.py)、[verify_service_token_stream.py](../../tools/verify_service_token_stream.py) | 区分跨运行生成比较与同一 token 流解码比较 |
| 离线汇总 | [summarize_slo_runs.py](../../tools/summarize_slo_runs.py) | 读取已有 k6 JSON；不发推理请求 |

## 7. 输入编码缓存候选：复用重复片段的 token IDs

**原因：**前缀KV命中不能避免输入文本重新编码；旧Nsight热16/32的编码线程CPU均值约13ms。

**方法：**[tokenizer.py](../../lmdeploy/tokenizer.py) 挂接 [Qwen2ChunkEncoder](../../lmdeploy/tokenizer_chunk_cache.py)。仅在精确匹配的Qwen2 fast、NFC、BPE、正则和身份后处理条件下，沿安全换行边界切片；LRU默认2048条、按Python对象估算16MiB，锁保护缓存。编码配置改变，或带截断、填充及额外kwargs时回退原编码。只缓存输入token IDs，不缓存模型输出。

**开关与文件：**源码 `LMDEPLOY_QWEN2_ENCODE_CACHE` 默认0；[Dockerfile.encode-cache](../../docker/Dockerfile.encode-cache) 显式设1。配套 [verify_encode_cache.py](../../tools/verify_encode_cache.py)、[summarize_current_runs.py](../../tools/summarize_current_runs.py) 来自交接，本轮保留未修改。

**验证与边界：**交接记录包含真实输入、Unicode、回退、并发及淘汰检查；历史微测和服务结果见 [PERFORMANCE.md](../../../docs/PERFORMANCE.md)。本轮分页A/B与最新sp04容量测试均未启用编码缓存，不能将这些成绩归给该候选；当前sp04镜像未打包该模块。

## 8. 短 query 候选：32行 prefill tile

**原因：**高前缀命中时新增query很少，64行tile存在填充工作。

**方法：**在 [attention_sm80_128.cu](../../src/turbomind/kernels/attention/kernel/attention_sm80_128.cu) 注册带形状分派的wrapper；显式 `TM_PREFILL_QUERY_TILE=32` 且满足SM86、CP1、受支持因果head128及 `max_q_len≤32` 时选择32行，其余保持64行。[impl_16816.h](../../src/turbomind/kernels/attention/impl_16816.h) 将Q共享内存行块改为 `min(CTA_Q×CTA_H,64)`，[registrar.h](../../src/turbomind/kernels/attention/registrar.h) 增加 `add_kernel` 支持wrapper。

**文件：**上述3个源码、[test_prefill_split_kv.cu](../../src/turbomind/kernels/attention/test_prefill_split_kv.cu)、[Dockerfile.short-prefill](../../docker/Dockerfile.short-prefill)、[run_inference_optimized.sh](../../tools/run_inference_optimized.sh)。Dockerfile继承编码缓存镜像并开启32行，不能误当成只改分页的镜像。

**验证与边界：**交接已有32/64边界和数值检查；本轮仅纠正旧测试资源输出为 `registry_base_linear_prefill`，防止把wrapper基类的线性属性当分页内核资源。源码默认64；最新sp04容量与分页A/B均固定64，不叠加32行分支收益。

## 9. 分页 KV 候选：消除每层历史 KV 展平复制

**原因：**旧sp04即使前缀命中约99.64%，每层仍分配连续 `tmp_kv` 并把整个历史KV复制进去。已有Nsight热16中FlattenKV占累计kernel时间8.09%；按28层、BF16、2 KV heads、head128和约7345-token输入推导，物化约210.6MB、读写合计约421.2MB/请求。这是逻辑字节估算，不是实际DRAM计数。

```text
原路径：ProcessKV写分页缓存 → FlattenKV复制全部历史 → 连续KV attention
候选：  ProcessKV写分页缓存 ────────────────────────→ 页表直读 attention
```

**方法：**

- [unified_attention_layer.cc](../../src/turbomind/models/llama/unified_attention_layer.cc) 在合格路径省略 `tmp_kv` 与 `invokeFlattenKV_v2_`，保留新增KV生产、K的RoPE以及原stream依赖。
- [paged_prefill.h](../../src/turbomind/kernels/attention/paged_prefill.h) 统一生产者/消费者资格；开关 `TM_PREFILL_PAGED_KV=1`，默认关闭。范围限SM86、FP16/BF16、head128、Q12/KV2、page64、CP1、未量化、非MLA、因果；正常不支持场景保留线性路径。
- [paged_prefill_iterator.h](../../src/turbomind/kernels/attention/paged_prefill_iterator.h) 按原 `block::Layout` 读取真实页表，预计算层/head字节偏移，tile恰好对应64-token页；逆序游标在第一页饱和以兼容流水线末尾预取，不假设物理页连续。
- [attention.cu](../../src/turbomind/kernels/attention/attention.cu) 与wrapper把空连续指针作为明确分页标记；标记存在但参数失配时立即报错，禁止落入线性空指针读取。`TM_PREFILL_LOG_DISPATCH` 分别输出实际32/64分页内核资源。
- 页表和缓存回收继续使用引擎既有 `inflight/ready/done` 生命周期约束；未改decode算法和缓存分配器。静态生命周期审查与双stream检查不代替取消/恢复/淘汰故障注入。

**真实分页验证：**新增 [test_prefill_paged_kv.cu](../../src/turbomind/kernels/attention/test_prefill_paged_kv.cu) 及CMake target。候选连续指针置空、线性副本填NaN，使用乱序页表、非零层/表/混合batch偏移、尾页和只读共享前缀，覆盖ProcessKV、bias/RoPE、GQA、causal/window、split和双stream；另有独立FP64 CPU参考。FP16/BF16各18组形状/分片及各2组双stream，32/64行均通过；memcheck 0 errors；非法布局/非因果空连续指针fail-fast。最大分页/线性绝对差FP16为0.00012207、BF16为0.000488281。旧线性测试通过不计作分页证据。

**构建与镜像：**[build_paged_prefill.sh](../../tools/build_paged_prefill.sh) 使用当前工作区独立 `build-paged-kv`，只复用第三方源码，不复用旧项目objects；[Dockerfile.paged-prefill](../../docker/Dockerfile.paged-prefill) 继承sp04、仅替换原生模块并校验hash。候选镜像 `lmdeploy_cu128:paged-kv-candidate-20261029`，ID `sha256:fa987b96aa211ec01b736a1d8e60fef3e6a954aa9d02113262301b24c6fe23a1`，native SHA-256 `3e1ac6ae9569d2fb482714acdb9a8ef17cbdc60df5a697f89521f5e63372116a`。当前仅保留构建产物，候选服务已停止。

**性能与限制：**固定BF16 batch4/Q26/KV7345/split8/tile64算子链293.499→226.611μs，下降22.79%，不能换算为请求QPS；分页内核仍255寄存器/线程且有spill。单次服务A/B中16并发QPS+3.33%、32并发+0.19%基本持平，两版均有EMFILE，不能宣称饱和QPS提高。512条串行输出文本/finish_reason一致；后来并发16的两条差异已在原sp04复现，但未固定实际批次，不能扩大为全部并发输出等价。详见 [分页验证说明](paged_prefill_kv_validation.md) 与 [原始结果](../../../artifacts/paged-kv/RESULTS.md)。

## 10. 连接资源与完整请求容量验收

**原因：**旧服务soft nofile=1024，服务日志出现 `socket.accept: Too many open files`。本轮FD采样峰值1057，已超过旧上限；结束10秒后回到47，支持有限连接积累而非本轮持续泄漏的解释。

**方法：**测试容器显式 `--ulimit nofile=65536:65536`，k6客户端同为65536；不同时改变keepalive或客户端连接模式。保留模型、tokenizer、context24576、split8、tile64，比较服务batch16/32和客户端并发32/48/56/64。原容器不改配置，临时容器结束后恢复原sp04。

**文件与复现：**[benchmark_concurrency_point.sh](../../tools/benchmark_concurrency_point.sh) 保存全量k6 summary/log，交叉检查成功数、iterations、SSE done和终态中断，并按独立 `p95≤400` 判定；[search_concurrency_400.sh](../../tools/search_concurrency_400.sh) 支持 `VARIANT=sp04`、新报告目录/容器后缀及退出恢复；[confirm_sp04_concurrency.sh](../../tools/confirm_sp04_concurrency.sh) 复用已核验的batch32容器作3分钟确认。[accept_paged_prefill.sh](../../tools/accept_paged_prefill.sh) 属于较早的分页450ms固定A/B，不能替代400ms容量测试。具体部署、预热和各档命令见 [并发结果与命令](concurrency_results_3090.md)。

**结果与边界：**5个90秒探索点、2个3分钟确认点共143,424条正式请求，失败/中断均0；通过点32/48的p95分别220/328ms，更高56/64未通过。QPS约180形成平台，不因更多在途请求同比提高。48是本轮最高已通过档，不是未测整数并发或生产突发的硬上限；p99和最大值可高于400ms。当前服务仍是batch16、nofile1024，测试配置未永久部署。

## 11. Nsight 指标口径与文件维护

本次仅整理既有Systems/Compute证据，未重新采集。热16/32的SM Active为74.55%/74.66%，属于原sp04的稳态窗口；r3旧记录67.32%属于另一阶段；本轮nvidia-smi GPU Util 100%也不是同一个指标。热16窗口从phase36.49秒缩到首末kernel30.06秒，再各去掉1秒得到28.06秒；窗口内部等待和空隙仍计入。正式完整请求QPS/p95没有进行这项裁剪。

原始 `.nsys-rep`、SQLite、`.ncu-rep` 都保留在 `/home/yutao.he/lmdeploy-sp04-ns-20261029/`，大文件仅建立索引，不复制进源码。NCU仅保存4个辅助kernel，缺少attention/GEMM的stall和achieved occupancy；Systems有部分事件可能不完整告警。指标表、CPU阶段、kernel归因、历史采集命令、只读分析命令、工具版本及全部路径见 [Nsight整理](nsight_analysis_3090.md)。

## 历史结果口径与维护约定

| 历史同并发 A/B（slo_load.js） | 原 mentor 镜像 | sp04 组合优化 |
| --- | --- | --- |
| 16 VU，3 分钟 | 79.43 QPS，p95 264ms | 170.03 QPS，p95 122ms |
| 32 VU，3 分钟 | 80.86 QPS，p95 486ms，p99 632ms | 167.07 QPS，p95 389.05ms，p99 793ms |

两版使用同一 `slo_load.js`、同一 5000 条数据，各自先完成 5000 次预热；正式 A/B 均零请求失败。`slo_load.js` 在测试目录内，属于测量工具：固定排列、完整 SSE 计时和成功检查，与原 `test.js` 的随机抽样等逻辑不同。170 QPS 不能直接当作原 `test.js` 历史截图的复测结果。

仓库中的[历史450ms验收记录](qps450_3090.md)和[GPU实施记录](prefill_split_kv_3090.md)保留历史证据。原始旧结果在本地 `D:/aispeech/reports/lmdeploy-qps450-20261028`、开发机 `/home/yutao.he/lmdeploy-qps450-20261028`；旧源码 `/home/yutao.he/lmdeploy-prefill-splitkv` 保留为来源，当前工作区为 `/home/yutao.he/lmdeploy-handoff-20261029/source`。最新400ms结果在 `artifacts/concurrency-400-resumed-20261029/`，使用test.measured.js和nofile65536，不能与旧表直接计算单项收益。

历史优化代码提交为 `65e8a7f`，后续文档提交为 `399e1cc5` 和仅文档的 `7b234c4`；2026-10-29 的核验基于 `399e1cc5` 加候选工作区快照。完整源码分支 `perf/v166-3090-optimized` 随后收录候选实现、测试和构建/验收脚本，不改写历史实验的源码或二进制身份。运行状态表是核验快照，不保证之后持续不变；复核时应同时检查镜像ID、原生模块hash、启动参数与环境，不能只看Git HEAD或镜像tag。

后续每项改动沿用“**原因 → 方法/示意 → 文件 → 验证与边界**”记录；修改开关、适用范围或镜像组成时同步更新。尚未完成的字节边界检查、更长解码覆盖及逐项消融，不能记成已完成成果。
