# Prefill 分页 KV 直读：证据、契约与定向验收

`perf/v166-3090-optimized` 收录本验证说明、分页候选源码、真实页表测试、Dockerfile 和构建/验收脚本。原始证据、大报告、模型及私有输入保留在服务器，发布边界及克隆/构建入口见[完整源码说明](v166_3090_optimized.md)。候选默认关闭，源码发布没有部署新服务或扩大下述验证结论。

实验记录时间：2026-10-29（UTC+8），首次现场检查为 15:21，已补齐本轮完整验收结果；源码发布说明更新于2026-10-30。实验代码基线为 `399e1cc5b3b8a5eba0f868136a900cb1e508d239`，候选在实验时为未提交改动，现随完整源码分支收录；实验源码快照和二进制 SHA-256 保存在仓库外 `artifacts/paged-kv/authorized-acceptance/`。真实分页数值与512条串行完整模型文本/finish_reason对照通过，固定 16 并发本轮 QPS +3.33%，32 并发基本持平（+0.19%）。两侧文件描述符上限异常限制了吞吐结论，详见末尾。

目标为完整请求 `p95 < 450 ms`、请求失败为零、流正常结束，同时提高 QPS。只验证删除每层历史 KV 展平复制这一结构修改；不恢复并发/context 预算扫描，不新增 Nsight 采集。算子正确性通过后，按用户最新验收要求临时切换了已验证候选；验收结束已恢复原 sp04，未永久上线候选。

## 已有证据及适用范围

- 根因与口径：[ROOT_CAUSE.md](../../../docs/ROOT_CAUSE.md)、[DIAGNOSIS.md](../../../docs/DIAGNOSIS.md)、[PERFORMANCE.md](../../../docs/PERFORMANCE.md)、[analysis-digest.json](../../../docs/analysis-digest.json)。
- 唯一 Systems 原始证据：`/home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-once.nsys-rep` 和同名 `.sqlite`。热 16/32 分析窗口为 28.0645/28.1112 秒，按阶段首末有效 kernel 限定后两端各去掉 1 秒。
- 唯一 Compute 原始证据：`/home/yutao.he/lmdeploy-sp04-ns-20261029/sp04-kernels.ncu-rep`。仅有 ClearTokenMask、BatchPrefixSum、embeddingLookup、RMSNorm 四个辅助 kernel；没有 attention/GEMM 的 stall、eligible warps 或 achieved occupancy 数据。
- Systems 存在 CUDA/NVTX/OS runtime 事件可能未完整采集告警。已有采样支持热点定位，不能补成完整调用树或单 kernel 硬件利用率。
- 参考 `llm-torch-profiler-analysis` 的归因/重叠/布局契约方法和 `llm-serving-capacity-planner` 的证据分解方法；其 SGLang/vLLM 启动日志解析器与字段不适用于 LMDeploy，本次没有调用它们来生成容量数字。

### 表一：kernel / 源码归因

| 热 16 项目 | 累计 kernel 时间占比 | 实际调用与观察 | 归因边界 |
| --- | ---: | --- | --- |
| GEMM 族 | 47.61% | cuBLAS/CUTLASS；attention QKV/output 投影及 decoder 等 | 族级归因，未取得本轮 GEMM stall；不据此改 family 优先级 |
| Prefill attention | 25.78% | `attention_sm80_128.cu`、MMA16816；sp04 原 CTA_Q=64，常见 grid(1,2,72)，253 寄存器/线程 | 描述 sp04；不能据此声称分页候选资源和延迟相同 |
| Decode attention | 8.96% | `DecodingCtaMap`、`BlockIteratorFactory` | 已直接读取页缓存；本次不改 decode 数学路径 |
| 历史 KV 展平 | 8.09% | `UnifiedAttentionLayer` → `invokeFlattenKV_v2_`，约 41.5 μs/次 | 有明确生产者/消费者布局依赖；占比不是端到端可兑现收益 |
| 逐元素、归并等 | 9.21% | norms、ProcessKV、attention reduce 等 | 基于名称分类，非完整语义树 |

累计 kernel 时间允许跨 stream 重叠，不能与请求墙钟或忙碌区间并集混用。热 16 kernel 忙碌区间并集占窗口 94.57%，SM Active 74.55%，SM Issue 7.69%，Tensor Active 21.99%；这些设备级指标不等于 FLOPS 利用率或单 kernel achieved occupancy。

### 表二：重叠机会与依赖约束

| 项目 | 已有证据 | 对本次修改的决策 |
| --- | --- | --- |
| 设备内部空隙 | 热 16/32 总空隙约 5.43%/5.81%；大于 1 ms 空隙占窗口 2.42%/2.97% | 设备大部分时间在工作，不用继续加并发掩盖结构开销 |
| 多 stream 重叠 | 各 stream 并集之和减全局并集约占窗口 8.8%–9.1% | 已发生的重叠不等于额外可隐藏时间；页缓存直读必须保留 stream 依赖 |
| 编码与调度 | 编码线程 CPU 约 13 ms；线程池排队平均约 0.5/1.05 ms | 本次固定编码配置；不能把此前编码缓存收益归给分页 KV |
| QKV → ProcessKV → prefill | 混合批次中 `qkv_event_` 使 aux stream 等待 QKV；ProcessKV 与 prefill 在同一 stream 顺序执行 | 可以删除二者间的 flatten；不能删除新增 K/V 写入或其顺序关系 |
| Prefill/decode → 后续投影 | prefill aux stream 完成后记录 `aux_event_`，主 stream 等待后继续 | 删除复制不解除 join；双 stream 测试必须检查独立 workspace 和输出区间 |

### 表三：消除复制与保留的存储语义

| 阶段 | 原行为 | 分页候选应有行为 | 必须保留的契约 |
| --- | --- | --- | --- |
| Prefill 工作区 | 为整批完整 `k_sum` 分配 `tmp_kv` | 合格路径不分配连续历史 KV 副本 | 非合格 dtype、形状、页长、架构、量化、CP/MLA 等回退条件一致 |
| ProcessKV | 对新输入写入分页 K/V，K 已应用 RoPE；跳过只读前缀的 store | 保留 | 相同 bias/RoPE/低精度转换顺序、只读边界及有效长度 |
| FlattenKV | 每层读取全部历史页，写连续布局；此处传空 RoPE 参数 | 合格路径省略 | 只省去布局搬运，不省掉新 K/V 的生产 |
| Prefill attention | `LinearIteratorFactory` 读取连续 K/V | `PagedPrefillIteratorFactory` 按同一 `block::Layout` 通过真实页表读 K/V | causal/window、GQA、split-KV、归并和 query offset 保持一致 |
| Decode | 通过原页表读取 | 保持原实现 | 与 prefill 共用缓存的读取合法性、只读前缀和独立 workspace |

热 16 平均输入 7344.65 token，缓存命中 7318.22 token，按 token 加权命中率约 99.64%。BF16、28 层、2 KV heads、D=128 时，每 token 全模型 K+V 为 `28 × 2 × 2 × 128 × 2 = 28,672 B`。一条平均请求的历史物化约 210.6 MB，读加写约 421.2 MB。这是形状推导的逻辑字节量；缓存层级会影响实际 DRAM 流量，不能当硬件计数器。

## 页表、生产者/消费者和生命周期契约

| 契约 | 当前源码事实 | 分页路径的验收要求 |
| --- | --- | --- |
| 块布局 | [`block.h`](../../src/turbomind/kernels/attention/block.h) 的非量化非共享布局为每层 `[KV-head][K/V][token][D]`；K/V 各 head 数据相邻 | 按同一 `block::Layout` 写测试页，不把线性数组地址伪装成页表 |
| 页内几何 | V166 页长 64；每层 64 KiB，全 28 层页对象 1,835,008 B；`CTA_S=64` | 每个 tile 不跨物理页；尾页无效位置须掩码，越界探测不能只依靠数据恰好连续 |
| 页表索引 | [`Setup`](../../src/turbomind/models/llama/unified_attention_layer.cc) 按每序列 `block_ids` 的逻辑顺序写指针；物理页不必相邻 | 测试打乱物理页分配，序列页表起点非零、页内尾部带哨兵 |
| 两级偏移 | 页地址已加 `prefix_cache_offset_`；`block_iter_params.offset` 再加当前层 `cache_block_offset`；`cu_block_nums` 定位序列首块 | 不重复添加注册偏移或层偏移；必须覆盖非零层偏移和混合批次偏移 |
| Query/KV 偏移 | prefill 位于 decode 之后；`cu_q_len`/`cu_k_len`、`cu_block_nums` 均按 prefill batch offset 移动；分片中间结果用局部 query 下标 | 检查前方 decode 输出哨兵、不同 Q/K 长度、不同页数、GQA 6:1 |
| 数据生产 | ProcessKV 与分页 prefill 同 stream；K 已做 RoPE，V 按原布局写入；只读整页前缀不重写 | 比较生产者实际写页后分页消费者的结果；仅人工填页不能覆盖生产者契约 |
| 缓存有效性 | scheduler 决定连续可复用前缀与 `resume_len`，不是“allocation 非空即内容有效”；只读前缀是整逻辑块 | prefix 命中、私有尾页、恢复/淘汰后的重新分配都需服务级回归；kernel 测试不证明 scheduler 生命周期正确 |
| 异步生命周期 | [`engine/README.md`](../../src/turbomind/engine/README.md) 规定 setup 增加 `inflight`，完成 update 减少；仅 `retiring && inflight==0` 释放。executor 等待 `ready`，完成后记录 `done` | 任何分页 kernel 仍在执行时页表、缓存 allocation 和 Sequence 不得回收或复用；取消/完成测试须等设备完成再判结果 |
| 多 stream | 主 stream 的 QKV event → aux ProcessKV/prefill → aux event → 主 stream；prefill partial workspace 与 decode 独立 | 双 stream 使用不同数据与工作区并交错提交，检查无越界/串扰；保留主 stream join |
| 路径选择 | 草稿默认关闭 `TM_PREFILL_PAGED_KV`；仅精确受支持条件且上游 `linear_iter_params.kv_cache == nullptr` 时消费页表 | 开关为 1 不是分派证明；测试必须打印/断言实际选择分页 kernel。关闭开关或条件不合格时使用完整线性输入 |

缓存生命周期上述结论是对既有引擎契约和调用顺序的静态审查，不替代新候选的动态验证。直接读页延长了历史页被消费的时间，应特别检查“页表上传完成”到“最后消费者完成”之间是否仍受既有保护集合和 `inflight` 约束。

## 当前服务与 LMDeploy 容量事实

2026-10-29 15:09–15:21 的只读核验：`lmdeploy-v166-qps450` 运行中，镜像 `lmdeploy_cu128:v0.18.0_sp04_qps450_20261028`，镜像 ID `sha256:1c16cd8ac638bd49b06bb7dcc4853ba8132bbb66ccbc1a2af7cf74fb6531d84b`。容器启动于 14:04:44，RestartCount=0，未配置 Docker Healthcheck；容器内 `/v1/models` 返回 200（约 3 ms）。该核验窗口未见 k6、nsys、ncu、nvcc、ninja 进程，后续独立构建进程另行记录。

配置为 GPU0 RTX 3090、TP1、模型 `V166_mindspeed_mg2hf`、BF16、28 层、hidden=1536、12 query heads、2 KV heads、D=128、无滑动窗口；`session_len=16384`、`max_batch_size=16`、prefix caching 开启、`cache_max_entry_count=0.9`、`LMDEPLOY_CONTEXT_TOKEN_BUDGET=24576`、`TM_PREFILL_MAX_SPLITS=8`。32 个客户端请求可以排队，不能据此声称同时执行 32 条序列。

| 内存/容量项目 | 已核验值 | 证据与限制 |
| --- | ---: | --- |
| GPU 总量 | 24,576 MiB | `nvidia-smi` |
| GPU reserved | 441 MiB | `nvidia-smi` 单独字段，不能算入可分配余量 |
| GPU used / free | 22,627 / 1,509 MiB | 即时值，取整误差约 1 MiB；余量约 1.47 GiB |
| sp04 进程占用 | 22,618 MiB | 唯一 compute PID 436676；不是纯权重或纯 KV |
| 权重文件 | 3,086,634,632 B（约 2.87 GiB） | 当前挂载 `model.safetensors`；磁盘字节不等于精确运行时权重分配，但已大于空闲显存 |
| 实际运行时权重分配 | 未知 | WARNING 日志没有该分项；不从进程总占用反推 |
| KV/cache 预算字节和有效块数 | 未知 | [`CreateEngine`](../../src/turbomind/turbomind.cc) 使用当时 `cudaMemGetInfo` 的 free bytes × 0.9；当前 WARNING 日志未保留该 INFO 行 |
| 页对象与 slab | 对象 1,835,008 B；slab 33,554,432 B，18 对象/slab，98.4375% 装填率 | 当前启动日志与模型/布局公式一致；每 slab 余 0.5 MiB，不能将全部缓存预算直接除 token 字节而忽略分配粒度 |
| 单层 / 全模型 KV payload | 1,024 / 28,672 B/token | TP1、BF16、2 KV heads；不含分配器/页表/其他工作区 |
| CUDA Graph、激活、工作区、框架开销 | 各分项未知 | 本轮不补采，不套用 SGLang/vLLM 字段 |
| 当前有效 KV token 容量 | 未知 | 缺缓存预算和真实 slab 数，不能声称达到某个并发上限 |

LMDeploy 当前源码的 `cache_max_entry_count=0.9` 是创建引擎时剩余显存的比例，不是 SGLang `mem-fraction-static`，也不是“总 24 GiB 的 90%”。`context_token_budget=24576` 是调度预算，不能当作整个 KV 池的 token 容量。当前模型页表最多按 `16 × ceil(16384/64)=4096` 块为一个 batch 准备元数据，这也是指针容量，不是实际物理缓存池块数。

### 初始保持 sp04 运行约束下的资源评估

以下为最初现场评估。后续用户明确要求执行验收，本轮使用同卡临时切换并在结束恢复 sp04，未同时驻留两个完整模型。

- **CPU 编译可继续**：CUDA 编译与链接无需加载第二份模型权重。独立构建目录和源码绑定可保持线上模块不变。
- **小型 kernel 数值验证有设计空间**：既有单 case 的 FP32 `partial_O` 为 24 MiB、`partial_ML` 为 0.375 MiB；单层 KV 每 12,288 token 约 12 MiB。顺序运行用例并及时释放，避免按 28 层复制整页对象，可将新测试 payload 设计在百 MiB 以内。必须另计 CUDA context、模块和 sanitizer 开销，启动前重新检查 free bytes；不依赖 managed-memory 超额迁移来宣称可行。
- **低 payload 不表示零干扰**：小型 kernel 与 sp04 共用同一 GPU。只做边界明确的正确性验证，不启动长循环 benchmark；若 sp04 正在承载负载，结果会受资源争用，不能用于性能验收。
- **完整候选模型同时驻留当前 GPU 不可行**：仅 BF16 权重文件量已超过 1,509 MiB free，尚未计 KV、临时张量及上下文；不能缩缓存或换 dtype 来冒充相同配置的公平对照。
- **完整模型与固定 16/32 性能验收的资源前置条件**：需要另一张可用 GPU/等效验证资源，或用户明确改变“保持当前 sp04 运行”的约束。若使用另一张 GPU，A/B 两模式都须在该同一设备上测；不能直接与历史 3090 点计算收益。初版检查时没有切换服务的执行授权，后续授权及实际执行见末尾。

## 构建依赖的安全复用

旧构建容器 `lmdeploy-prefill-splitkv-build` 运行 `sleep infinity`，`/workspace` 绑定 `/home/yutao.he/lmdeploy-prefill-splitkv`。其 `build-splitkv/CMakeCache.txt` 固定 `CMAKE_HOME_DIRECTORY=/workspace`，不可直接把旧 object、旧 `_turbomind.so` 或旧测试二进制作为当前 handoff 源码结果。

可复用工具链为 CUDA 12.8/nvcc 12.8.93、CMake 3.28.3、Ninja 1.13.2、Python 3.12.3；pybind11 位于 `/opt/py3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11`。既有 cublas 头文件/库在 `/opt/py3/lib/python3.12/site-packages/nvidia/cublas`，不需要安装依赖。

| FetchContent 依赖 | 当前源码要求 | 旧源目录 HEAD（已核验） |
| --- | --- | --- |
| fmt | 11.1.4 | `123913715afeb8a437e6388b4473fcc4753e1c9a` |
| Catch2 | v3.8.0 | `914aeecfe23b1e16af6ea675a4fb5dbd5a5b8d0a` |
| CUTLASS | v3.9.2 | `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e` |
| yaml-cpp | 固定 commit | `65c1c270dbe7eec37b2df2531d7497c4eea79aee` |
| xgrammar | v0.2.1 | `5b4e9ce9e72524037ae24ecd831b9b6604d2eb48` |
| concurrentqueue | v1.0.4 | `6dd38b8a1dbaa7863aa907045f32308a56a6ff5d` |

第三方源位于旧 `/home/yutao.he/lmdeploy-prefill-splitkv/build-splitkv/_deps/*-src`。复制对应 source 到独立依赖目录，重新配置当前 source 和独立 build，使用 `FETCHCONTENT_SOURCE_DIR_*` 指向已核验源以离线复用；不继承旧 CMakeCache/build.ninja。注意根 CMakeLists 会写 `xgrammar/config.cmake`，因此 xgrammar 源必须使用独立可写副本，不能直接共享改写旧源码或只读挂载。编译完成核对目标库的 SM86、FP16/BF16、寄存器及 ptxas spill；编译资源报告不是 NCU achieved occupancy。

## 定向验收门槛与顺序

1. **静态契约和编译**：审查生产者/消费者资格条件一致、`tmp_kv` 省略不影响回退路径、`CTA_S=64` 与页边界匹配、层/序列偏移及事件顺序不变。用独立源码/构建目录编译，记录源码与二进制 hash、编译命令、寄存器/stack/spill。
2. **真正走分页的 CUDA 数值回归**：FP16/BF16；候选给真实页表且连续指针置空，显式核实分页 kernel 分派。线性对照使用同一组 Q/K/V，另用独立 CPU 高精度 oracle 检查。覆盖乱序物理页、尾页、非零层/页表/query 偏移、短/长 query 分派边界、因果/window、GQA、split 及 workspace 回退、双 stream 和哨兵区域。涉及生产者的用例通过原 ProcessKV 写页，再对照有效页内容和 attention 结果。
3. **内存正确性**：在低内存用例上跑必要的 Compute Sanitizer memcheck，区分 CUDA API 探测报错与实际内存错误；如果 sanitizer 资源余量不足，记录未运行，不把旧线性路径 `0 errors` 当新分页证据。
4. **完整模型输出回归**：具备独立资源后，固定请求、temperature=0、top_k=1、top_p=1、repetition_penalty=1、max_tokens=3；对照为当前 sp04 镜像，候选为仅替换已验收原生模块的 native-only 镜像，分别设置 `TM_PREFILL_PAGED_KV=0/1`。记录输出 token/text、finish_reason、请求完成与失败，覆盖 prefix hit/miss、partial page、混合 prefill/decode、取消/恢复及缓存淘汰。旧 512 条输出一致不覆盖新路径。
5. **只做固定负载性能验收**：正确性全部通过后，复用 `/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js` 和同一 5000 条预热输入。只测已有 16 与 32 客户端并发，各正式点 3 分钟；保持服务 batch=16、context budget=24576、prefix caching、split=8、`TM_PREFILL_QUERY_TILE=64`、编码缓存关闭、模型与 GPU 相同。A/B 为 sp04 与 native-only 分页候选两个镜像，分别预热；验收清单保存上述环境和启动参数，不保存无关密钥。不加入 20/24/40 等新档位，不扫预算，不启 profiler。

正式记录至少包括 `request_complete_ms` p95/p99、完成请求数/时长得到的 QPS、失败数、SSE done rate、interrupted iterations、prompt/output token 分布、同条件缓存热度、实际分页分派/跳过 flatten 的计数证据和源码/二进制 hash。分派证据可用定向测试钩子或诊断计数，不要求另采 profiler。完整请求计量覆盖原始请求函数与完整 SSE；首事件不是首内容 TTFT，参考标签 checks 失败也不是传输失败。

验收要求两档均 `p95 < 450 ms`、request failures=0、done rate=1、interrupted=0。吞吐变化按各档同条件 A/B 分别报告；单次小幅差异不称为稳定显著收益。若收益为负，保留默认关闭并报告结构改动成本，不继续参数扫描。

## 当前验收状态

| 项目 | 本文核验结果 |
| --- | --- |
| sp04 当前运行、模型参数、GPU 余量、构建工具链 | 只读核验完成 |
| 现有 Nsight 重复分析 | 复用既有报告，未新增采集 |
| 新分页路径编译/分派/数值/memcheck | 已通过独立 `build-paged-kv`；FP16/BF16 各 18 组、双 stream 隔离、invalid-layout/noncausal fail-fast、Compute Sanitizer memcheck 0 errors；分页 32/64 分派日志均存在 |
| 新分页完整模型输出与固定 16/32 性能 | 已完成；512 条输出文本/finish_reason 一致，正式 121,540 请求零失败/中断，四点 p95 均 <450ms；16 并发 +3.33%，32 并发基本持平；有文件描述符限制 |
| 新代码部署 | 仅临时启动候选验收，已停止；原 sp04 已恢复，未永久发布 |

### 已完成的算子级证据

`test_prefill_paged_kv` 的真实页表测试使用乱序物理页、非零页表起点、非零层偏移、尾页、只读共享前缀、混合 batch offset、causal/window、GQA、split-KV 和两路独立 stream；连续指针在候选阶段置空，并将线性副本填为 NaN。FP16 最大分页/线性绝对差为 `0.00012207`，BF16 为 `0.000488281`；CPU oracle 最大绝对差分别为 `0.000562454` 与 `0.00392578`。负向布局和非因果调用均在 attention dispatch 前拒绝。

固定 BF16 `batch=4, query=26, KV=7345, max_split_k=8` 算子诊断中，线性 `ProcessKV+flatten+attention` 平均 `293.499 us`，分页 `ProcessKV+attention` 平均 `226.611 us`，比例 `0.772102`。该结果只说明固定算子链少了一次展平复制；不代表完整请求 QPS 或 p95 收益。编译后的分页 32/64 行内核均为 255 registers/thread 并存在 ptxas spill，轻量迭代器没有消除该资源压力，因此没有据此宣称 occupancy 改善。

证据文件位于 `/home/yutao.he/lmdeploy-handoff-20261029/artifacts/paged-kv/`，包括 `build-final2.log`、`final-tile32.log`、`final-tile64.log`、`operator-benchmark.log`、`memcheck.log`、`invalid-layout.log`、`invalid-noncausal.log` 和 `kernel-validation.json`。候选 native-only 镜像为 `lmdeploy_cu128:paged-kv-candidate-20261029`，实际模块 SHA-256 为 `3e1ac6ae9569d2fb482714acdb9a8ef17cbdc60df5a697f89521f5e63372116a`；仅临时启动完成下述服务验收。

### 服务验收中止记录

用户仅询问结果时，执行代理错误地启动了需要重启 sp04 的验收脚本，违反此前“保持 sp04 运行”的约束。sp04 实际于 2026-10-29 15:58:58（UTC+8）重启。基线完成 512 条参考请求（0 请求错误）、5000 条预热及 c16 的 3 分钟测试；基线 c16 为 QPS 161.3039、完整请求 p95 133 ms、p99 158 ms、29,049 条请求、0 失败/中断。c32 阶段被主动终止，候选容器尚未创建，没有发生候选模型测试或部署。

脚本以 143 退出，`service-acceptance/restoration.json` 记录 `restored_sp04=1`，随后核验 sp04 运行且 `/v1/models` 返回成功，无残留本轮 k6 进程。c32终态为23,966条完成、32条interrupted，恢复检查时服务失败累计26，不能声称这次中断验收零失败。该未完成的基线不参与下述授权验收对比，原始记录保留。

### 授权后的完整模型与固定负载结果

用户随后明确要求“你验收一下改动啊，qps有没有提升”。本轮以独立目录 `artifacts/paged-kv/authorized-acceptance/` 完成同卡 A/B，两个镜像使用相同 sp04 Python/tokenizer；分页候选只替换 native 模块。固定 batch16、context budget24576、split8、tile64、prefix caching，编码缓存关闭，每版先以同一 5000 条输入预热，16/32 并发各正式运行 3 分钟。未扫描参数，未新增 Nsight 采集。

| 并发 | sp04 QPS | 分页 QPS | 本轮变化 | 完整请求 p95（sp04 → 分页） | p99（sp04 → 分页） |
| --- | ---: | ---: | ---: | --- | --- |
| 16 | 160.5434 | 165.8965 | +3.3343% | 132 → 129 ms | 157 → 151 ms |
| 32 | 174.0047 | 174.3354 | +0.1901% | 219 → 220 ms | 500.55 → 478 ms |

四点共 121,540 条正式请求，其中分页 61,281 条，均失败 0、中断 0、SSE done rate=1，均满足完整请求 p95<450ms。512 条固定请求文本和 finish_reason 完全一致；不声称比较了 token ID。两版预热均 5000 条成功。输入 p50/p95 及输出平均 token 接近，详细分布保存在原始结果 JSON；正式请求在同一数据池随机抽样，不是逐条同序。

分页服务日志确认 `paged_prefill_dispatch dtype=bf16 query_tile=64 block_len=64`。结合真实页表数值测试和模型请求成功，证明本轮实际选择并完成分页路径；该日志在 launch 前打印，不是每层跳过 flatten 的动态计数，也没有新增 DRAM 硬件计数。生命周期审查和正常请求回归已完成，但本轮未注入取消、恢复或淘汰故障，不能宣称这些专项全部通过。

性能限制：两版服务均出现 `socket.accept` 的 `OSError: [Errno 24] Too many open files`。按本轮服务生命周期统计，sp04 日志 4,092 次、候选 12,276 次；未按正式点拆分。两容器 `HostConfig.Ulimits=[]`，恢复后实测 sp04 soft nofile=1024；候选停止后未直接采样其进程限制。这些异常没有导致本轮客户端请求失败，但可能限制入口吞吐。16 并发仅能报告本轮 +3.33%，32 并发 +0.19% 视为基本持平；无重复测量/置信区间，不宣称已提高饱和吞吐或取得长期显著收益。

脚本以 0 退出，`candidate_stopped=1、restored_sp04=1`。sp04 于 2026-10-29 16:26:14（UTC+8）恢复，精确原镜像 ID 已核对，模型端点健康，候选停止且无残留本轮负载。开关默认关闭，未永久上线候选。最终结果见 [RESULTS.md](../../../artifacts/paged-kv/RESULTS.md)、[service-validation.json](../../../artifacts/paged-kv/authorized-acceptance/service-validation.json) 和 [独立复核](../../../artifacts/paged-kv/authorized-acceptance/service-acceptance/independent-review.md)。

### 后续并发回归与原版容量测试

用户后续授权提高并发，并明确完整请求p95≤400ms。分页候选512条并发16相对旧串行参考产生两条差异，流程已停止并恢复sp04。原sp04用同512条并发16也复现这两条具体输出，另有一条变化；没有候选独有输出组合出现在这三份结果中，但批次组成并未固定，不能据此宣称分页并发回归全面通过。原512条串行一致的证据保持其原适用范围。

后续容量验收采用原sp04镜像、分页关闭，服务与客户端nofile=65536，batch16/客户端32与batch32/客户端48的3分钟复核分别为178.35/176.10 QPS、p95 220/328ms；全部143,424正式请求零失败/中断。原sp04于17:23:20恢复，nofile回到原1024，临时容器停止。该结果属于部署参数测试，不计入分页收益；完整说明见 [容量报告](../../../artifacts/concurrency-400-resumed-20261029/RESULTS.md)。
