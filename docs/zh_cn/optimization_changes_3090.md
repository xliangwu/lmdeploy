# V166 / RTX 3090 优化改动简记

更新：2026-10-29。记录每项改动的原因、方法和文件；后续新增优化继续补在此文档中。

目标：**完整 SSE 响应 p95 <450ms、请求无失败，尽量提高 QPS**。适用验收环境为 V166、RTX 3090、单 GPU；已验证成绩来自充分预热、长输入、最多生成 3 token 的负载。

## 先看代码来源

```text
v0.18.0 发布提交 110965c7
  ├─ main 后续 26 个提交 → 7998cdb → 我们的优化提交 65e8a7f
  └─ mentor 异步前处理提交 ccc395d → v0.18.0_optimize

实际验收镜像：mentor sp01 → GPU 分片 r3 → CPU 优化与预算配置 sp04
```

mentor 已把模板渲染和输入 encode 改为 `await asyncio.to_thread(...)`，文件为 `lmdeploy/serve/processors/multimodal.py`。**这是 mentor 的已有优化，不计作我们新增的成果。**

2026-10-29 已核对：mentor 与 sp04 容器中的该文件都与 `v0.18.0_optimize` 一致。源码分支基于 `main`，没有合入这次异步提交，靠镜像继承保留它；因此“直接全量安装当前源码”不能默认等价于已测镜像。两条基础分支的 `src/turbomind/`、`tokenizer.py` 和 `turbomind.py` 原始内容一致。

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

## 结果口径与维护约定

| 同并发 A/B | 原 mentor 镜像 | sp04 组合优化 |
| --- | --- | --- |
| 16 VU，3 分钟 | 79.43 QPS，p95 264ms | 170.03 QPS，p95 122ms |
| 32 VU，3 分钟 | 80.86 QPS，p95 486ms，p99 632ms | 167.07 QPS，p95 389.05ms，p99 793ms |

两版使用同一 `slo_load.js`、同一 5000 条数据，各自先完成 5000 次预热；正式 A/B 均零请求失败。`slo_load.js` 在测试目录内，属于测量工具：固定排列、完整 SSE 计时和成功检查，与原 `test.js` 的随机抽样等逻辑不同。170 QPS 不能直接当作原 `test.js` 历史截图的复测结果。

仓库中的[完整验收记录](qps450_3090.md)和[GPU 实施记录](prefill_split_kv_3090.md)保留历史证据。原始结果在本地 `D:/aispeech/reports/lmdeploy-qps450-20261028`，开发机 `/home/yutao.he/lmdeploy-qps450-20261028`；源码在开发机 `/home/yutao.he/lmdeploy-prefill-splitkv`。

本次代码已于 2026-10-29 提交为 `65e8a7f`，本地与开发机同一提交，尚未推送 GitHub；历史报告内“尚未提交”等文字属于当时快照。本文不宣称服务持续运行，现场容器状态需要另查。

后续每项改动沿用“**原因 → 方法/示意 → 文件 → 验证与边界**”记录；修改开关、适用范围或镜像组成时同步更新。尚未完成的字节边界检查、更长解码覆盖及逐项消融，不能记成已完成成果。
