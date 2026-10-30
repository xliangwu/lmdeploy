# V166 / RTX 3090 完整优化源码入口

发布分支：[`perf/v166-3090-optimized`](https://github.com/xliangwu/lmdeploy/tree/perf/v166-3090-optimized)。发布说明更新于2026-10-30；本文引用的 GPU、模型和容量实验记录来自2026-10-29及此前。

本分支包含完整 LMDeploy 仓库、历史已验收优化，以及当前输入编码缓存、短 query 和 prefill 分页 KV 候选的源码、测试、Dockerfile 与辅助脚本。先前 `7b234c4` 是仅文档提交，本次将完整源码整理到同一分支。**源码交付不改变运行服务：原 sp04 保持运行，分页候选没有永久上线。** 候选在源码中默认关闭；历史专用候选 Dockerfile 会显式开启编码缓存或32行 tile，不能把这些镜像等同于当前服务。

优化目标是完整请求 p95<450ms、零请求失败并提高 QPS；后续原 sp04 容量实验使用更严格的 p95≤400ms。所有结果保留各自的镜像、配置和门槛，不能用线性路径测试替代分页正确性证据，也不能将原 sp04 容量成绩归因于分页。

## 获取与阅读

克隆仓库并切换到完整优化分支：

```bash
git clone https://github.com/xliangwu/lmdeploy.git lmdeploy-v166-3090
cd lmdeploy-v166-3090
git checkout perf/v166-3090-optimized
git rev-parse HEAD
```

| 阅读入口 | 内容 |
| --- | --- |
| [修改方法与版本记录](optimization_changes_3090.md) | 每项优化的原因、实现文件、开关、证据及限制 |
| [分页 KV 验证](paged_prefill_kv_validation.md) | 页表、生命周期、布局契约、真实页表算子测试及固定16/32服务 A/B |
| [并发结果与命令](concurrency_results_3090.md) / [CSV](concurrency_results_3090.csv) | 原 sp04 的七个正式点、nofile条件、完整请求SLA和恢复状态 |
| [Nsight 分析](nsight_analysis_3090.md) | 复用已有 Systems/Compute 报告，解释统计窗口、FlattenKV热点和采集边界 |
| [历史 GPU 实施记录](prefill_split_kv_3090.md) / [历史450ms验收](qps450_3090.md) | sp01→r3→sp04 的历史来源与部署记录 |

## 源码范围与默认开关

历史优化包括 prefill 专用 split-KV 缓冲、混合批次下标修复、分片上限、首次反分词尾部上下文、上下文预算接入，以及 GEMM 静态注册链接修复。完整源码同时补齐原sp04通过镜像继承的mentor异步前处理：仅移植提交 `ccc395d` 中模板渲染和encode的两处 `await asyncio.to_thread(...)`，保留main其余逻辑，归属仍为mentor已有优化。当前候选在这些历史实现之上继续开发，不能将“仓库有代码”视为“线上已启用”。

| 功能 | 主要源码入口 | 源码默认与当前服务 |
| --- | --- | --- |
| mentor异步文本前处理 | [multimodal.py](../../lmdeploy/serve/processors/multimodal.py) | 模板渲染和encode在线程池执行，无独立环境开关；原sp04已通过镜像继承 |
| prefill split-KV | [unified_attention_layer.cc](../../src/turbomind/models/llama/unified_attention_layer.cc)、[attention_universal.h](../../src/turbomind/kernels/attention/attention_universal.h) | 原 sp04 显式 `TM_PREFILL_MAX_SPLITS=8`；设1可关闭分片 |
| 首次反分词尾部上下文 | [tokenizer.py](../../lmdeploy/tokenizer.py) | `LMDEPLOY_QWEN2_BYTELEVEL_TAIL` 默认关闭；原 sp04 显式设1 |
| 上下文预算 | [turbomind.py](../../lmdeploy/turbomind/turbomind.py) | 未设置时保持引擎默认；原 sp04 显式24576 |
| 输入编码片段缓存候选 | [tokenizer_chunk_cache.py](../../lmdeploy/tokenizer_chunk_cache.py) | `LMDEPLOY_QWEN2_ENCODE_CACHE=0`；[专用Dockerfile](../../docker/Dockerfile.encode-cache)显式设1 |
| 短 query 候选 | [attention_sm80_128.cu](../../src/turbomind/kernels/attention/kernel/attention_sm80_128.cu) | `TM_PREFILL_QUERY_TILE=64`；[专用Dockerfile](../../docker/Dockerfile.short-prefill)显式设32，并继承编码缓存镜像 |
| prefill 分页 KV 候选 | [paged_prefill.h](../../src/turbomind/kernels/attention/paged_prefill.h)、[paged_prefill_iterator.h](../../src/turbomind/kernels/attention/paged_prefill_iterator.h) | `TM_PREFILL_PAGED_KV=0`；[分页Dockerfile](../../docker/Dockerfile.paged-prefill)也默认关闭，由验收显式开启 |

分页路径仅覆盖 SM86、FP16/BF16、head128、Q12/KV2、64-token页、CP1、未量化、非MLA、因果 attention；正常不支持场景保持线性路径。候选仍保留 ProcessKV、新增K的RoPE、原掩码、GQA、分片归并及 stream 依赖，只在生产者/消费者资格一致时省略每层 `tmp_kv` 和历史KV展平复制。

## 已验证内容与限制

| 对象 | 已有证据 | 适用边界 |
| --- | --- | --- |
| 本次源码异步前处理补齐 | [test_text_preprocess_async.py](../../tests/test_lmdeploy/test_text_preprocess_async.py) 的5项CPU回归通过：worker阻塞时loop可响应、空content/参数传递、reasoning回退、模板None拒绝、异常传播 | 实际导入当前源码；使用受控模板/tokenizer，不加载模型；完整源码组合未新增服务A/B或GPU性能验收 |
| 真分页算子 | [test_prefill_paged_kv.cu](../../src/turbomind/kernels/attention/test_prefill_paged_kv.cu)：乱序真实页、连续指针为空、线性副本NaN、独立FP64参考；FP16/BF16、32/64行、偏移、tail、共享只读前缀、RoPE、mask、split、双stream与memcheck通过 | 这些证据来自分页测试本身；旧线性测试通过不计作分页通过 |
| 固定算子链 | BF16 batch4/Q26/KV7345/split8/tile64：293.499→226.611μs，下降22.79% | 不是完整请求QPS收益；分页内核仍有255寄存器/线程及spill |
| 分页完整模型 | 512条串行请求文本及finish_reason一致；固定16/32 A/B共121,540条正式请求零失败/中断，均p95<450ms | 未比较token ID；16并发本轮QPS +3.33%，32并发 +0.19%基本持平；两版均出现EMFILE，未证明稳定饱和吞吐提升 |
| 分页后续并发对照 | 并发16相对串行参考的两条差异也在原sp04复现 | 未固定实际batch，不能宣称全部并发输出等价；未完成取消/恢复/淘汰专项故障注入 |
| 原sp04容量 | nofile65536下共143,424条正式请求零失败/中断；batch16/并发32确认178.35 QPS、p95 220ms；batch32/并发48确认176.10 QPS、p95 328ms | 分页关闭；不是分页收益、全局最优或长期容量上限；nofile65536与batch32未永久应用 |

最后一次服务恢复记录为2026-10-29 17:23:20 +08:00：`lmdeploy-v166-qps450`，原sp04镜像，batch16、context24576、split8，实际nofile1024/524288。已测试 native 模块SHA-256分别为原sp04的 `92b2aab612a4c50341bc05bb1635c46802d98091940606eda1c751591e5f2347` 和分页候选的 `3e1ac6ae9569d2fb482714acdb9a8ef17cbdc60df5a697f89521f5e63372116a`。这些是实验产物身份；新机器或新源码构建不能自动沿用其验证结论。

## 构建入口

[build_paged_prefill.sh](../../tools/build_paged_prefill.sh) 会在独立 `build-paged-kv` 中编译分页测试、线性回归和原生模块，并准备 `build-paged-kv/image/`。编译和链接在 CPU 上执行，不运行 GPU 测试、不加载模型，也不启动或停止服务；仍需要 CUDA 编译工具链及相应头文件/库。默认SM86、单GPU构建。

历史工具链为 CUDA12.8/nvcc12.8.93、CMake3.28.3、Ninja1.13.2、Python3.12.3及已安装的pybind11、cuBLAS等依赖。脚本使用当前 `python` 的 `site-packages/nvidia/` 目录，需在已准备好的同类构建环境或容器中执行。它不是自动创建工具链的安装器。

在匹配工具链的环境中，从当前克隆目录执行独立构建：

```bash
export JOBS=6
bash tools/build_paged_prefill.sh
```

如需离线复用已核验的第三方源码，可设置 `DEPENDENCY_SEED` 为包含 `fmt-src`、`catch2-src`、`repo-cutlass-src`、`yaml-cpp-src`、`xgrammar-src`、`concurrentqueue-src` 的目录；脚本会复制到新构建目录。不要复用旧项目的objects、CMakeCache或旧测试二进制。第三方版本与xgrammar可写副本要求见[分页验证说明](paged_prefill_kv_validation.md)。

在已具备本项目Python依赖的环境中，从仓库根目录运行异步前处理CPU回归；该检查不加载模型或使用GPU：

```bash
export CUDA_VISIBLE_DEVICES=
python -m unittest discover -s tests/test_lmdeploy -p test_text_preprocess_async.py
unset CUDA_VISIBLE_DEVICES
```

在有可用SM86 GPU的验证环境中运行真实页表数值测试。以下会使用GPU，不能将其与CPU编译混为一谈；需要避免与服务负载争用：

```bash
export TM_PREFILL_PAGED_KV=1
export TM_PREFILL_LOG_DISPATCH=1
export TM_PREFILL_QUERY_TILE=64
./build-paged-kv/bin/test_prefill_paged_kv
export TM_PREFILL_QUERY_TILE=32
./build-paged-kv/bin/test_prefill_paged_kv
unset TM_PREFILL_PAGED_KV TM_PREFILL_LOG_DISPATCH TM_PREFILL_QUERY_TILE
```

`--invalid-layout`、`--invalid-noncausal` 为预期报错的负向入口；`--benchmark` 只运行固定形状算子诊断，不执行完整正确性套件。memcheck、输出回归和服务A/B仍按[定向验收门槛](paged_prefill_kv_validation.md)完成，不能只运行benchmark就标记通过。

分页镜像构建入口为 [Dockerfile.paged-prefill](../../docker/Dockerfile.paged-prefill)。它依赖本地已存在且ID匹配的sp04基础镜像，仅替换已验证的原生模块，并核对模块SHA-256；`build-paged-kv/image` 是对应build context。镜像文件和已编译 `.so` 不随Git上传，不能只凭相同tag认为是历史已测产物。

## 服务验收入口与外部依赖

[accept_paged_prefill.sh](../../tools/accept_paged_prefill.sh) 是原服务器固定16/32、450ms口径的A/B流程；不传参数只显示计划，不执行服务操作：

```bash
bash tools/accept_paged_prefill.sh
```

传入 `--execute` 会重启/临时停止sp04并在流程结束恢复，需先满足该次服务切换授权、真实分页正确性、同一模块哈希、基础/候选镜像、全新报告目录及固定负载条件。脚本要求外部 `kernel-validation.json`，它必须对应本次真实验证产物，不能复制旧的“passed”状态来放行新二进制。现存候选容器或报告目录会触发拒绝覆盖。仓库发布本身不执行该流程。

[benchmark_concurrency_point.sh](../../tools/benchmark_concurrency_point.sh)、[search_concurrency_400.sh](../../tools/search_concurrency_400.sh)、[confirm_sp04_concurrency.sh](../../tools/confirm_sp04_concurrency.sh) 随仓库保留，记录历史400ms容量验收的实现；完整命令见[并发报告](concurrency_results_3090.md)。保留脚本不表示本次恢复参数扫描或启动新负载。编码缓存检查见[verify_encode_cache.py](../../tools/verify_encode_cache.py)，其中模型默认固定为 `/models/V166_mindspeed_mg2hf`。

完整源码不等于新机器clone后即可直接重跑历史压测。上述验收脚本固定了原服务器的容器名、镜像ID、Python3.12 native路径、GPU0、23333端口、宿主模型挂载和工作负载位置；换环境需先适配并重新记录环境身份。外部依赖包括：

- V166模型及tokenizer：宿主 `/data/resources/resource/olive.core.llm`，容器 `/models/V166_mindspeed_mg2hf`。
- k6客户端：`/data/users/xueliang.wu/me-k6-tools/k6-sse`。
- 测量脚本：`/home/yutao.he/lmdeploy-sp04-ns-20261029/test.measured.js`，以及它导入的外部 `core/uuid.js`、`k6-utils/index.js`。历史脚本SHA-256为 `4621d1deeba247deda83d1bea996bf19cbf53f88ab9b73f8c67b147ad94bce3d`。
- 5000条私有输入：`/data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt`；SHA-256为 `abdf163be67a84181b30b8b32291d1ec2726b0d78477f6dca669419bc53fd663`。
- Docker基础/候选镜像、已编译模块和本次验证清单，均需要在运行环境单独准备和核验。

这些模型、私有输入、客户端二进制、外部测量代码、原始业务日志及大型 `.nsys-rep`、SQLite、`.ncu-rep` 报告不随Git发布。本文及配套报告中的 `../../../artifacts/`、`../../../docs/`、`/home/`、`/data/` 链接是原服务器证据定位，不是GitHub可下载资源。已有Nsight分析继续复用原报告，未新增采集。
