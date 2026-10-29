# V166 / RTX 3090 prefill split-KV 实施与部署记录

> 本文保留r3历史构建与诊断。当前工作区、sp04运行快照及后续分页候选状态见[修改方法](optimization_changes_3090.md)；最新400ms并发结果见[并发验收](concurrency_results_3090.md)，已有Nsight指标和文件见[分析索引](nsight_analysis_3090.md)。历史构建容器运行状态以现场核验为准。

日期：2026-10-28，Asia/Hong_Kong。源码基于 xliangwu/lmdeploy 的 7998cdbaf713393b9155358a147226def2557467，工作分支 perf/prefill-split-kv。尚未提交或推送 GitHub。

当前服务已于 2026-10-29 切换至 sp04；同16/32并发最终对比、ByteLevel解码优化及运行配置见 [最终验收记录](qps450_3090.md)。下文保留 r3 的历史部署与诊断信息。

## r3 部署记录（已被 sp04 替代）

- 机器：10.12.7.41；服务端口：23333。
- 新容器：lmdeploy-v166-splitkv，ID 11cf85ef37f7735bc2de7421abc5e9d24324d75f6a78868ccb2dcc01cc499e09。
- 新镜像：lmdeploy_cu128:v0.18.0_sp02_splitkv_20261028_r3。
- 镜像 ID：sha256:121fdb0907b0f3ea7922913a7c0c74902c83157e4aeb59a53980a1fb8a8c5b0b。
- 生效设置：TM_PREFILL_MAX_SPLITS=8。模型、端口、TP1、session-len=16384、max-batch-size=16、cache-max-entry-count=0.9 与 prefix caching 设置均保持原样。
- 新镜像继承 mentor 镜像的 Python 代码与运行依赖，替换了从指定 GitHub 源码编译的 TurboMind 原生模块。异步模板渲染与 tokenizer encode 文件 SHA-256 未变。
- 仅构建 SM86、单 GPU；此镜像不是多卡/其他架构的通用发行版。BUILD_MULTI_GPU=OFF。
- 旧容器 c258526a2236 已停止，旧镜像 lmdeploy_cu128:v0.18.0_sp01 原样保留。
- 最终检查：原生模块 SHA-256 为 92b2aab612a4c50341bc05bb1635c46802d98091940606eda1c751591e5f2347，无 profiler 注入，HTTP 服务正常，容器重启次数 0。

## 实现内容

1. 为常规 attention 的 prefill 单独分配 partial_O、partial_ML、split_cnt，避免和 decode 在不同 CUDA stream 上写入同一片临时存储。
2. prefill query 总数不超过 2048 且 workspace 能容纳分片结果时启用分片；CP、MLA（Multi-head Latent Attention）和旧架构路径保持原行为。
3. split_cnt 写入统一减去 offset_q，匹配混合 prefill/decode 批次中的局部 query 下标；输出和归并使用相同偏移规则。
4. 新增 TM_PREFILL_MAX_SPLITS（1..128，构造时校验）用于控制实验和快速回退。1 为未分片；最终部署明确设置 8，不使用无限制启发式的 128 上限。
5. 修复 CMake 3.28 构建中 GEMM 静态注册对象丢失的问题，将 WHOLE_ARCHIVE 通过普通接口链接传递。否则原生库虽能 import，Gemm.data_types() 却为空，模型不能启动。
6. 增加独立 CUDA 回归、固定请求输出验收、可复现构建及带自动回退的部署脚本。正式部署直接启动 lmdeploy，不加载此前的 profiling 包装。

## 验收证据

| 检查 | 结果与范围 |
| --- | --- |
| FP16 数值、GQA、因果掩码、滑动窗口、offset、workspace 边界 | 96 组形状/分片检查 + 8 组双 stream 检查通过 |
| BF16 同上 | 96 + 8 组通过 |
| 独立 CPU FP64 oracle | 首尾 query / 部分 heads 检查，发生大于未分片对照阈值的差异时另外检查失败元素 |
| Compute Sanitizer memcheck | 内存访问检查 0 errors；使用 --report-api-errors no，排除了注册过程对不支持的 shared-memory 配置的 API 探测报错，不将其描述为全部 CUDA API 零错误 |
| 新原生模块但关闭 prefill 分片 | 与 mentor 的 128 条固定输出全部一致 |
| 启用 8 路分片 | 128 条与扩展的 512 条固定样本均无 HTTP 错误、无内容或 finish_reason 差异 |
| Python 预处理继承检查 | multimodal.py、async_engine.py、messages.py 的哈希与 mentor 镜像相同 |

数值测试加入幅度 0.5、2、5，覆盖近似均匀和尖峰 softmax。BF16 改变归约顺序并不保证逐 bit 相同：一例新结果 0.031250、旧结果 0.034668，而 FP64 为 0.031224。对照阈值和高精度阈值区别记录；高幅度 oracle 的绝对项按 dtype 舍入精度与 V 幅度缩放（BF16 max(0.002, max|V|/256)，FP16 max(0.002, max|V|/2048)），另加 2% 输出相对项。业务验收仍使用输出字符串完全相等，不用这个数值容差放行标签变化。

初始 128 路上限在业务样本 3838 上把 Z2 变成 Z1，该候选被回退；关闭分片的新模块与旧版一致，限制为 8 后两套样本全部一致。不能把 8 路结论外推为任意分片数都不改变输出，也不能据固定子集声称所有未来输入逐 bit 不变。

## 新版性能结果

只运行新版一次 16 VU、3 分钟压测；旧版使用 2026-10-27 已保存结果，没有重跑旧性能基线。沿用原 k6-sse test.js、5000 行输入、max_tokens=3。原脚本按随机顺序抽样，这不是逐请求配对 A/B，也不是统计显著性结论。

| 指标 | 旧版保存结果 | 新版（8 路） |
| --- | ---: | ---: |
| HTTP 请求数 | 13,566 | 14,156 |
| QPS | 75.2883 | 78.5752 |
| HTTP 平均耗时 | 205.38 ms | 196.53 ms |
| HTTP p95 | 269.98 ms | 269.46 ms |
| HTTP p99 | 311.91 ms | 329.68 ms |
| HTTP 成功检查 | 100% | 100% |
| 原脚本标签一致检查 | 88.28% | 87.69% |

本次观察 QPS +4.37%，p95 基本不变，p99 增加约 17.77ms。标签检查样本不是同一组，因此不能据此独立判定质量回退或提升；另行完成的固定样本对照全部一致。原标签检查接受首字符相等，不是严格准确率。

HTTP p95 低于 450ms，但本轮沿用旧脚本，没有额外自定义“发请求至 DONE”的端到端指标，不把 SSE 首事件计时当成严格首内容 TTFT。单算子诊断的数倍加速没有转换成数倍服务加速；当前只证实上述一次运行的服务指标，不宣称已经找到最高 QPS。显存占用约 22,724 MiB。

## 构建及复核

远端源码：/home/yutao.he/lmdeploy-prefill-splitkv；独立构建容器 lmdeploy-prefill-splitkv-build 已停止，可启动后继续使用。

编译并打包原生模块（在构建容器 /workspace 中执行）：

```bash
bash tools/build_prefill_split_kv.sh
```

该构建已使用 CUDA 12.8、CMake 3.28.3、镜像自带 pybind11 3.1.0；显式指向 pip CUDA 头文件和库。未安装 Python 包或修改生产 Python 环境。构建出现已有 DLPack enum 的 ODR 警告，已记录；单 GPU 导入、dtype、模型启动和输出验证通过，不代表其他后端/功能的兼容性已验证。

构建派生镜像（在远端源码目录执行；PATCH_SHA256 来自实际源码清单）：

```bash
docker build \
  --network=none \
  --build-arg SOURCE_REV=7998cdbaf713393b9155358a147226def2557467 \
  --build-arg PATCH_SHA256=6babd1e38d7933fae5512dc24f977e881cc11f88263122138258ae6aa97dd4f4 \
  -t lmdeploy_cu128:v0.18.0_sp02_splitkv_20261028_r3 \
  -f docker/Dockerfile.prefill-split-kv \
  build-splitkv/image
```

上述 patch hash 覆盖生成 r3 时的四个实现文件：unified_attention_layer.cc/.h、attention_universal.h、kernels/gemm/CMakeLists.txt。最终测试与脚本另外由 source-manifest.json 记录，避免把测试后补充文件冒充已编译实现差异。

核对现有服务与 512 条参考（只做正确性检查）：

```bash
python3 tools/verify_prefill_split_kv.py \
  --mode compare \
  --dataset /data/users/xueliang.wu/me-k6-tools/llm/dfm2.0-s5-s-v3.6.49/testcase_dfm2.0-s5-s-v3.6.49.txt \
  --reference /home/yutao.he/lmdeploy-profile-v166-20261027/deploy-20261028/mentor-reference-512.json \
  --url http://127.0.0.1:23333 \
  --concurrency 8
```

## 回退

下列操作会停止新版并恢复旧版，需要在有切换授权时执行。当前无需执行，服务已经运行新版。

```bash
docker stop -t 20 lmdeploy-v166-splitkv
docker start c258526a2236
docker exec -d c258526a2236 sh -c 'exec /opt/py3/bin/python /tmp/lmdeploy-deep/diagnostic_entry.py --state /tmp/lmdeploy-deep/original.json --restore > /tmp/lmdeploy-deep/restored-server.log 2>&1'
```

旧容器原入口是 Bash，所以只 docker start 不会自动恢复推理；第三条会直接 exec 原始命令，不启用诊断包装。状态文件保留在旧容器内，不导出其中的环境变量。验收失败的候选容器已停止并改名保留，新旧镜像均未删除。

## 证据文件

本地报告目录 D:/aispeech/reports/lmdeploy-splitkv-deploy-20261028 保存新版 k6 汇总、固定样本验收摘要、CUDA 测试与 memcheck 日志、部署验证以及完整文件清单。原始测试日志含输入片段，仅保留在远端 /home/yutao.he/lmdeploy-profile-v166-20261027/deploy-20261028/。
