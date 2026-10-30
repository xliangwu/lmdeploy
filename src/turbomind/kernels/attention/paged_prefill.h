// Copyright (c) OpenMMLab. All rights reserved.
// 作用：统一prefill分页KV直读的资格判断，保证生产者跳过展平与消费者选路完全一致。
#pragma once

#include <cstdlib>
#include <cstring>

namespace turbomind::attention {

// 读取进程启动时的分页开关；首次取值后固定，避免生产者与消费者在运行中改变布局。
inline bool PagedPrefillEnabled()
{
    // 默认关闭；仅显式设为1才启用，其他模型和既有部署保持原路径。
    static const bool enabled = [] {
        const char* value = std::getenv("TM_PREFILL_PAGED_KV");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// 在分配连续临时缓冲之前按硬件、每 rank 形状和页表可用性判断是否支持分页直读。
inline bool CanUsePagedPrefill(int arch,
                              int head_dim,
                              int query_heads,
                              int kv_heads,
                              int block_len,
                              int cp_size,
                              int quant_policy,
                              bool has_blocks)
{
    // 本轮只开放V166当前每rank的SM86、12/2头、未量化KV、64-token页；更广场景需单独验证。
    // CTA_S=64不得跨越物理页，页表和层偏移仍沿用引擎提供的数据。
    return PagedPrefillEnabled() && arch == 86 && head_dim == 128 && query_heads == 12 && kv_heads == 2
           && block_len == 64 && cp_size == 1 && quant_policy == 0 && has_blocks;
}

// 消费者用完整参数再次校验契约；非因果、非16位输入或负层偏移都不能进入受限分页内核。
template<class Params>
inline bool CanUsePagedPrefill(const Params& params)
{
    return sizeof(*params.q) == 2 && params.causal && params.block_iter_params.offset >= 0
           && CanUsePagedPrefill(params.arch,
                                 params.size_per_head,
                                 params.num_heads,
                                 params.num_kv_heads,
                                 params.block_iter_params.block_len,
                                 (int)params.cp_size,
                                 params.quant_policy,
                                 params.block_iter_params.block_ptrs && params.block_iter_params.cu_block_nums);
}

}  // namespace turbomind::attention
