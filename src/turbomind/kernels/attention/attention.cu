// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/attention/paged_prefill.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    using namespace attention;

    // 空连续缓冲是生产者选择分页布局的明确标记；先校验再查registry，避免不支持的
    // 形状或非因果调用选中线性内核后访问空指针。已有非空连续缓冲调用仍按原布局执行。
    if (!params.linear_iter_params.kv_cache) {
        TM_CHECK(CanUsePagedPrefill(params))
            << "Prefill KV layout mismatch: null linear KV requires supported causal paged KV; arch="
            << params.arch << ", head_dim=" << params.size_per_head << ", query_heads=" << params.num_heads
            << ", kv_heads=" << params.num_kv_heads << ", block_len=" << params.block_iter_params.block_len
            << ", cp_size=" << (int)params.cp_size << ", quant_policy=" << params.quant_policy
            << ", causal=" << params.causal;
    }

    auto&    reg = Registry::instance();
    AttnDesc desc{};
    desc.mode      = AttnDesc::kPrefill;
    desc.head_dim  = params.size_per_head;
    desc.data_type = data_type_v<T>;
    desc.causal    = params.causal;

    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No attention kernel found: " + to_string(desc);

    TM_SCOPE_CALL(kernel->Launch(&params, reg.sm_count()));
}

template void dispatchAttention(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
