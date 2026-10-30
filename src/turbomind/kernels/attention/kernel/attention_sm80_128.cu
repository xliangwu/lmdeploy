// Copyright (c) OpenMMLab. All rights reserved.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_16816.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/kernels/attention/paged_prefill.h"
#include "src/turbomind/kernels/attention/paged_prefill_iterator.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 16;
constexpr int kStages  = 2;

template<class T, bool Causal = true, int QueryTile = kCTA_Q, bool Paged = false>
using KT = AttentionUniversal<
    arch::Sm80,
    Mainloop<Sm80_CpAsync<kStages>, Impl<MMA_16816, T, T, 1, QueryTile, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    std::conditional_t<Paged,
                       PagedPrefillIteratorFactory<T, kCTA_S, kHeadDim>,
                       LinearIteratorFactory<T, kCTA_S, kHeadDim>>,
    AttentionCtaMap,
    Causal>;

template<class T>
class SmallQueryKernel final: public KernelImpl<KT<T>> {
    KernelImpl<KT<T, true, 32>> small_;
    KernelImpl<KT<T, true, 64, true>> paged_;
    KernelImpl<KT<T, true, 32, true>> small_paged_;
    const bool small_enabled_ = [] {
        const char* value = std::getenv("TM_PREFILL_QUERY_TILE");
        return value && std::strcmp(value, "32") == 0;
    }();

public:
    // 根据已声明的KV布局和query长度选择实际内核；分页标记失配时禁止回退到线性读取。
    bool Launch(const void* params, int sm_count) const override
    {
        // 只优化已验收的SM86短query且无context parallel场景，长prefill仍用原64行内核。
        const auto& p = *static_cast<const typename KT<T>::ParamType*>(params);
        const bool use_small = small_enabled_ && p.arch == 86 && p.cp_size == 1 && p.max_q_len <= 32;
        // 连续缓冲为空表示上游明确选择分页直读；不能仅凭形状改变其他调用者的布局契约。
        if (!p.linear_iter_params.kv_cache) {
            // dispatchAttention入口也检查此条件；保留这里的保护以覆盖直接Kernel::Launch调用。
            TM_CHECK(CanUsePagedPrefill(p))
                << "Prefill KV layout mismatch: paged request cannot fall back to a linear kernel";
            // 只替换KV地址迭代器；Q变换、causal/window掩码、GQA和分片归并均沿用原实现。
            static const bool log_dispatch = std::getenv("TM_PREFILL_LOG_DISPATCH") != nullptr;
            static std::atomic<bool> reported[2]{};
            if (log_dispatch && !reported[use_small ? 0 : 1].exchange(true)) {
                // 分别记录真正将启动的32/64行分页成员内核，不能使用wrapper继承的线性内核属性。
                // localSizeBytes为CUDA报告的每线程本地内存；max_ctas是理论资源上限，不是实测occupancy。
                const auto& info = use_small ? small_paged_.info() : paged_.info();
                std::fprintf(stderr,
                             "paged_prefill_dispatch dtype=%s query_tile=%d block_len=%d max_q_len=%d max_k_len=%d "
                             "warps=%d registers=%d local_bytes_per_thread=%zu static_smem=%zu dynamic_smem=%d "
                             "max_ctas=%d\n",
                             turbomind::to_string(data_type_v<T>),
                             use_small ? 32 : 64,
                             p.block_iter_params.block_len,
                             p.max_q_len,
                             p.max_k_len,
                             info.num_warps,
                             info.attr.numRegs,
                             info.attr.localSizeBytes,
                             info.attr.sharedSizeBytes,
                             info.dynamic_smem_size,
                             info.max_active_ctas);
            }
            return use_small ? small_paged_.Launch(params, sm_count) : paged_.Launch(params, sm_count);
        }
        if (use_small) {
            static const bool log_dispatch = std::getenv("TM_PREFILL_LOG_DISPATCH") != nullptr;
            static std::atomic<bool> reported{false};
            if (log_dispatch && !reported.exchange(true)) {
                // 仅诊断开关开启时输出一次实际选路，不记录任何输入内容。
                std::fprintf(stderr, "short_prefill_dispatch query_tile=32 max_q_len=%d warps=%d registers=%d max_ctas=%d\n",
                             p.max_q_len, small_.info().num_warps, small_.info().attr.numRegs, small_.info().max_active_ctas);
            }
            return small_.Launch(params, sm_count);
        }
        return KernelImpl<KT<T>>::Launch(params, sm_count);
    }
};

namespace {
Registrar reg([](Collector& c) {
    // 热缓存下常见未命中query少于32，显式开关允许用较小行块减少填充计算。
    // 默认64保持既有行为；不改变KV分片上限、数据类型或非因果路径。
    const char* tile = std::getenv("TM_PREFILL_QUERY_TILE");
    TM_CHECK(!tile || std::strcmp(tile, "32") == 0 || std::strcmp(tile, "64") == 0)
        << "TM_PREFILL_QUERY_TILE must be 32 or 64";
    const bool small = tile && std::strcmp(tile, "32") == 0;
    if (small || PagedPrefillEnabled()) {
        c.add_kernel<SmallQueryKernel<half>>();
    }
    else {
        c.add<KT<half>>();
    }
    c.add<KT<half, false>>();
#if ENABLE_BF16
    if (small || PagedPrefillEnabled()) {
        c.add_kernel<SmallQueryKernel<nv_bfloat16>>();
    }
    else {
        c.add<KT<nv_bfloat16>>();
    }
    c.add<KT<nv_bfloat16, false>>();
#endif
});
}  // namespace

}  // namespace turbomind::attention
