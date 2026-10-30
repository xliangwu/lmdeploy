// Copyright (c) OpenMMLab. All rights reserved.
// 作用：为64-token页、128维未量化KV的prefill提供轻量分页迭代器，减少热点循环的地址状态。
#pragma once

#include <cstdint>

#include "block_iterator.h"

namespace turbomind {

template<class T, int CTA_S, int HeadDim>
struct PagedPrefillIterator {
    static_assert(CTA_S == 64 && HeadDim == 128 && sizeof(T) == 2);
    static constexpr int kKeyToValue = CTA_S * HeadDim;

    char**   table_begin_;
    char**   table_current_;
    uint32_t head_offset_;

    // 仅保留页表起点、当前页游标和层/head字节偏移，不让K地址跨越整个计算循环存活。
    __device__ PagedPrefillIterator(char** block_ptrs, uint32_t head_offset):
        table_begin_{block_ptrs}, table_current_{block_ptrs}, head_offset_{head_offset}
    {
    }

    // 本受限路径一整个tile恰好是一页，tile编号就是页表下标，无需除余定位页内token。
    __device__ void SetTile(int tile_id)
    {
        table_current_ = table_begin_ + tile_id;
    }

    // 逆向访问下一页；流水线可能在最后一个有效tile后继续预取，因此游标最小停在第一页。
    __device__ void Advance()
    {
        if (table_current_ > table_begin_) {
            --table_current_;
        }
    }

    // K/V在同一head内连续存储；V偏移为编译期常量，只开放未量化路径实际使用的两个索引。
    template<int Index>
    __device__ const T* OffsetPtr(int offset) const
    {
        static_assert(Index == 0 || Index == 1);
        // 地址只在当前加载位置生成，以缩短64位K指针的寄存器生命周期。
        const T* key = reinterpret_cast<const T*>(*table_current_ + head_offset_);
        return key + offset + Index * kKeyToValue;
    }
};

template<class T, int CTA_S, int HeadDim>
struct PagedPrefillIteratorFactory {
    static_assert(CTA_S == 64 && HeadDim == 128 && sizeof(T) == 2);
    // 保留BlockLayout类型和成员顺序，以复用已有CreateCacheIterFactory参数适配器。
    using BlockLayout = block::Layout<block::Config<T, T, HeadDim, false>>;

    BlockLayout block_layout_;
    char**      block_ptrs_;
    const int*  cu_block_nums_;
    int         offset_;

    // 每个CTA只按生产者的block::Layout计算一次层/head偏移，迭代器后续仅加载页表和加常量。
    __device__ auto Create(int batch_idx, int head_idx)
    {
        // 资格检查保证层偏移非负且只有两个KV head，head内偏移最多32768字节。
        // 两项先转uint32_t再相加，避免合法int层偏移接近INT_MAX时发生有符号溢出。
        const uint32_t head_offset = static_cast<uint32_t>(offset_)
                                     + static_cast<uint32_t>(block_layout_.k_data(head_idx, 0));
        return PagedPrefillIterator<T, CTA_S, HeadDim>{block_ptrs_ + cu_block_nums_[batch_idx], head_offset};
    }
};

}  // namespace turbomind
