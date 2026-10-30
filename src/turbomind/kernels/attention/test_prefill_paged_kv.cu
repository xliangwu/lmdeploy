// Copyright (c) OpenMMLab. All rights reserved.
// 作用：用真实乱序页表验证 prefill 分页 KV 直读、ProcessKV 写入边界和双 stream 隔离。
// 执行：TM_PREFILL_PAGED_KV=1 TM_PREFILL_QUERY_TILE=32 TM_PREFILL_LOG_DISPATCH=1 ./test_prefill_paged_kv
// 也可设 TM_PREFILL_QUERY_TILE=64 单独覆盖生产默认的 64 行消费者；本测试不做性能扫描。
// 定向算子诊断：TM_PREFILL_PAGED_KV=1 TM_PREFILL_QUERY_TILE=64 ./test_prefill_paged_kv --benchmark
// --benchmark 只验证并计时固定 BF16 batch4/query26/KV7345，不运行完整正确性套件，不代表服务端到端收益。
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/paged_prefill.h"

namespace {

// 将 CUDA 错误转换为测试失败，禁止仅打印错误后继续给出成功结论。
void Check(cudaError_t result)
{
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// 管理少量托管内存，便于同步后进行独立 CPU 检查；size 的单位是元素。
template<class T> struct Buffer {
    T* data{};
    size_t size{};
    explicit Buffer(size_t count, T initial = T{}): size(count)
    {
        Check(cudaMallocManaged(&data, count * sizeof(T)));
        std::fill_n(data, count, initial);
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// 统计累计 token/page 数量；仅用于测试固定形状的内存规划。
int Sum(const std::vector<int>& values)
{
    return std::accumulate(values.begin(), values.end(), 0);
}

// 各序列独立向上取整，不能用整批 token 总数代替真实页表长度。
int PageCount(const std::vector<int>& lengths)
{
    int count = 0;
    for (int length : lengths) count += (length + 63) / 64;
    return count;
}

template<class T> struct Case {
    static constexpr int H = 12, KH = 2, D = 128, Page = 64, Guard = 8, MaxSplits = 8;
    static constexpr int PackedHeads = H + 2 * KH, PageGuard = 128, TablePrefix = 3;
    using Layout = turbomind::block::Layout<turbomind::block::Config<T, T, D>>;

    std::string name;
    std::vector<int> qlens, klens;
    int offset, tokens, cache_tokens, logical_pages, physical_pages, layer_bytes, page_bytes;
    bool transform, shared_prefix, saw_split{};
    float amplitude;
    double max_linear_error{}, max_cpu_error{}, max_producer_error{};
    Layout layout{turbomind::block::Config<T, T, D>{KH, Page}};
    cudaStream_t stream{};
    Buffer<T> qkv, bias, flat, reference, output;
    Buffer<unsigned char> pages;
    Buffer<char*> table;
    Buffer<int> q_offsets, k_offsets, block_offsets, readonly_blocks, split_counts;
    Buffer<float> partial_o, partial_ml;
    Buffer<bool> finished;
    std::vector<unsigned char> before_process, writable, after_process;
    turbomind::AttentionParams<T> params{};

    // 为一个形状准备独立 stream、真实页表、三层页存储及严格的可写范围快照。
    Case(std::string label, std::vector<int> qs, std::vector<int> ks, int prefix, int window,
         bool use_transform = true, bool share = false, float scale = .5f):
        name(std::move(label)), qlens(qs), klens(ks), offset(prefix), tokens(Sum(qs)), cache_tokens(Sum(ks)),
        logical_pages(PageCount(ks)), physical_pages(logical_pages + 3),
        layer_bytes(KH * 2 * Page * D * sizeof(T)), page_bytes(3 * layer_bytes + 2 * PageGuard),
        transform(use_transform), shared_prefix(share), amplitude(scale),
        qkv((prefix + tokens) * PackedHeads * D), bias(PackedHeads * D),
        flat(KH * 2 * (cache_tokens + Page) * D, T(6.f)),
        reference((prefix + tokens + Guard) * H * D, T(5.f)),
        output((prefix + tokens + Guard) * H * D, T(5.f)),
        pages(size_t(physical_pages) * page_bytes, 0xa5), table(TablePrefix + logical_pages + Guard),
        q_offsets(qs.size() + 1), k_offsets(qs.size() + 1), block_offsets(qs.size() + 1),
        readonly_blocks(qs.size(), share ? 1 : 0), split_counts(tokens + 2 * Guard, -99),
        partial_o(size_t(tokens) * MaxSplits * H * D + 2 * Guard, -777.f),
        partial_ml(size_t(tokens) * MaxSplits * H * 2 + 2 * Guard, -777.f), finished(qs.size(), false)
    {
        if (qs.empty() || qs.size() != ks.size()) throw std::runtime_error("invalid case shape");
        // 测试布局公式独立书写，先明确生产布局满足未量化 [H,2,S,D] 契约。
        if (layout.layer_size() != layer_bytes) throw std::runtime_error("unexpected block layer size");
        for (int h = 0; h < KH; ++h) for (int ti = 0; ti < Page; ++ti) {
            if (layout.k_data(h, ti) != ((h * 2) * Page + ti) * D * int(sizeof(T)) ||
                layout.v_data(h, ti) != ((h * 2 + 1) * Page + ti) * D * int(sizeof(T)))
                throw std::runtime_error("block layout violates independent H2SD contract");
        }
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::mt19937 generator(20261029 + tokens + offset);
        std::uniform_real_distribution<float> uniform(-amplitude, amplitude);
        for (size_t i = 0; i < qkv.size; ++i) qkv.data[i] = T(uniform(generator));
        for (size_t i = 0; i < bias.size; ++i) bias.data[i] = T(uniform(generator) * .2f);

        // 页表前方保留其他批次的三个条目；物理页随机重排，额外物理页始终不可写。
        std::vector<int> permutation(logical_pages);
        std::iota(permutation.begin(), permutation.end(), 1);
        std::shuffle(permutation.begin(), permutation.end(), generator);
        for (size_t i = 0; i < table.size; ++i) table.data[i] = PhysicalPage(physical_pages - 1);
        for (int i = 0; i < logical_pages; ++i) table.data[TablePrefix + i] = PhysicalPage(permutation[i]);
        q_offsets.data[0] = prefix;
        k_offsets.data[0] = 17;
        block_offsets.data[0] = TablePrefix;
        for (size_t b = 0; b < qs.size(); ++b) {
            if (qs[b] <= 0 || ks[b] < qs[b]) throw std::runtime_error("invalid sequence length");
            q_offsets.data[b + 1] = q_offsets.data[b] + qs[b];
            k_offsets.data[b + 1] = k_offsets.data[b] + ks[b];
            block_offsets.data[b + 1] = block_offsets.data[b] + (ks[b] + Page - 1) / Page;
        }
        if (share) {
            // 两个完整 prefill 都包含首个只读页，必须跳过其中的写入；最后一页各自独立。
            if (qs != std::vector<int>({65, 65}) || ks != qs)
                throw std::runtime_error("shared case must exercise readonly writes");
            table.data[block_offsets.data[1]] = table.data[block_offsets.data[0]];
        }
        for (size_t b = 0; b < qs.size(); ++b) {
            for (int ti = 0; ti < ks[b]; ++ti) {
                // 共享页只初始化一次，令两个请求确实读取同一份已有缓存内容。
                if (share && b == 1 && ti < Page) continue;
                for (int h = 0; h < KH; ++h) for (int v = 0; v < 2; ++v)
                    for (int d = 0; d < D; ++d) Cache(b, ti, h, v)[d] = T(uniform(generator));
            }
        }
        before_process.assign(pages.data, pages.data + pages.size);
        writable.assign(pages.size, 0);
        for (size_t b = 0; b < qs.size(); ++b) {
            const int first = std::max(ks[b] - qs[b], readonly_blocks.data[b] * Page);
            for (int ti = first; ti < ks[b]; ++ti) for (int h = 0; h < KH; ++h) for (int v = 0; v < 2; ++v) {
                const size_t address = reinterpret_cast<unsigned char*>(Cache(b, ti, h, v)) - pages.data;
                std::fill_n(writable.begin() + address, D * sizeof(T), 1);
            }
        }

        params.q = qkv.data;
        params.k = qkv.data + H * D;
        params.v = qkv.data + (H + KH) * D;
        params.stride = PackedHeads * D;
        params.q_bias = use_transform ? bias.data : nullptr;
        params.k_bias = use_transform ? bias.data + H * D : nullptr;
        params.v_bias = use_transform ? bias.data + (H + KH) * D : nullptr;
        params.token_num = tokens;
        params.batch_size = qs.size();
        params.max_q_len = *std::max_element(qs.begin(), qs.end());
        params.max_k_len = *std::max_element(ks.begin(), ks.end());
        params.num_heads = H;
        params.num_kv_heads = KH;
        params.size_per_head = D;
        params.cu_q_len = q_offsets.data;
        params.cu_k_len = k_offsets.data;
        params.readonly_block_num = readonly_blocks.data;
        params.finished = finished.data;
        params.window_size = window;
        params.inv_sqrt_dh = std::log2(std::exp(1.f)) / std::sqrt(float(D));
        params.block_iter_params = {table.data, block_offsets.data, layer_bytes, Page};
        params.partial_O = partial_o.data + Guard;
        params.partial_ML = partial_ml.data + Guard;
        params.split_cnt = split_counts.data + Guard;
        params.offset_q = prefix;
        params.arch = 86;
        params.stream = stream;
        if (use_transform) {
            params.rope_param.type = turbomind::RopeType::kDefault;
            params.rope_param.dim = D;
            params.rope_param.scale_factor = -std::log2(10000.f) / D;
            params.rope_param.inv_factor = 1.f;
        }
    }

    // stream 在成员内存释放前销毁；正常验证已等待全部工作结束。
    ~Case() { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }

    // 返回物理页起点，保留页首 guard；层偏移另外通过 BlockIteratorParams 提供。
    char* PhysicalPage(int physical)
    {
        return reinterpret_cast<char*>(pages.data + size_t(physical) * page_bytes + PageGuard);
    }

    // 独立按未量化 [H,2,S,D] 公式解析真实页表，不复用生产者 Layout 的寻址实现。
    T* Cache(size_t b, int token, int head, int value)
    {
        char* block = table.data[block_offsets.data[b] + token / Page];
        const int data_offset = ((head * 2 + value) * Page + token % Page) * D * sizeof(T);
        return reinterpret_cast<T*>(block + layer_bytes + data_offset);
    }

    // 模拟低精度 bias 与相邻元素 RoPE，返回一个维度的主机参考值。
    float Transform(const T* input, const T* input_bias, int dim, int position, bool rotate)
    {
        const int pair = dim & ~1;
        const float x = float(T(float(input[pair]) + (input_bias ? float(input_bias[pair]) : 0.f)));
        const float y = float(T(float(input[pair + 1]) + (input_bias ? float(input_bias[pair + 1]) : 0.f)));
        if (!rotate) return dim & 1 ? y : x;
        const float angle = position * std::exp2(pair * params.rope_param.scale_factor);
        const float c = float(T(std::cos(angle))), s = float(T(std::sin(angle)));
        // 每次乘法及加减均按目标 dtype 舍入，对应 FastRoPE 的低精度运算语义。
        return dim & 1 ? float(T(float(T(c * y)) + float(T(s * x))))
                       : float(T(float(T(c * x)) - float(T(s * y))));
    }

    // 执行生产者和线性参考；候选之后只能访问已处理的真实页，绝不自行构造线性 KV。
    void Prepare()
    {
        turbomind::invokeProcessKV_v2_(params);
        params.linear_iter_params = {flat.data, 2 * (cache_tokens + Page) * D, (cache_tokens + Page) * D};
        turbomind::invokeFlattenKV_v2_(params, cache_tokens);
        params.max_split_k = 1;
        params.out = reference.data;
        turbomind::dispatchAttention(params);
        Check(cudaGetLastError());
        Check(cudaStreamSynchronize(stream));
        VerifyProducer();
        after_process.assign(pages.data, pages.data + pages.size);
        // 线性副本故意污染为 NaN，避免任何意外复用连续布局仍能通过数值对照。
        std::fill_n(flat.data, flat.size, T(std::numeric_limits<float>::quiet_NaN()));
        params.linear_iter_params = {nullptr, 0, 0};
    }

    // 检查生产者仅写可写 token，并确认 flatten 的内容逐元素等于同一分页缓存。
    void VerifyProducer()
    {
        for (size_t i = 0; i < pages.size; ++i)
            if (!writable[i] && pages.data[i] != before_process[i])
                throw std::runtime_error(name + ": ProcessKV overwrote readonly page/layer/tail/guard");
        const double tolerance = std::is_same_v<T, nv_bfloat16> ? .025 : .003;
        for (size_t b = 0; b < qlens.size(); ++b) for (int ti = 0; ti < klens[b]; ++ti) {
            const int first = std::max(klens[b] - qlens[b], readonly_blocks.data[b] * Page);
            for (int h = 0; h < KH; ++h) for (int v = 0; v < 2; ++v) for (int d = 0; d < D; ++d) {
                const float cached = float(Cache(b, ti, h, v)[d]);
                const int seq = k_offsets.data[b] - k_offsets.data[0];
                const size_t linear = (h * 2 * (cache_tokens + Page) + v * (cache_tokens + Page) + seq + ti) * D + d;
                if (!std::isfinite(cached) || cached != float(flat.data[linear]))
                    throw std::runtime_error(name + ": flatten does not match page layout");
                if (ti < first) continue;
                const int row = q_offsets.data[b] + ti - (klens[b] - qlens[b]);
                const T* input = qkv.data + row * PackedHeads * D + (H + v * KH + h) * D;
                const T* input_bias = transform ? bias.data + (H + v * KH + h) * D : nullptr;
                const float expected = Transform(input, input_bias, d, ti, transform && !v);
                max_producer_error = std::max(max_producer_error, double(std::abs(cached - expected)));
                if (std::abs(cached - expected) > tolerance * std::max(1.f, amplitude))
                    throw std::runtime_error(name + ": ProcessKV bias/RoPE CPU oracle mismatch");
            }
        }
    }

    // 在两路 stream 提交前统一重置主机可见输出，避免 GPU 工作期间托管内存的 CPU 写入。
    void ResetOutput()
    {
        std::fill_n(output.data, output.size, T(5.f));
        std::fill_n(split_counts.data, split_counts.size, -99);
    }

    // 必须满足分页资格且连续指针为空；禁用分页时测试直接失败，不能误测线性路径。
    void Launch(int splits)
    {
        if (params.linear_iter_params.kv_cache || !turbomind::attention::CanUsePagedPrefill(params))
            throw std::runtime_error("paged test requires TM_PREFILL_PAGED_KV=1 and null linear pointer");
        params.max_split_k = splits;
        params.out = output.data;
        turbomind::dispatchAttention(params);
        Check(cudaGetLastError());
    }

    // 复用同一页表和输入，提交包含生产者的完整算子链；定向比较仅改变 KV 消费布局。
    void LaunchPipeline(bool paged)
    {
        auto p = params;
        p.max_split_k = MaxSplits;
        // 两条路径都写入同一个输出缓冲；调用方随后用已验证的线性参考比较分页结果。
        p.out = output.data;
        p.linear_iter_params = paged ? turbomind::LinearIteratorParams{nullptr, 0, 0}
                                     : turbomind::LinearIteratorParams{
                                           flat.data, 2 * (cache_tokens + Page) * D, (cache_tokens + Page) * D};
        if (paged && !turbomind::attention::CanUsePagedPrefill(p))
            throw std::runtime_error("fixed benchmark requires a supported paged layout");
        turbomind::invokeProcessKV_v2_(p);
        if (!paged) turbomind::invokeFlattenKV_v2_(p, cache_tokens);
        turbomind::dispatchAttention(p);
        Check(cudaGetLastError());
    }

    // 输出精度阈值随 V 的幅度及 dtype 改变，绝对误差与相对误差同时约束。
    double Tolerance() const
    {
        return std::max(.002, double(amplitude) * (std::is_same_v<T, nv_bfloat16> ? 1. / 256 : 1. / 2048));
    }

    // 检查全量输出、所有内存保护区，并确保 attention 消费者没有改写缓存。
    void Verify()
    {
        Check(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < output.size; ++i) {
            const bool valid = i >= size_t(offset) * H * D && i < size_t(offset + tokens) * H * D;
            const float expected = valid ? float(reference.data[i]) : 5.f;
            const float actual = float(output.data[i]);
            if (valid) max_linear_error = std::max(max_linear_error, double(std::abs(actual - expected)));
            if (!std::isfinite(actual) || !std::isfinite(expected)
                || (valid ? std::abs(actual - expected) > 2 * Tolerance() + .02 * std::abs(expected)
                                                 : actual != expected))
                throw std::runtime_error(name + ": paged/linear output mismatch at " + std::to_string(i));
        }
        for (int i = 0; i < Guard; ++i) {
            if (split_counts.data[i] != -99 || split_counts.data[Guard + tokens + i] != -99 ||
                partial_o.data[i] != -777.f || partial_o.data[partial_o.size - Guard + i] != -777.f ||
                partial_ml.data[i] != -777.f || partial_ml.data[partial_ml.size - Guard + i] != -777.f)
                throw std::runtime_error(name + ": split workspace guard overwritten");
        }
        for (int i = 0; i < tokens; ++i) saw_split = saw_split || split_counts.data[Guard + i] > 1;
        if (std::memcmp(pages.data, after_process.data(), pages.size) != 0)
            throw std::runtime_error(name + ": paged attention modified cache");
        Oracle();
    }

    // 从真实页表读取 K/V，以 FP64 softmax 检查首尾 query 和每个 GQA 组的边界 head。
    void Oracle()
    {
        for (size_t b = 0; b < qlens.size(); ++b) for (int qi : {0, qlens[b] - 1}) for (int h : {0, 5, 6, 11}) {
            const int row = q_offsets.data[b] + qi, position = klens[b] - qlens[b] + qi;
            const int end = position + 1, begin = std::max(0, end - params.window_size), kh = h / (H / KH);
            std::vector<double> scores(end - begin), expected(D, 0.), query(D);
            const T* input = qkv.data + row * PackedHeads * D + h * D;
            for (int d = 0; d < D; ++d)
                query[d] = Transform(input, transform ? bias.data + h * D : nullptr, d, position, transform);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int ti = begin; ti < end; ++ti) {
                double dot = 0.;
                for (int d = 0; d < D; ++d) dot += query[d] * float(Cache(b, ti, kh, 0)[d]);
                maximum = std::max(maximum, scores[ti - begin] = dot / std::sqrt(double(D)));
            }
            double denominator = 0.;
            for (int ti = begin; ti < end; ++ti) {
                const double probability = std::exp(scores[ti - begin] - maximum);
                denominator += probability;
                for (int d = 0; d < D; ++d) expected[d] += probability * float(Cache(b, ti, kh, 1)[d]);
            }
            for (int d = 0; d < D; ++d) {
                const double value = expected[d] / denominator;
                const double actual = float(output.data[(row * H + h) * D + d]);
                max_cpu_error = std::max(max_cpu_error, std::abs(actual - value));
                if (std::abs(actual - value) > Tolerance() + .02 * std::abs(value))
                    throw std::runtime_error(name + ": CPU attention oracle mismatch at row/head/dim="
                        + std::to_string(row) + "/" + std::to_string(h) + "/" + std::to_string(d)
                        + " actual=" + std::to_string(actual) + " expected=" + std::to_string(value));
            }
        }
    }
};

// 固定正确性形状覆盖 query 分派边界、非整页上下文、分片归并、窗口及共享只读页。
template<class T> void RunSuite(const char* dtype)
{
    constexpr int unlimited = 1 << 28;
    const std::vector<std::vector<int>> queries{{1}, {31}, {32}, {33}, {63}, {64}, {65},
                                               {1, 31, 32, 33, 64, 65}, {65, 65}};
    const std::vector<std::vector<int>> contexts{{65}, {12289}, {1025}, {321}, {577}, {4095}, {4097},
                                                {65, 129, 193, 257, 513, 1025}, {65, 65}};
    bool saw_split = false;
    double max_linear_error = 0., max_cpu_error = 0., max_producer_error = 0.;
    for (size_t i = 0; i < queries.size(); ++i) {
        const int window = i == 3 ? 73 : i == 5 ? 128 : i == 7 ? 65 : unlimited;
        Case<T> c(std::string(dtype) + " case=" + std::to_string(i), queries[i], contexts[i], i % 2 ? 7 : 3,
                  window, i != 0, i == 8, i == 3 ? 2.f : .5f);
        c.Prepare();
        for (int splits : {1, 8}) { c.ResetOutput(); c.Launch(splits); c.Verify(); }
        saw_split = saw_split || c.saw_split;
        max_linear_error = std::max(max_linear_error, c.max_linear_error);
        max_cpu_error = std::max(max_cpu_error, c.max_cpu_error);
        max_producer_error = std::max(max_producer_error, c.max_producer_error);
    }
    // max_split_k 仅是上限；必须观察到真实多个分片，才算覆盖过归并路径。
    if (!saw_split) throw std::runtime_error(std::string(dtype) + ": no actual split-KV reduction observed");
    // 两个非默认 prefill stream 使用独立页表和 workspace，检查 32/64 行分派的资源隔离。
    Case<T> first(std::string(dtype) + " stream-a", {31}, {4097}, 7, unlimited);
    Case<T> second(std::string(dtype) + " stream-b", {33}, {257}, 3, 73);
    first.Prepare(); second.Prepare();
    for (int repeat = 0; repeat < 2; ++repeat) {
        first.ResetOutput(); second.ResetOutput();
        first.Launch(8); second.Launch(8);
        // 两路 GPU 工作都结束后才访问托管内存；此检查不声称测量了 decode/prefill 实际重叠。
        Check(cudaStreamSynchronize(first.stream)); Check(cudaStreamSynchronize(second.stream));
        first.Verify(); second.Verify();
    }
    max_linear_error = std::max({max_linear_error, first.max_linear_error, second.max_linear_error});
    max_cpu_error = std::max({max_cpu_error, first.max_cpu_error, second.max_cpu_error});
    max_producer_error = std::max({max_producer_error, first.max_producer_error, second.max_producer_error});
    std::cout << dtype << ": 18 real-page shape/split checks + 2 two-stream isolation checks passed"
              << " max_abs_linear=" << max_linear_error << " max_abs_cpu=" << max_cpu_error
              << " max_abs_process_kv=" << max_producer_error << '\n';
}

#if ENABLE_BF16
// 先热身再使用同一 CUDA stream 的 event 计时；返回固定算子链的每轮平均微秒数。
double MeasurePipeline(Case<nv_bfloat16>& test, bool paged)
{
    constexpr int warmup = 10, iterations = 100;
    cudaEvent_t begin{}, end{};
    Check(cudaEventCreate(&begin));
    Check(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) test.LaunchPipeline(paged);
    Check(cudaEventRecord(begin, test.stream));
    for (int i = 0; i < iterations; ++i) test.LaunchPipeline(paged);
    Check(cudaEventRecord(end, test.stream));
    Check(cudaEventSynchronize(end));
    float elapsed_ms{};
    Check(cudaEventElapsedTime(&elapsed_ms, begin, end));
    Check(cudaEventDestroy(begin));
    Check(cudaEventDestroy(end));
    return double(elapsed_ms) * 1000. / iterations;
}

// 只运行一个接近已有热请求均值的形状；先验数值，再比较是否删除 flatten 的算子链耗时。
void RunFixedBenchmark()
{
    Case<nv_bfloat16> test("BF16 fixed-pipeline", {26, 26, 26, 26}, {7345, 7345, 7345, 7345}, 7, 1 << 28);
    test.Prepare();
    test.ResetOutput();
    // 两路使用相同 max_split_k=8；线性参考也重新执行 ProcessKV，避免混入旧单分片参考。
    test.ResetOutput();
    test.LaunchPipeline(false);
    Check(cudaStreamSynchronize(test.stream));
    test.VerifyProducer();
    // 先确认固定场景的线性生产者+展平+attention输出仍与 Prepare 的参考一致。
    test.Verify();
    std::fill_n(test.flat.data, test.flat.size, nv_bfloat16(std::numeric_limits<float>::quiet_NaN()));
    test.ResetOutput();
    test.LaunchPipeline(true);
    test.Verify();
    std::cout << "fixed_pipeline_correctness linear_and_paged_compare passed max_abs_linear=" << test.max_linear_error
              << " max_abs_cpu=" << test.max_cpu_error << " max_abs_process_kv=" << test.max_producer_error << '\n';

    // 计时期间不访问托管内存的主机副本；两条路径分别热身，排除验证后的首次页面迁移。
    const double linear_us = MeasurePipeline(test, false);
    const double paged_us = MeasurePipeline(test, true);
    const char* tile = std::getenv("TM_PREFILL_QUERY_TILE");
    std::cout << "operator_diagnostic_only dtype=BF16 batch=4 query=26 kv=7345 max_split_k=8 query_tile="
              << (tile ? tile : "64") << " warmup=10 iterations=100"
              << " linear_process_flatten_attention_mean_us=" << linear_us
              << " paged_process_attention_mean_us=" << paged_us
              << " paged_over_linear=" << paged_us / linear_us
              << " end_to_end_qps_claim=false\n";
}
#endif

}  // namespace

// 默认只验数值和布局，显式 --benchmark 执行固定算子诊断；所有模式均要求分页开关和真实 SM86。
int main(int argc, char** argv)
{
    try {
        if (!turbomind::attention::PagedPrefillEnabled())
            throw std::runtime_error("required environment: TM_PREFILL_PAGED_KV=1");
        if (argc > 2) throw std::runtime_error("at most one test mode is allowed");
        const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
        if (argc > 1 && !benchmark) {
            // 子进程负向检查必须在分派入口拒绝不支持的布局，而不是尝试读取空连续指针。
            const std::string mode = argv[1];
            if (mode != "--invalid-layout" && mode != "--invalid-noncausal")
                throw std::runtime_error("supported arguments: --benchmark, --invalid-layout, --invalid-noncausal");
            turbomind::AttentionParams<half> invalid{};
            char* block{};
            int offsets[2]{};
            invalid.arch = 86;
            invalid.num_heads = 12;
            invalid.num_kv_heads = 2;
            invalid.size_per_head = 128;
            invalid.causal = mode != "--invalid-noncausal";
            invalid.block_iter_params = {&block, offsets, 0, mode == "--invalid-layout" ? 32 : 64};
            turbomind::dispatchAttention(invalid);
            throw std::runtime_error("invalid layout was unexpectedly accepted by dispatchAttention");
        }
        int device{};
        cudaDeviceProp prop{};
        Check(cudaGetDevice(&device));
        Check(cudaGetDeviceProperties(&prop, device));
        if (prop.major != 8 || prop.minor != 6) throw std::runtime_error("test requires an actual SM86 GPU");
        if (benchmark) {
#if ENABLE_BF16
            RunFixedBenchmark();
            return 0;
#else
            throw std::runtime_error("benchmark requires an ENABLE_BF16 build");
#endif
        }
        RunSuite<half>("FP16");
#if ENABLE_BF16
        RunSuite<nv_bfloat16>("BF16");
#else
        throw std::runtime_error("test requires an ENABLE_BF16 build");
#endif
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
