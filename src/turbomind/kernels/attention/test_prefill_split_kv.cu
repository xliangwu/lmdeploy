// Copyright (c) OpenMMLab. All rights reserved.
// 验证 prefill split-KV 的因果/GQA 数值、混合批次 offset、workspace 边界及双 stream 隔离。
// 执行：./test_prefill_split_kv；可选 --benchmark 输出单 kernel 的诊断计时。
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/registry.h"

namespace {

void Check(cudaError_t error)
{
    // CUDA 错误直接令测试失败，避免只打印差异却返回成功。
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> struct Buffer {
    T* data{};
    size_t size{};
    explicit Buffer(size_t n, T value = T{}): size(n)
    {
        // 托管内存便于独立 CPU oracle 读取；计时前先预热，排除首次迁移。
        Check(cudaMallocManaged(&data, n * sizeof(T)));
        std::fill_n(data, n, value);
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<class T> struct Case {
    static constexpr int H = 12, KH = 2, D = 128, WS = 4096, Guard = 8;
    std::vector<int> qlens, klens;
    int offset, tokens{}, cache_tokens{}, window;
    float value_scale;
    cudaStream_t stream{};
    Buffer<T> q, kv, reference, output;
    Buffer<float> partial_o, partial_ml;
    Buffer<int> q_offsets, k_offsets, split_counts;
    Buffer<bool> finished;
    turbomind::AttentionParams<T> params{};

    static int Sum(const std::vector<int>& v)
    {
        // 计算各序列有效长度之和，不包含预留 decode 行或 padding。
        int total = 0;
        for (int x : v) total += x;
        return total;
    }

    Case(std::vector<int> qs, std::vector<int> ks, int prefix, int win = 1 << 28, float amplitude = .5f):
        qlens(qs), klens(ks), offset(prefix), tokens(Sum(qs)), cache_tokens(Sum(ks)), window(win), value_scale(amplitude),
        q((prefix + tokens) * H * D), kv(KH * 2 * (cache_tokens + 64) * D),
        reference((prefix + tokens) * H * D, T(5.f)), output((prefix + tokens) * H * D, T(5.f)),
        partial_o(WS * H * D, std::numeric_limits<float>::quiet_NaN()),
        partial_ml(WS * H * 2, std::numeric_limits<float>::quiet_NaN()),
        q_offsets(qs.size()+1), k_offsets(qs.size()+1), split_counts(tokens+2*Guard, -99), finished(qs.size(), false)
    {
        // 模拟 decode 位于前方的全局 query 偏移；KV 起点也故意不从零开始。
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::mt19937 gen(20261027);
        std::uniform_real_distribution<float> uniform(-amplitude, amplitude);
        for (size_t i = 0; i < q.size; ++i) q.data[i] = T(uniform(gen));
        for (size_t i = 0; i < kv.size; ++i) kv.data[i] = T(uniform(gen));
        q_offsets.data[0] = offset;
        k_offsets.data[0] = 17;
        for (size_t i = 0; i < qs.size(); ++i) {
            q_offsets.data[i+1] = q_offsets.data[i] + qs[i];
            k_offsets.data[i+1] = k_offsets.data[i] + ks[i];
        }
        params.q = q.data;
        params.stride = H * D;
        params.token_num = tokens;
        params.batch_size = qs.size();
        params.max_q_len = *std::max_element(qs.begin(), qs.end());
        params.max_k_len = *std::max_element(ks.begin(), ks.end());
        params.num_heads = H;
        params.num_kv_heads = KH;
        params.size_per_head = D;
        params.cu_q_len = q_offsets.data;
        params.cu_k_len = k_offsets.data;
        params.finished = finished.data;
        params.window_size = window;
        params.inv_sqrt_dh = std::log2(std::exp(1.f)) / std::sqrt(float(D));
        params.linear_iter_params = {kv.data, 2*(cache_tokens+64)*D, (cache_tokens+64)*D};
        params.partial_O = partial_o.data;
        params.partial_ML = partial_ml.data;
        params.split_cnt = split_counts.data + Guard;
        params.offset_q = offset;
        params.arch = 86;
        params.stream = stream;
    }

    ~Case() { cudaStreamDestroy(stream); }

    void Launch(int splits, bool baseline = false)
    {
        // 固定 workspace 容量；超出两路分片预算时覆盖原来的单分片回退行为。
        params.max_split_k = tokens <= WS/2 ? std::min(splits, WS/tokens) : 1;
        params.out = baseline ? reference.data : output.data;
        // 仅在测试中以arch元数据80强制走原64行回退，实际仍在同一3090执行。
        // 该字段在attention实现中只影响外层选路，不改变矩阵数值；候选保持86以覆盖短query32行路径。
        params.arch = baseline ? 80 : 86;
        turbomind::dispatchAttention(params);
        params.arch = 86;
        Check(cudaGetLastError());
    }

    double AbsoluteTolerance() const
    {
        // 概率和输出均使用低精度：绝对误差随 V 的幅度缩放，阈值取半个 dtype epsilon * max|V|。
        // 默认幅度 0.5 保持原 0.002 下限；再加上 2% 的输出相对误差项，避免强输入时误判。
        const double unit = std::is_same_v<T, nv_bfloat16> ? 1.0/256 : 1.0/2048;
        return std::max(.002, unit*value_scale);
    }

    double OracleValue(size_t index)
    {
        // 对失败的单个输出元素独立计算 FP64 参考，定位是否仅为 BF16 归约顺序误差。
        const int row = index / (H*D), head = (index / D) % H, dim = index % D;
        size_t b = 0;
        while (row >= q_offsets.data[b+1]) ++b;
        const int qi = row-q_offsets.data[b], kh = head/(H/KH);
        const int end = klens[b]-qlens[b]+qi+1, begin = std::max(0,end-window);
        const int seq = k_offsets.data[b]-k_offsets.data[0];
        std::vector<double> scores(end-begin);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int k=begin; k<end; ++k) {
            double score = 0.;
            for (int d=0; d<D; ++d)
                score += double(float(q.data[(row*H+head)*D+d])) * float(kv.data[(kh*2*(cache_tokens+64)+seq+k)*D+d]);
            maximum = std::max(maximum,scores[k-begin]=score/std::sqrt(double(D)));
        }
        double numerator=0.,denominator=0.;
        for (int k=begin; k<end; ++k) {
            const double probability=std::exp(scores[k-begin]-maximum);
            numerator += probability*float(kv.data[(kh*2*(cache_tokens+64)+(cache_tokens+64)+seq+k)*D+dim]);
            denominator += probability;
        }
        return numerator/denominator;
    }

    void Verify()
    {
        // 检查所有输出、前方 decode 行，以及 split_cnt 首尾的保护区域。
        Check(cudaStreamSynchronize(stream));
        for (int i = 0; i < offset*H*D; ++i)
            if (float(output.data[i]) != 5.f) throw std::runtime_error("decode prefix overwritten");
        for (size_t i = offset*H*D; i < output.size; ++i) {
            float actual = float(output.data[i]), expected = float(reference.data[i]);
            if (!std::isfinite(actual)) throw std::runtime_error("non-finite split output");
            if (std::abs(actual-expected) > 2*AbsoluteTolerance() + .02f*std::abs(expected)) {
                // BF16 尖峰分布下未分片结果也有舍入误差；超出对照阈值时用同阈值的 FP64 oracle 裁决。
                const double oracle = OracleValue(i);
                if (std::abs(actual-oracle) > AbsoluteTolerance() + .02*std::abs(oracle))
                    throw std::runtime_error("split output fails FP64 oracle: actual=" + std::to_string(actual)
                        + " unsplit=" + std::to_string(expected) + " index=" + std::to_string(i)
                        + " q=" + std::to_string(params.max_q_len) + " kv=" + std::to_string(params.max_k_len)
                        + " splits=" + std::to_string(params.max_split_k) + " oracle=" + std::to_string(oracle));
            }
        }
        for (int i = 0; i < Guard; ++i)
            if (split_counts.data[i] != -99 || split_counts.data[Guard+tokens+i] != -99)
                throw std::runtime_error("split count guard overwritten");
    }

    void Oracle()
    {
        // 独立 FP64 softmax oracle 检查每条序列首尾 query，覆盖 GQA 和滑动窗口。
        for (size_t b = 0; b < qlens.size(); ++b) for (int qi : {0, qlens[b]-1}) for (int h : {0, 5, 11}) {
            const int end = klens[b] - qlens[b] + qi + 1;
            const int begin = std::max(0, end-window);
            const int seq = k_offsets.data[b] - k_offsets.data[0];
            const int kh = h / (H/KH), row = q_offsets.data[b]+qi;
            std::vector<double> scores(end-begin), expected(D, 0.);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int k = begin; k < end; ++k) {
                double score = 0.;
                for (int d = 0; d < D; ++d)
                    score += double(float(q.data[(row*H+h)*D+d])) * float(kv.data[(kh*2*(cache_tokens+64)+seq+k)*D+d]);
                maximum = std::max(maximum, scores[k-begin] = score/std::sqrt(double(D)));
            }
            double denominator = 0.;
            for (int k = begin; k < end; ++k) {
                const double probability = std::exp(scores[k-begin]-maximum);
                denominator += probability;
                for (int d = 0; d < D; ++d)
                    expected[d] += probability * float(kv.data[(kh*2*(cache_tokens+64)+(cache_tokens+64)+seq+k)*D+d]);
            }
            for (int d = 0; d < D; ++d)
                if (std::abs(float(output.data[(row*H+h)*D+d])-expected[d]/denominator) > AbsoluteTolerance() + .02*std::abs(expected[d]/denominator))
                    throw std::runtime_error("CPU attention oracle mismatch");
        }
    }
};

template<class T> void RunSuite(const char* dtype)
{
    // 各形状均检查多个分片上限；2049-token 案例验证 workspace 预算回退。
    const std::vector<std::vector<int>> queries{{1},{2},{31},{64},{65},{1,32,64},{64,128},{2049}};
    const std::vector<std::vector<int>> contexts{{4096},{12288},{12288},{12288},{4096},{64,4096,12288},{128,4096},{4096}};
    // 扩展输入幅度，避免只测试近似均匀 softmax 而漏掉尖峰注意力分布。
    for (float amplitude : {.5f, 2.f, 5.f}) for (size_t i = 0; i < queries.size(); ++i) {
        Case<T> c(queries[i], contexts[i], i%2 ? 7 : 0, i==6 ? 128 : 1<<28, amplitude);
        c.Launch(1, true);
        Check(cudaStreamSynchronize(c.stream));
        try {
            for (int splits : {4,8,16,128}) { c.Launch(splits); c.Verify(); c.Oracle(); }
        } catch (const std::exception& error) {
            // 失败时保留 dtype、幅度和形状，区分精度差异与索引错误。
            throw std::runtime_error(std::string(dtype) + " amplitude=" + std::to_string(amplitude) + " " + error.what());
        }
    }
    // 两个 stream 同时计算不同阶段，独立 workspace 不能互相污染。
    Case<T> prefill({31},{12288},7), decode({1},{4096},0);
    prefill.Launch(1,true); decode.Launch(1,true); Check(cudaDeviceSynchronize());
    for (int repeat = 0; repeat < 8; ++repeat) {
        prefill.Launch(128); decode.Launch(128); Check(cudaDeviceSynchronize());
        prefill.Verify(); decode.Verify();
    }
    std::cout << dtype << ": 96 shape/split checks + 8 concurrent checks passed\n";
}

void PrintKernelResources()
{
    // wrapper继承的是线性64行内核属性；实际短query/分页资源由分派日志单独报告。
    // 这里仅展示registry基类元数据，不能作为实际分页资源或硬件实测occupancy证据。
    turbomind::attention::AttnDesc desc{};
    desc.mode = turbomind::attention::AttnDesc::kPrefill;
    desc.head_dim = 128;
    desc.data_type = turbomind::data_type_v<nv_bfloat16>;
    desc.query_group_sz = 6;
    desc.causal = true;
    const auto* kernel = turbomind::attention::Registry::instance().Find(desc);
    if (!kernel) throw std::runtime_error("BF16 prefill kernel not found");
    const auto& info = kernel->info();
    std::cout << "registry_base_linear_prefill warps=" << info.num_warps
              << " registers_per_thread=" << info.attr.numRegs
              << " max_active_ctas_per_sm=" << info.max_active_ctas
              << " dynamic_smem=" << info.dynamic_smem_size << '\n';
}

void Benchmark()
{
    // 按32/64行分派边界覆盖不同query长度；CUDA event仅作算子证据。
    for (int qlen : {15, 31, 32, 33, 47, 63, 64, 65}) {
        Case<nv_bfloat16> c({qlen},{12288},0);
        cudaEvent_t begin, end;
        Check(cudaEventCreate(&begin)); Check(cudaEventCreate(&end));
        for (int i = 0; i < 5; ++i) c.Launch(8);
        Check(cudaEventRecord(begin,c.stream));
        for (int i = 0; i < 50; ++i) c.Launch(8);
        Check(cudaEventRecord(end,c.stream)); Check(cudaEventSynchronize(end));
        float ms{}; Check(cudaEventElapsedTime(&ms,begin,end));
        std::cout << "BF16 Q" << qlen << " KV12288 split_limit=8 mean_us=" << ms*1000/50 << '\n';
        cudaEventDestroy(begin); cudaEventDestroy(end);
    }
}
} // namespace

int main(int argc, char** argv)
{
    // 任一数值、越界或 CUDA 错误都返回非零，便于 CMake/CI 直接判定失败。
    try {
        PrintKernelResources();
        RunSuite<half>("FP16"); RunSuite<nv_bfloat16>("BF16");
        if (argc>1 && std::string(argv[1])=="--benchmark") Benchmark();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
