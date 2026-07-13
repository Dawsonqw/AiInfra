#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "onnx_parser/backend/cuda/cuda_tensor.h"
#include "onnx_parser/backend/cuda/cuda_utils.h"
#include "onnx_parser/backend/cuda/operators/relu.h"

namespace {

class CudaStream {
public:
    CudaStream() { AIINFRA_CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    operator cudaStream_t() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEventPair {
public:
    CudaEventPair() {
        AIINFRA_CUDA_CHECK(cudaEventCreate(&start_));
        AIINFRA_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEventPair() {
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }

    CudaEventPair(const CudaEventPair&) = delete;
    CudaEventPair& operator=(const CudaEventPair&) = delete;

    float measure(const std::function<void()>& operation, cudaStream_t stream) {
        AIINFRA_CUDA_CHECK(cudaEventRecord(start_, stream));
        operation();
        AIINFRA_CUDA_CHECK(cudaEventRecord(stop_, stream));
        AIINFRA_CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        AIINFRA_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

class ReluFixture {
public:
    static constexpr std::size_t kElements = 1U << 20U;
    static constexpr int kWarmupIterations = 10;

    ReluFixture()
        : input({"input", aiinfra::onnx::DataType::Float32,
                 {static_cast<int64_t>(kElements)}}),
          output({"output", aiinfra::onnx::DataType::Float32,
                  {static_cast<int64_t>(kElements)}}),
          host_input(kElements), host_output(kElements, 0.0F) {
        std::mt19937 generator(20260713U);
        std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
        for (auto& value : host_input) value = distribution(generator);
        warmup();
    }

    std::size_t bytes() const { return kElements * sizeof(float); }
    cudaStream_t stream() const { return stream_; }

    float measure_h2d() {
        return events.measure([&] {
            input.copy_from_host(host_input.data(), bytes(), stream_);
        }, stream_);
    }

    float measure_kernel() {
        return events.measure([&] {
            operation.run({&input}, {&output}, stream_);
        }, stream_);
    }

    float measure_d2h() {
        return events.measure([&] {
            output.copy_to_host(host_output.data(), bytes(), stream_);
        }, stream_);
    }

private:
    void warmup() {
        for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
            input.copy_from_host(host_input.data(), bytes(), stream_);
            operation.run({&input}, {&output}, stream_);
            output.copy_to_host(host_output.data(), bytes(), stream_);
        }
        AIINFRA_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    CudaStream stream_;
    aiinfra::onnx::cuda::Tensor input;
    aiinfra::onnx::cuda::Tensor output;
    aiinfra::onnx::cuda::operators::ReluOperator operation;
    CudaEventPair events;
    std::vector<float> host_input;
    std::vector<float> host_output;
};

void benchmark_relu(benchmark::State& state) {
    ReluFixture fixture;
    double h2d_total = 0.0;
    double kernel_total = 0.0;
    double d2h_total = 0.0;
    for (auto _ : state) {
        const auto h2d = fixture.measure_h2d();
        const auto kernel = fixture.measure_kernel();
        const auto d2h = fixture.measure_d2h();
        h2d_total += h2d;
        kernel_total += kernel;
        d2h_total += d2h;
        state.SetIterationTime((h2d + kernel + d2h) / 1000.0);
    }

    const auto iterations = static_cast<double>(state.iterations());
    const auto bytes = static_cast<double>(fixture.bytes());
    state.counters["H2D_ms"] = h2d_total / iterations;
    state.counters["H2D_GBps"] = bytes / ((h2d_total / iterations) * 1.0e6);
    state.counters["Kernel_ms"] = kernel_total / iterations;
    state.counters["Kernel_GBps"] = 2.0 * bytes / ((kernel_total / iterations) * 1.0e6);
    state.counters["D2H_ms"] = d2h_total / iterations;
    state.counters["D2H_GBps"] = bytes / ((d2h_total / iterations) * 1.0e6);
}

}  // namespace

BENCHMARK(benchmark_relu)->UseManualTime();
BENCHMARK_MAIN();
