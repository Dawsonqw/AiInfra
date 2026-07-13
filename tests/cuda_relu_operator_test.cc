#include <algorithm>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_parser/backend/cuda/cuda_utils.h"
#include "onnx_parser/backend/cuda/cuda_tensor.h"
#include "onnx_parser/backend/cuda/operators/relu.h"

namespace {

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

struct BenchmarkStats {
    double milliseconds = 0.0;
    double gigabytes_per_second = 0.0;
};

template <typename Operation>
BenchmarkStats benchmark(const std::string& name, std::size_t bytes, int iterations,
                         cudaStream_t stream, Operation&& operation) {
    CudaEventPair events;
    double total_milliseconds = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        total_milliseconds += events.measure(operation, stream);
    }
    const auto average_milliseconds = total_milliseconds / iterations;
    const auto gigabytes_per_second =
        static_cast<double>(bytes) / (average_milliseconds * 1.0e6);
    std::cout << std::left << std::setw(8) << name
              << " avg_ms=" << std::fixed << std::setprecision(4) << average_milliseconds
              << " bandwidth_GB_s=" << std::setprecision(2) << gigabytes_per_second
              << " iterations=" << iterations << '\n';
    return {average_milliseconds, gigabytes_per_second};
}

}  // namespace

TEST(CudaReluOperatorTest, ComputesCorrectlyAndReportsBandwidth) {
    constexpr std::size_t kElements = 1U << 20U;
    constexpr int kWarmupIterations = 10;
    constexpr int kBenchmarkIterations = 100;
    const std::size_t bytes = kElements * sizeof(float);

    CudaStream stream;

    aiinfra::onnx::cuda::Tensor input({
        "input", aiinfra::onnx::DataType::Float32, {static_cast<int64_t>(kElements)}});
    aiinfra::onnx::cuda::Tensor output({
        "output", aiinfra::onnx::DataType::Float32, {static_cast<int64_t>(kElements)}});
    aiinfra::onnx::cuda::operators::ReluOperator operation;

    std::vector<float> host_input(kElements);
    std::vector<float> host_output(kElements, 0.0F);
    std::mt19937 generator(20260713U);
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    for (auto& value : host_input) value = distribution(generator);

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        input.copy_from_host(host_input.data(), bytes, stream);
        operation.run({&input}, {&output}, stream);
        output.copy_to_host(host_output.data(), bytes, stream);
    }
    AIINFRA_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "CUDA ReLU single-operator benchmark: elements=" << kElements
              << " bytes=" << bytes << '\n';
    const auto h2d = benchmark("H2D", bytes, kBenchmarkIterations, stream, [&] {
        input.copy_from_host(host_input.data(), bytes, stream);
    });
    const auto kernel = benchmark("Kernel", 2U * bytes, kBenchmarkIterations, stream, [&] {
        operation.run({&input}, {&output}, stream);
    });
    const auto d2h = benchmark("D2H", bytes, kBenchmarkIterations, stream, [&] {
        output.copy_to_host(host_output.data(), bytes, stream);
    });
    AIINFRA_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t index = 0; index < kElements; ++index) {
        EXPECT_FLOAT_EQ(host_output[index], std::max(host_input[index], 0.0F));
    }
    EXPECT_GT(h2d.milliseconds, 0.0);
    EXPECT_GT(kernel.milliseconds, 0.0);
    EXPECT_GT(d2h.milliseconds, 0.0);
    EXPECT_GT(h2d.gigabytes_per_second, 0.0);
    EXPECT_GT(kernel.gigabytes_per_second, 0.0);
    EXPECT_GT(d2h.gigabytes_per_second, 0.0);

}
