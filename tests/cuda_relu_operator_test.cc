#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

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

}  // namespace

TEST(CudaReluOperatorTest, ComputesCorrectResultWithoutExecutor) {
    constexpr std::size_t kElements = 1024;
    const std::size_t bytes = kElements * sizeof(float);
    CudaStream stream;
    aiinfra::onnx::cuda::Tensor input({
        "input", aiinfra::onnx::DataType::Float32, {static_cast<int64_t>(kElements)}});
    aiinfra::onnx::cuda::Tensor output({
        "output", aiinfra::onnx::DataType::Float32, {static_cast<int64_t>(kElements)}});
    aiinfra::onnx::cuda::operators::ReluOperator operation;

    std::vector<float> host_input(kElements);
    std::vector<float> host_output(kElements, 0.0F);
    for (std::size_t index = 0; index < kElements; ++index) {
        host_input[index] = static_cast<float>(index) - 512.0F;
    }

    input.copy_from_host(host_input.data(), bytes, stream);
    operation.run({&input}, {&output}, stream);
    output.copy_to_host(host_output.data(), bytes, stream);
    AIINFRA_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t index = 0; index < kElements; ++index) {
        EXPECT_FLOAT_EQ(host_output[index], std::max(host_input[index], 0.0F));
    }
}
