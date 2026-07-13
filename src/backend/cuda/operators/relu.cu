#include "onnx_parser/backend/cuda/operators/relu.h"

#include <stdexcept>

#include "onnx_parser/backend/cuda/cuda_utils.h"

namespace aiinfra::onnx::cuda::operators {
namespace {

__global__ void relu_kernel(const float* input, float* output, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] > 0.0F ? input[index] : 0.0F;
}

}  // namespace

void ReluOperator::run(const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs,
                       cudaStream_t stream) const {
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0]->desc().dtype != DataType::Float32 ||
        outputs[0]->desc().dtype != DataType::Float32) {
        throw std::invalid_argument("CUDA Relu currently supports one float32 tensor");
    }
    const auto count = inputs[0]->desc().numel();
    relu_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(inputs[0]->data()), static_cast<float*>(outputs[0]->data()), count);
    AIINFRA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace aiinfra::onnx::cuda::operators
