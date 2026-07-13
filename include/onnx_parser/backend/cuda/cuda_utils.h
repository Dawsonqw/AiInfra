#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace aiinfra::onnx::cuda {

inline void check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
}

}  // namespace aiinfra::onnx::cuda

#define AIINFRA_CUDA_CHECK(expression) \
    ::aiinfra::onnx::cuda::check((expression), #expression, __FILE__, __LINE__)
