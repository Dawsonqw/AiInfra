#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "onnx_parser/onnx_parser.h"
#include "onnx_parser/backend/cuda/cuda_tensor.h"

namespace aiinfra::onnx::cuda {

class Operator {
public:
    virtual ~Operator() = default;
    virtual void run(const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs,
                     cudaStream_t stream) const = 0;
    virtual OpKind type() const noexcept = 0;
};

}  // namespace aiinfra::onnx::cuda
