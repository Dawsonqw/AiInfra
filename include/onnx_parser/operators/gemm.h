#pragma once

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class GemmOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    const char* type() const noexcept override { return "Gemm"; }
};

}  // namespace aiinfra::onnx::operators
