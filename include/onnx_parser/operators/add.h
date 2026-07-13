#pragma once

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class AddOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    const char* type() const noexcept override { return "Add"; }
};

}  // namespace aiinfra::onnx::operators
