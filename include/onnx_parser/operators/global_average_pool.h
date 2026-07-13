#pragma once

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class GlobalAveragePoolOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    const char* type() const noexcept override { return "GlobalAveragePool"; }
};

}  // namespace aiinfra::onnx::operators
