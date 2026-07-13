#pragma once

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class ConvOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    OpKind type() const noexcept override { return OpKind::Conv; }
};

}  // namespace aiinfra::onnx::operators
