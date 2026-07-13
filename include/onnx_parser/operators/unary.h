#pragma once

#include "onnx_parser/op_kind.h"
#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class UnaryOperator : public Operator {
public:
    explicit UnaryOperator(OpKind type) : type_(type) {}

    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    OpKind type() const noexcept override { return type_; }

private:
    OpKind type_ = OpKind::Unknown;
};

}  // namespace aiinfra::onnx::operators
