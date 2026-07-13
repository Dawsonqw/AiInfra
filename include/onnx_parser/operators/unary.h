#pragma once

#include <string>
#include <utility>

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class UnaryOperator : public Operator {
public:
    explicit UnaryOperator(std::string type) : type_(std::move(type)) {}

    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    const char* type() const noexcept override { return type_.c_str(); }

private:
    std::string type_;
};

}  // namespace aiinfra::onnx::operators
