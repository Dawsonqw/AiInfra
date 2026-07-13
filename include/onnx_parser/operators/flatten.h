#pragma once

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

class FlattenOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    const char* type() const noexcept override { return "Flatten"; }

private:
    static Dimension product(const std::vector<Dimension>& shape, int64_t begin, int64_t end);
};

}  // namespace aiinfra::onnx::operators
