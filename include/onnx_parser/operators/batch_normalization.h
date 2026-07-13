#pragma once

#include "onnx_parser/operators/unary.h"

namespace aiinfra::onnx::operators {

class BatchNormalizationOperator final : public UnaryOperator {
public:
    BatchNormalizationOperator();
};

}  // namespace aiinfra::onnx::operators
