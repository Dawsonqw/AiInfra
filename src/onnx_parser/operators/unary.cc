#include "onnx_parser/operators/unary.h"

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

std::vector<TensorInfo> UnaryOperator::infer_shape(const OperatorContext& context) const {
    const auto& input = require_input(context, 0);
    return {make_output(context, 0, input.shape, input.dtype)};
}

}  // namespace aiinfra::onnx::operators
