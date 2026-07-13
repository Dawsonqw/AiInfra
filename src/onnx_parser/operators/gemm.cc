#include "onnx_parser/operators/gemm.h"

#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

std::vector<TensorInfo> GemmOperator::infer_shape(const OperatorContext& context) const {
    const auto& lhs = require_input(context, 0);
    const auto& rhs = require_input(context, 1);
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2) throw std::runtime_error("Gemm expects 2-D tensors");
    const bool trans_a = context.int_attribute("transA", 0) != 0;
    const bool trans_b = context.int_attribute("transB", 0) != 0;
    const Dimension rows = trans_a ? lhs.shape[1] : lhs.shape[0];
    const Dimension columns = trans_b ? rhs.shape[0] : rhs.shape[1];
    return {make_output(context, 0, {rows, columns}, lhs.dtype)};
}

}  // namespace aiinfra::onnx::operators
