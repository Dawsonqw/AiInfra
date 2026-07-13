#include "onnx_parser/operators/global_average_pool.h"

#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

std::vector<TensorInfo> GlobalAveragePoolOperator::infer_shape(const OperatorContext& context) const {
    const auto& input = require_input(context, 0);
    if (input.shape.size() != 4) throw std::runtime_error("GlobalAveragePool expects a 4-D tensor");
    auto shape = input.shape;
    shape[2] = {true, 1, {}};
    shape[3] = {true, 1, {}};
    return {make_output(context, 0, std::move(shape), input.elem_type)};
}

}  // namespace aiinfra::onnx::operators
