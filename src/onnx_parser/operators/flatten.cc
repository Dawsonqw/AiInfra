#include "onnx_parser/operators/flatten.h"

#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

Dimension FlattenOperator::product(const std::vector<Dimension>& shape, int64_t begin, int64_t end) {
    int64_t value = 1;
    for (int64_t index = begin; index < end; ++index) {
        if (!shape[index].has_value) return {};
        value *= shape[index].value;
    }
    return {true, value, {}};
}

std::vector<TensorInfo> FlattenOperator::infer_shape(const OperatorContext& context) const {
    const auto& input = require_input(context, 0);
    int64_t axis = context.int_attribute("axis", 1);
    if (axis < 0) axis += static_cast<int64_t>(input.shape.size());
    if (axis < 0 || axis > static_cast<int64_t>(input.shape.size())) throw std::runtime_error("Flatten axis is invalid");
    return {make_output(context, 0, {product(input.shape, 0, axis), product(input.shape, axis, input.shape.size())}, input.elem_type)};
}

}  // namespace aiinfra::onnx::operators
