#include "onnx_parser/operators/max_pool.h"

#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

std::vector<TensorInfo> MaxPoolOperator::infer_shape(const OperatorContext& context) const {
    const auto& input = require_input(context, 0);
    if (input.shape.size() != 4) throw std::runtime_error("MaxPool expects a 4-D tensor");
    const auto kernel = spatial_attribute(context, "kernel_shape", {1, 1});
    const auto strides = spatial_attribute(context, "strides", {1, 1});
    const auto pads = padding_attribute(context);
    Shape shape = input.shape;
    shape[2] = convolution_dimension(input.shape[2], kernel[0], strides[0], pads[0], pads[2], 1);
    shape[3] = convolution_dimension(input.shape[3], kernel[1], strides[1], pads[1], pads[3], 1);
    return {make_output(context, 0, std::move(shape), input.elem_type)};
}

}  // namespace aiinfra::onnx::operators
