#include "onnx_parser/operators/conv.h"

#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {

std::vector<TensorInfo> ConvOperator::infer_shape(const OperatorContext& context) const {
    const auto& input = require_input(context, 0);
    const auto& weight = require_input(context, 1);
    if (input.shape.size() != 4 || weight.shape.size() != 4) throw std::runtime_error("Conv expects 4-D tensors");
    const auto kernel = spatial_attribute(context, "kernel_shape", {weight.shape[2].value, weight.shape[3].value});
    const auto strides = spatial_attribute(context, "strides", {1, 1});
    const auto dilations = spatial_attribute(context, "dilations", {1, 1});
    const auto pads = padding_attribute(context);
    Shape shape = {input.shape[0], weight.shape[0],
                   convolution_dimension(input.shape[2], kernel[0], strides[0], pads[0], pads[2], dilations[0]),
                   convolution_dimension(input.shape[3], kernel[1], strides[1], pads[1], pads[3], dilations[1])};
    return {make_output(context, 0, std::move(shape), input.elem_type)};
}

}  // namespace aiinfra::onnx::operators
