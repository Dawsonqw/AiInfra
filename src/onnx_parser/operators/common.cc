#include "onnx_parser/operators/common.h"

#include <stdexcept>
#include <utility>

namespace aiinfra::onnx::operators {

TensorInfo make_output(const OperatorContext& context, std::size_t index, Shape shape,
                       DataType dtype) {
    if (index >= context.node().outputs.size()) {
        throw std::runtime_error("operator output index is invalid");
    }
    TensorInfo result;
    result.name = context.node().outputs[index];
    result.dtype = dtype;
    result.shape = std::move(shape);
    result.shape_inferred = true;
    return result;
}

const TensorInfo& require_input(const OperatorContext& context, std::size_t index) {
    if (index >= context.inputs().size() || context.inputs()[index] == nullptr) {
        throw std::runtime_error("operator input is missing: " +
                                 op_kind_name(context.node().kind));
    }
    return *context.inputs()[index];
}

std::vector<int64_t> spatial_attribute(const OperatorContext& context,
                                       const std::string& name,
                                       std::vector<int64_t> fallback) {
    auto values = context.ints_attribute(name, std::move(fallback));
    if (values.size() == 1) values.resize(2, values.front());
    if (values.size() != 2) throw std::runtime_error("expected two spatial " + name);
    return values;
}

std::vector<int64_t> padding_attribute(const OperatorContext& context) {
    auto values = context.ints_attribute("pads", {0, 0, 0, 0});
    if (values.size() == 2) values = {values[0], values[1], values[0], values[1]};
    if (values.size() != 4) throw std::runtime_error("expected two or four spatial pads");
    return values;
}

Dimension convolution_dimension(const Dimension& input, int64_t kernel, int64_t stride,
                                int64_t pad_before, int64_t pad_after, int64_t dilation) {
    if (!input.has_value) return {};
    const auto numerator = input.value + pad_before + pad_after - dilation * (kernel - 1) - 1;
    return {true, numerator / stride + 1, {}};
}

}  // namespace aiinfra::onnx::operators
