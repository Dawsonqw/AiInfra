#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_parser/operator.h"

namespace aiinfra::onnx::operators {

using Shape = std::vector<Dimension>;

TensorInfo make_output(const OperatorContext& context, std::size_t index, Shape shape,
                       int32_t elem_type);
const TensorInfo& require_input(const OperatorContext& context, std::size_t index);
std::vector<int64_t> spatial_attribute(const OperatorContext& context,
                                       const std::string& name,
                                       std::vector<int64_t> fallback);
std::vector<int64_t> padding_attribute(const OperatorContext& context);
Dimension convolution_dimension(const Dimension& input, int64_t kernel, int64_t stride,
                                int64_t pad_before, int64_t pad_after, int64_t dilation);

}  // namespace aiinfra::onnx::operators
