#include "onnx_parser/operators/add.h"

#include <algorithm>
#include <stdexcept>

#include "onnx_parser/operators/common.h"

namespace aiinfra::onnx::operators {
namespace {

Dimension broadcast_dimension(const Dimension* lhs, const Dimension* rhs) {
    if (lhs == nullptr) return *rhs;
    if (rhs == nullptr) return *lhs;
    if (lhs->has_value && lhs->value == 1) return *rhs;
    if (rhs->has_value && rhs->value == 1) return *lhs;
    if (lhs->has_value && rhs->has_value && lhs->value != rhs->value) {
        throw std::runtime_error("Add shape broadcast mismatch");
    }
    return lhs->has_value || !lhs->parameter.empty() ? *lhs : *rhs;
}

}  // namespace

std::vector<TensorInfo> AddOperator::infer_shape(const OperatorContext& context) const {
    const auto& lhs = require_input(context, 0);
    const auto& rhs = require_input(context, 1);
    const auto rank = std::max(lhs.shape.size(), rhs.shape.size());
    Shape shape(rank);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto lhs_index = lhs.shape.size() > offset ? lhs.shape.size() - 1 - offset : lhs.shape.size();
        const auto rhs_index = rhs.shape.size() > offset ? rhs.shape.size() - 1 - offset : rhs.shape.size();
        const Dimension* left = lhs_index < lhs.shape.size() ? &lhs.shape[lhs_index] : nullptr;
        const Dimension* right = rhs_index < rhs.shape.size() ? &rhs.shape[rhs_index] : nullptr;
        shape[rank - 1 - offset] = broadcast_dimension(left, right);
    }
    return {make_output(context, 0, std::move(shape), lhs.elem_type)};
}

}  // namespace aiinfra::onnx::operators
