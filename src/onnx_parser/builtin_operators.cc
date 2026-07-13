#include "onnx_parser/operator_registry.h"

#include <algorithm>
#include <stdexcept>

namespace aiinfra::onnx {
namespace {

using Shape = std::vector<Dimension>;

TensorInfo output(const OperatorContext& context, std::size_t index, Shape shape,
                 int32_t elem_type) {
    if (index >= context.node().outputs.size()) throw std::runtime_error("operator output index is invalid");
    TensorInfo result;
    result.name = context.node().outputs[index];
    result.elem_type = elem_type;
    result.shape = std::move(shape);
    result.shape_inferred = true;
    return result;
}

const TensorInfo& require_input(const OperatorContext& context, std::size_t index) {
    if (index >= context.inputs().size() || context.inputs()[index] == nullptr) {
        throw std::runtime_error("operator input is missing: " + context.node().op_type);
    }
    return *context.inputs()[index];
}

class UnaryOperator : public Operator {
public:
    explicit UnaryOperator(const char* type) : type_(type) {}
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& input = require_input(context, 0);
        return {output(context, 0, input.shape, input.elem_type)};
    }
    const char* type() const noexcept override { return type_; }

private:
    const char* type_;
};

class AddOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
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
        return {output(context, 0, std::move(shape), lhs.elem_type)};
    }
    const char* type() const noexcept override { return "Add"; }

private:
    static Dimension broadcast_dimension(const Dimension* lhs, const Dimension* rhs) {
        if (lhs == nullptr) return *rhs;
        if (rhs == nullptr) return *lhs;
        if (lhs->has_value && lhs->value == 1) return *rhs;
        if (rhs->has_value && rhs->value == 1) return *lhs;
        if (lhs->has_value && rhs->has_value && lhs->value != rhs->value) {
            throw std::runtime_error("Add shape broadcast mismatch");
        }
        return lhs->has_value || !lhs->parameter.empty() ? *lhs : *rhs;
    }
};

std::vector<int64_t> spatial_attribute(const OperatorContext& context, const std::string& name,
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

class ConvOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& input = require_input(context, 0);
        const auto& weight = require_input(context, 1);
        if (input.shape.size() != 4 || weight.shape.size() != 4) throw std::runtime_error("Conv expects 4-D tensors");
        const auto kernel = spatial_attribute(context, "kernel_shape", {weight.shape[2].value, weight.shape[3].value});
        const auto strides = spatial_attribute(context, "strides", {1, 1});
        const auto dilations = spatial_attribute(context, "dilations", {1, 1});
        const auto pads = padding_attribute(context);
        Shape shape = {input.shape[0], {true, weight.shape[0].value, {}},
                       convolution_dimension(input.shape[2], kernel[0], strides[0], pads[0], pads[2], dilations[0]),
                       convolution_dimension(input.shape[3], kernel[1], strides[1], pads[1], pads[3], dilations[1])};
        return {output(context, 0, std::move(shape), input.elem_type)};
    }
    const char* type() const noexcept override { return "Conv"; }
};

class MaxPoolOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& input = require_input(context, 0);
        if (input.shape.size() != 4) throw std::runtime_error("MaxPool expects a 4-D tensor");
        const auto kernel = spatial_attribute(context, "kernel_shape", {1, 1});
        const auto strides = spatial_attribute(context, "strides", {1, 1});
        const auto pads = padding_attribute(context);
        Shape shape = input.shape;
        shape[2] = convolution_dimension(input.shape[2], kernel[0], strides[0], pads[0], pads[2], 1);
        shape[3] = convolution_dimension(input.shape[3], kernel[1], strides[1], pads[1], pads[3], 1);
        return {output(context, 0, std::move(shape), input.elem_type)};
    }
    const char* type() const noexcept override { return "MaxPool"; }
};

class GlobalAveragePoolOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& input = require_input(context, 0);
        if (input.shape.size() != 4) throw std::runtime_error("GlobalAveragePool expects a 4-D tensor");
        auto shape = input.shape;
        shape[2] = {true, 1, {}};
        shape[3] = {true, 1, {}};
        return {output(context, 0, std::move(shape), input.elem_type)};
    }
    const char* type() const noexcept override { return "GlobalAveragePool"; }
};

class FlattenOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& input = require_input(context, 0);
        int64_t axis = context.int_attribute("axis", 1);
        if (axis < 0) axis += static_cast<int64_t>(input.shape.size());
        if (axis < 0 || axis > static_cast<int64_t>(input.shape.size())) throw std::runtime_error("Flatten axis is invalid");
        return {output(context, 0, {product(input.shape, 0, axis), product(input.shape, axis, input.shape.size())}, input.elem_type)};
    }
    const char* type() const noexcept override { return "Flatten"; }

private:
    static Dimension product(const Shape& shape, int64_t begin, int64_t end) {
        int64_t value = 1;
        for (int64_t index = begin; index < end; ++index) {
            if (!shape[index].has_value) return {};
            value *= shape[index].value;
        }
        return {true, value, {}};
    }
};

class GemmOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override {
        const auto& lhs = require_input(context, 0);
        const auto& rhs = require_input(context, 1);
        if (lhs.shape.size() != 2 || rhs.shape.size() != 2) throw std::runtime_error("Gemm expects 2-D tensors");
        const bool trans_a = context.int_attribute("transA", 0) != 0;
        const bool trans_b = context.int_attribute("transB", 0) != 0;
        const Dimension rows = trans_a ? lhs.shape[1] : lhs.shape[0];
        const Dimension columns = trans_b ? rhs.shape[0] : rhs.shape[1];
        return {output(context, 0, {rows, columns}, lhs.elem_type)};
    }
    const char* type() const noexcept override { return "Gemm"; }
};

class BatchNormalizationOperator final : public UnaryOperator {
public:
    BatchNormalizationOperator() : UnaryOperator("BatchNormalization") {}
};

}  // namespace

void register_builtin_operators(OperatorRegistry& registry) {
    registry.register_operator("Identity", [] { return std::make_unique<UnaryOperator>("Identity"); });
    registry.register_operator("Relu", [] { return std::make_unique<UnaryOperator>("Relu"); });
    registry.register_operator("BatchNormalization", [] { return std::make_unique<BatchNormalizationOperator>(); });
    registry.register_operator("Add", [] { return std::make_unique<AddOperator>(); });
    registry.register_operator("Conv", [] { return std::make_unique<ConvOperator>(); });
    registry.register_operator("MaxPool", [] { return std::make_unique<MaxPoolOperator>(); });
    registry.register_operator("GlobalAveragePool", [] { return std::make_unique<GlobalAveragePoolOperator>(); });
    registry.register_operator("Flatten", [] { return std::make_unique<FlattenOperator>(); });
    registry.register_operator("Gemm", [] { return std::make_unique<GemmOperator>(); });
}

}  // namespace aiinfra::onnx
