#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "onnx_parser/onnx_parser.h"

namespace aiinfra::onnx {

class OperatorContext {
public:
    OperatorContext(const NodeInfo& node, std::vector<const TensorInfo*> inputs,
                    std::size_t output_count)
        : node_(node), inputs_(std::move(inputs)), output_count_(output_count) {}

    const NodeInfo& node() const noexcept { return node_; }
    const std::vector<const TensorInfo*>& inputs() const noexcept { return inputs_; }
    std::size_t output_count() const noexcept { return output_count_; }

    const AttributeInfo* attribute(const std::string& name) const noexcept;
    int64_t int_attribute(const std::string& name, int64_t fallback) const;
    std::vector<int64_t> ints_attribute(const std::string& name,
                                        std::vector<int64_t> fallback) const;

private:
    const NodeInfo& node_;
    std::vector<const TensorInfo*> inputs_;
    std::size_t output_count_ = 0;
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual std::vector<TensorInfo> infer_shape(const OperatorContext& context) const = 0;
    virtual const char* type() const noexcept = 0;
};

}  // namespace aiinfra::onnx
