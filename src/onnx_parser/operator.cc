#include "onnx_parser/operator.h"

#include <utility>

namespace aiinfra::onnx {

const AttributeInfo* OperatorContext::attribute(const std::string& name) const noexcept {
    for (const auto& candidate : node_.attributes) {
        if (candidate.name == name) return &candidate;
    }
    return nullptr;
}

int64_t OperatorContext::int_attribute(const std::string& name, int64_t fallback) const {
    const auto* value = attribute(name);
    if (value == nullptr) return fallback;
    if (const auto* scalar = std::get_if<int64_t>(&value->value)) return *scalar;
    return fallback;
}

std::vector<int64_t> OperatorContext::ints_attribute(const std::string& name,
                                                     std::vector<int64_t> fallback) const {
    const auto* value = attribute(name);
    if (value == nullptr) return fallback;
    if (const auto* values = std::get_if<std::vector<int64_t>>(&value->value)) return *values;
    return fallback;
}

}  // namespace aiinfra::onnx
