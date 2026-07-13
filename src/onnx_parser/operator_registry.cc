#include "onnx_parser/operator_registry.h"

#include <stdexcept>

namespace aiinfra::onnx {

void OperatorRegistry::register_operator(OpKind op_type, Creator creator) {
    if (op_type == OpKind::Unknown || !creator) {
        throw std::invalid_argument("operator registration is invalid");
    }
    if (!creators_.emplace(op_type, std::move(creator)).second) {
        throw std::invalid_argument("operator already registered: " + op_kind_name(op_type));
    }
}

std::unique_ptr<Operator> OperatorRegistry::create(const NodeInfo& node) const {
    const auto found = creators_.find(node.kind);
    if (found == creators_.end()) {
        const auto source_name = node.source_op_type.empty()
            ? op_kind_name(node.kind)
            : node.source_op_type;
        throw std::runtime_error("operator is not registered: " + source_name);
    }
    return found->second();
}

bool OperatorRegistry::contains(OpKind op_type) const noexcept {
    return creators_.find(op_type) != creators_.end();
}

OperatorRegistry& OperatorRegistry::global() {
    static OperatorRegistry registry;
    return registry;
}

}  // namespace aiinfra::onnx
