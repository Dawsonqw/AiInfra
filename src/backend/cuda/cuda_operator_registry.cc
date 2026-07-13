#include "onnx_parser/backend/cuda/cuda_operator_registry.h"

#include <stdexcept>
#include <utility>

#include "onnx_parser/backend/cuda/operators/identity.h"
#include "onnx_parser/backend/cuda/operators/relu.h"

namespace aiinfra::onnx::cuda {

void OperatorRegistry::register_operator(OpKind kind, Creator creator) {
    if (kind == OpKind::Unknown || !creator) throw std::invalid_argument("invalid CUDA operator registration");
    if (!creators_.emplace(kind, std::move(creator)).second) {
        throw std::invalid_argument("CUDA operator already registered: " + op_kind_name(kind));
    }
}

std::unique_ptr<Operator> OperatorRegistry::create(const NodeInfo& node) const {
    const auto found = creators_.find(node.kind);
    if (found == creators_.end()) {
        const auto name = node.source_op_type.empty() ? op_kind_name(node.kind) : node.source_op_type;
        throw std::runtime_error("CUDA operator is not registered: " + name);
    }
    return found->second(node);
}

bool OperatorRegistry::contains(OpKind kind) const noexcept {
    return creators_.find(kind) != creators_.end();
}

OperatorRegistry& OperatorRegistry::global() {
    static OperatorRegistry registry;
    return registry;
}

void register_builtin_operators(OperatorRegistry& registry) {
    registry.register_operator(OpKind::Identity, [](const NodeInfo&) {
        return std::make_unique<operators::IdentityOperator>();
    });
    registry.register_operator(OpKind::Relu, [](const NodeInfo&) {
        return std::make_unique<operators::ReluOperator>();
    });
}

}  // namespace aiinfra::onnx::cuda
