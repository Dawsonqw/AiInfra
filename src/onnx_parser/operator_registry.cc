#include "onnx_parser/operator_registry.h"

#include <stdexcept>

namespace aiinfra::onnx {

std::string OperatorRegistry::key(const std::string& op_type, const std::string& domain) {
    return (domain.empty() ? "ai.onnx" : domain) + ":" + op_type;
}

void OperatorRegistry::register_operator(std::string op_type, Creator creator,
                                         std::string domain) {
    if (op_type.empty() || !creator) throw std::invalid_argument("operator registration is invalid");
    const auto operator_key = key(op_type, domain);
    if (!creators_.emplace(operator_key, std::move(creator)).second) {
        throw std::invalid_argument("operator already registered: " + operator_key);
    }
}

std::unique_ptr<Operator> OperatorRegistry::create(const NodeInfo& node) const {
    const auto found = creators_.find(key(node.op_type, node.domain));
    if (found == creators_.end()) {
        throw std::runtime_error("operator is not registered: " + key(node.op_type, node.domain));
    }
    return found->second();
}

bool OperatorRegistry::contains(const std::string& op_type, const std::string& domain) const noexcept {
    return creators_.find(key(op_type, domain)) != creators_.end();
}

OperatorRegistry& OperatorRegistry::global() {
    static OperatorRegistry registry;
    return registry;
}

}  // namespace aiinfra::onnx
