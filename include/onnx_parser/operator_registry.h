#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "onnx_parser/operator.h"

namespace aiinfra::onnx {

class OperatorRegistry {
public:
    using Creator = std::function<std::unique_ptr<Operator>()>;

    void register_operator(OpKind op_type, Creator creator);
    std::unique_ptr<Operator> create(const NodeInfo& node) const;
    bool contains(OpKind op_type) const noexcept;
    std::size_t size() const noexcept { return creators_.size(); }

    static OperatorRegistry& global();

private:
    struct OpKindHash {
        std::size_t operator()(OpKind kind) const noexcept {
            return static_cast<std::size_t>(kind);
        }
    };
    std::unordered_map<OpKind, Creator, OpKindHash> creators_;
};

void register_builtin_operators(OperatorRegistry& registry = OperatorRegistry::global());

}  // namespace aiinfra::onnx
