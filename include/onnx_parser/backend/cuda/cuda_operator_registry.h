#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "onnx_parser/backend/cuda/cuda_operator.h"

namespace aiinfra::onnx::cuda {

class OperatorRegistry {
public:
    using Creator = std::function<std::unique_ptr<Operator>(const NodeInfo&)>;

    void register_operator(OpKind kind, Creator creator);
    std::unique_ptr<Operator> create(const NodeInfo& node) const;
    bool contains(OpKind kind) const noexcept;

    static OperatorRegistry& global();

private:
    struct Hash {
        std::size_t operator()(OpKind kind) const noexcept {
            return static_cast<std::size_t>(kind);
        }
    };
    std::unordered_map<OpKind, Creator, Hash> creators_;
};

void register_builtin_operators(OperatorRegistry& registry = OperatorRegistry::global());

}  // namespace aiinfra::onnx::cuda
