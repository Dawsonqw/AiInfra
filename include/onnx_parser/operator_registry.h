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

    void register_operator(std::string op_type, Creator creator,
                           std::string domain = {});
    std::unique_ptr<Operator> create(const NodeInfo& node) const;
    bool contains(const std::string& op_type, const std::string& domain = {}) const noexcept;
    std::size_t size() const noexcept { return creators_.size(); }

    static OperatorRegistry& global();

private:
    static std::string key(const std::string& op_type, const std::string& domain);
    std::unordered_map<std::string, Creator> creators_;
};

void register_builtin_operators(OperatorRegistry& registry = OperatorRegistry::global());

}  // namespace aiinfra::onnx
