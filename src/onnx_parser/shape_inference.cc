#include "onnx_parser/shape_inference.h"

#include <stdexcept>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "onnx_parser/operator_registry.h"

namespace aiinfra::onnx {
namespace {

void merge_tensor(std::unordered_map<std::string, TensorInfo>* tensors, const TensorInfo& tensor) {
    if (tensor.name.empty()) return;
    auto [found, inserted] = tensors->try_emplace(tensor.name, tensor);
    if (!inserted && tensor.shape_inferred && !found->second.shape_inferred) found->second = tensor;
}

void collect_tensors(const GraphInfo& graph, std::unordered_map<std::string, TensorInfo>* tensors) {
    for (const auto& tensor : graph.tensors) merge_tensor(tensors, tensor);
    for (const auto& tensor : graph.inputs) merge_tensor(tensors, tensor);
    for (const auto& tensor : graph.initializers) merge_tensor(tensors, tensor);
    for (const auto& tensor : graph.value_infos) merge_tensor(tensors, tensor);
    for (const auto& tensor : graph.outputs) merge_tensor(tensors, tensor);
}

void update_graph_views(GraphInfo* graph, const std::unordered_map<std::string, TensorInfo>& tensors) {
    graph->tensors.clear();
    graph->tensors.reserve(tensors.size());
    for (const auto& [name, tensor] : tensors) graph->tensors.push_back(tensor);
    auto update = [&tensors](std::vector<TensorInfo>* values) {
        for (auto& value : *values) {
            const auto found = tensors.find(value.name);
            if (found != tensors.end()) value = found->second;
        }
    };
    update(&graph->inputs);
    update(&graph->outputs);
    update(&graph->value_infos);
    update(&graph->initializers);
}

}  // namespace

void ShapeInference::infer(GraphInfo& graph) const {
    auto& registry = OperatorRegistry::global();
    if (!registry.contains("Identity")) register_builtin_operators(registry);

    std::unordered_map<std::string, TensorInfo> tensors;
    collect_tensors(graph, &tensors);
    for (const auto node_index : graph.topological_order) {
        const auto& node = graph.nodes.at(static_cast<std::size_t>(node_index));
        if (!registry.contains(node.op_type, node.domain)) {
            if (allow_unknown_operators_) {
                spdlog::warn("shape inference skipped unregistered operator {}", node.op_type);
                continue;
            }
            throw std::runtime_error("shape inference requires registered operator: " + node.op_type);
        }

        std::vector<const TensorInfo*> inputs;
        inputs.reserve(node.inputs.size());
        for (const auto& input_name : node.inputs) {
            if (input_name.empty()) {
                inputs.push_back(nullptr);
                continue;
            }
            const auto found = tensors.find(input_name);
            if (found == tensors.end()) throw std::runtime_error("shape inference input not found: " + input_name);
            inputs.push_back(&found->second);
        }
        const auto operation = registry.create(node);
        const auto inferred = operation->infer_shape(OperatorContext(node, std::move(inputs), node.outputs.size()));
        if (inferred.size() != node.outputs.size()) {
            throw std::runtime_error("operator returned an incorrect output count: " + node.op_type);
        }
        for (const auto& tensor : inferred) tensors[tensor.name] = tensor;
    }
    update_graph_views(&graph, tensors);
}

}  // namespace aiinfra::onnx
