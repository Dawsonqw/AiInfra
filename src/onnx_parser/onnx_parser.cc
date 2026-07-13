#include "onnx_parser/onnx_parser.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "onnx/onnx.pb.h"
#include "onnx_parser/onnx_op_mapping.h"
#include "onnx_parser/shape_inference.h"

namespace aiinfra::onnx {
namespace {

DataType onnx_data_type(int32_t type) {
    switch (type) {
    case ::onnx::TensorProto::FLOAT: return DataType::Float32;
    case ::onnx::TensorProto::FLOAT16: return DataType::Float16;
    case ::onnx::TensorProto::INT32: return DataType::Int32;
    case ::onnx::TensorProto::INT64: return DataType::Int64;
    default: return DataType::Unknown;
    }
}

Dimension parse_dimension(const ::onnx::TensorShapeProto_Dimension& dim) {
    Dimension result;
    if (dim.has_dim_value()) {
        result.has_value = true;
        result.value = dim.dim_value();
    } else if (dim.has_dim_param()) {
        result.parameter = dim.dim_param();
    }
    return result;
}

TensorInfo parse_value_info(const ::onnx::ValueInfoProto& value) {
    TensorInfo result;
    result.name = value.name();
    if (!value.has_type() || !value.type().has_tensor_type()) {
        return result;
    }

    const auto& tensor_type = value.type().tensor_type();
    result.elem_type = tensor_type.elem_type();
    result.dtype = onnx_data_type(result.elem_type);
    if (tensor_type.has_shape()) {
        for (const auto& dim : tensor_type.shape().dim()) {
            result.shape.push_back(parse_dimension(dim));
        }
        result.shape_inferred = true;
    }
    return result;
}

TensorInfo parse_initializer(const ::onnx::TensorProto& initializer) {
    TensorInfo result;
    result.name = initializer.name();
    result.elem_type = initializer.data_type();
    result.dtype = onnx_data_type(result.elem_type);
    result.is_initializer = true;
    for (const auto dim : initializer.dims()) {
        Dimension parsed;
        parsed.has_value = true;
        parsed.value = dim;
        result.shape.push_back(parsed);
    }
    result.initializer_bytes = initializer.has_raw_data()
        ? static_cast<int64_t>(initializer.raw_data().size())
        : static_cast<int64_t>(initializer.ByteSizeLong());
    return result;
}

int attribute_value_count(const ::onnx::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ::onnx::AttributeProto::FLOATS: return attribute.floats_size();
    case ::onnx::AttributeProto::INTS: return attribute.ints_size();
    case ::onnx::AttributeProto::STRINGS: return attribute.strings_size();
    case ::onnx::AttributeProto::TENSORS: return attribute.tensors_size();
    case ::onnx::AttributeProto::GRAPHS: return attribute.graphs_size();
    case ::onnx::AttributeProto::TYPE_PROTOS: return attribute.type_protos_size();
    default: return 1;
    }
}

AttributeValue parse_attribute_value(const ::onnx::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ::onnx::AttributeProto::INT: return attribute.i();
    case ::onnx::AttributeProto::FLOAT: return attribute.f();
    case ::onnx::AttributeProto::STRING: return attribute.s();
    case ::onnx::AttributeProto::INTS:
        return std::vector<int64_t>(attribute.ints().begin(), attribute.ints().end());
    case ::onnx::AttributeProto::FLOATS:
        return std::vector<float>(attribute.floats().begin(), attribute.floats().end());
    case ::onnx::AttributeProto::STRINGS:
        return std::vector<std::string>(attribute.strings().begin(), attribute.strings().end());
    default: return int64_t{0};
    }
}

NodeInfo parse_node(const ::onnx::NodeProto& node, int32_t index) {
    NodeInfo result;
    result.index = index;
    result.name = node.name();
    result.kind = onnx_op_kind(node.op_type(), node.domain());
    result.source_op_type = node.op_type();
    result.domain = node.domain();
    result.inputs.assign(node.input().begin(), node.input().end());
    result.outputs.assign(node.output().begin(), node.output().end());
    for (const auto& attribute : node.attribute()) {
        result.attributes.push_back({attribute.name(), attribute.type(),
                                     attribute_value_count(attribute),
                                     parse_attribute_value(attribute)});
    }
    return result;
}

void validate_and_order(GraphInfo* graph) {
    std::unordered_set<std::string> available;
    for (const auto& input : graph->inputs) {
        if (!available.insert(input.name).second) {
            throw std::runtime_error("duplicate graph input: " + input.name);
        }
    }
    std::unordered_set<std::string> initializer_names;
    for (const auto& initializer : graph->initializers) {
        if (!initializer_names.insert(initializer.name).second) {
            throw std::runtime_error("duplicate initializer: " + initializer.name);
        }
        // ONNX IR permits an initializer to also appear as a graph input.
        available.insert(initializer.name);
    }

    std::unordered_map<std::string, int32_t> producers;
    for (const auto& node : graph->nodes) {
        for (const auto& output : node.outputs) {
            if (output.empty()) continue;
            if (!producers.emplace(output, node.index).second) {
                throw std::runtime_error("tensor has multiple producers: " + output);
            }
        }
    }

    std::vector<int32_t> indegree(graph->nodes.size(), 0);
    std::vector<std::vector<int32_t>> consumers(graph->nodes.size());
    for (const auto& node : graph->nodes) {
        std::unordered_set<int32_t> dependencies;
        for (const auto& input : node.inputs) {
            if (input.empty()) continue;  // Optional ONNX input.
            const auto producer = producers.find(input);
            if (producer == producers.end()) {
                if (available.find(input) == available.end()) {
                    throw std::runtime_error("node '" + node.name +
                                             "' references unknown tensor: " + input);
                }
                continue;
            }
            if (producer->second != node.index && dependencies.insert(producer->second).second) {
                ++indegree[node.index];
                consumers[producer->second].push_back(node.index);
            }
        }
    }

    std::queue<int32_t> ready;
    for (int32_t index = 0; index < static_cast<int32_t>(graph->nodes.size()); ++index) {
        if (indegree[index] == 0) ready.push(index);
    }
    while (!ready.empty()) {
        const int32_t current = ready.front();
        ready.pop();
        graph->topological_order.push_back(current);
        for (const auto consumer : consumers[current]) {
            if (--indegree[consumer] == 0) ready.push(consumer);
        }
    }
    if (graph->topological_order.size() != graph->nodes.size()) {
        throw std::runtime_error("graph contains a cycle");
    }
}

}  // namespace

ModelInfo OnnxParser::parse_file(const std::string& path, bool infer_shapes) const {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open ONNX model: " + path);

    ::onnx::ModelProto model;
    if (!model.ParseFromIstream(&input)) {
        throw std::runtime_error("failed to parse ONNX protobuf: " + path);
    }
    if (!model.has_graph()) throw std::runtime_error("ONNX model has no graph: " + path);

    ModelInfo result;
    result.ir_version = model.ir_version();
    result.producer_name = model.producer_name();
    result.producer_version = model.producer_version();
    for (const auto& opset : model.opset_import()) {
        const auto domain = opset.domain().empty() ? "ai.onnx" : opset.domain();
        result.opsets.push_back(domain + ":" + std::to_string(opset.version()));
    }
    const auto& graph = model.graph();
    result.graph.name = graph.name();
    for (const auto& value : graph.input()) result.graph.inputs.push_back(parse_value_info(value));
    for (const auto& value : graph.output()) result.graph.outputs.push_back(parse_value_info(value));
    for (const auto& value : graph.value_info()) result.graph.value_infos.push_back(parse_value_info(value));
    for (const auto& initializer : graph.initializer()) {
        result.graph.initializers.push_back(parse_initializer(initializer));
    }
    for (int32_t index = 0; index < graph.node_size(); ++index) {
        result.graph.nodes.push_back(parse_node(graph.node(index), index));
    }
    result.graph.tensors = result.graph.inputs;
    result.graph.tensors.insert(result.graph.tensors.end(), result.graph.initializers.begin(),
                                result.graph.initializers.end());
    result.graph.tensors.insert(result.graph.tensors.end(), result.graph.value_infos.begin(),
                                result.graph.value_infos.end());
    result.graph.tensors.insert(result.graph.tensors.end(), result.graph.outputs.begin(),
                                result.graph.outputs.end());
    validate_and_order(&result.graph);
    if (infer_shapes) ShapeInference().infer(result.graph);
    return result;
}

}  // namespace aiinfra::onnx
