#include "onnx_parser/onnx_parser.h"

#include <exception>
#include <iostream>

namespace {
std::string shape_to_string(const aiinfra::onnx::TensorInfo& tensor) {
    std::string result = "[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i != 0) result += ",";
        result += tensor.shape[i].has_value ? std::to_string(tensor.shape[i].value)
                                           : (tensor.shape[i].parameter.empty() ? "?" : tensor.shape[i].parameter);
    }
    return result + "]";
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: onnx_parser_cli <model.onnx>\n";
        return 2;
    }
    try {
        const auto model = aiinfra::onnx::OnnxParser().parse_file(argv[1]);
        const auto& graph = model.graph;
        std::cout << "graph=" << graph.name << " nodes=" << graph.nodes.size()
                  << " initializers=" << graph.initializers.size() << "\n";
        for (const auto& input : graph.inputs) {
            std::cout << "input " << input.name << " dtype=" << input.elem_type
                      << " shape=" << shape_to_string(input) << "\n";
        }
        for (const auto& output : graph.outputs) {
            std::cout << "output " << output.name << " dtype=" << output.elem_type
                      << " shape=" << shape_to_string(output) << "\n";
        }
        for (const auto index : graph.topological_order) {
            const auto& node = graph.nodes[index];
            std::cout << "node[" << index << "] " << aiinfra::onnx::op_kind_name(node.kind)
                      << " name=" << node.name
                      << " inputs=" << node.inputs.size() << " outputs=" << node.outputs.size() << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "onnx_parser: " << error.what() << "\n";
        return 1;
    }
}
