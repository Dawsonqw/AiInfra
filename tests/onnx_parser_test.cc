#include "onnx_parser/onnx_parser.h"

#include <cstdio>
#include <fstream>

#include <gtest/gtest.h>

#include "onnx/onnx.pb.h"
#include "onnx_parser/operator_registry.h"

namespace {

TEST(OnnxParserTest, BuildsTopologicalOrderAndInfersShapes) {
    const std::string path = "onnx_parser_test.onnx";
    ::onnx::ModelProto model;
    model.set_ir_version(8);
    auto* opset = model.add_opset_import();
    opset->set_version(18);
    auto* graph = model.mutable_graph();
    graph->set_name("test_graph");
    auto* input = graph->add_input();
    input->set_name("x");
    input->mutable_type()->mutable_tensor_type()->set_elem_type(::onnx::TensorProto::FLOAT);
    input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto* downstream = graph->add_node();
    downstream->set_name("downstream");
    downstream->set_op_type("Identity");
    downstream->add_input("y");
    downstream->add_output("z");
    auto* upstream = graph->add_node();
    upstream->set_name("upstream");
    upstream->set_op_type("Identity");
    upstream->add_input("x");
    upstream->add_output("y");
    graph->add_output()->set_name("z");

    bool serialized = false;
    {
        std::ofstream file(path, std::ios::binary);
        serialized = model.SerializeToOstream(&file);
    }
    ASSERT_TRUE(serialized);

    const auto parsed = aiinfra::onnx::OnnxParser().parse_file(path);
    EXPECT_EQ(parsed.graph.nodes.size(), 2U);
    EXPECT_EQ(parsed.graph.topological_order, std::vector<int32_t>({1, 0}));
    ASSERT_FALSE(parsed.graph.outputs.empty());
    ASSERT_EQ(parsed.graph.outputs.front().shape.size(), 1U);
    EXPECT_EQ(parsed.graph.outputs.front().shape.front().value, 1);
    std::remove(path.c_str());
}

TEST(OperatorRegistryTest, CreatesRegisteredBuiltinOperator) {
    aiinfra::onnx::OperatorRegistry registry;
    aiinfra::onnx::register_builtin_operators(registry);
    EXPECT_TRUE(registry.contains("Conv"));
    EXPECT_TRUE(registry.contains("Relu"));
    aiinfra::onnx::NodeInfo node;
    node.op_type = "Relu";
    EXPECT_NE(registry.create(node), nullptr);
}

}  // namespace
