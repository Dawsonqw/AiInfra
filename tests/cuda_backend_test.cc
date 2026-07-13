#include <array>
#include <utility>

#include <gtest/gtest.h>

#include "onnx_parser/backend/cuda/cuda_executor.h"

namespace {

aiinfra::onnx::TensorInfo tensor(std::string name) {
    aiinfra::onnx::TensorInfo result;
    result.name = std::move(name);
    result.dtype = aiinfra::onnx::DataType::Float32;
    result.elem_type = 1;
    result.shape = {{true, 4, {}}};
    result.shape_inferred = true;
    return result;
}

TEST(CudaExecutorTest, ExecutesIdentityAndReluGraph) {
    aiinfra::onnx::GraphInfo graph;
    graph.inputs.push_back(tensor("input"));
    graph.outputs.push_back(tensor("output"));
    graph.tensors = {tensor("input"), tensor("hidden"), tensor("output")};

    aiinfra::onnx::NodeInfo identity;
    identity.index = 0;
    identity.kind = aiinfra::onnx::OpKind::Identity;
    identity.inputs = {"input"};
    identity.outputs = {"hidden"};
    aiinfra::onnx::NodeInfo relu;
    relu.index = 1;
    relu.kind = aiinfra::onnx::OpKind::Relu;
    relu.inputs = {"hidden"};
    relu.outputs = {"output"};
    graph.nodes = {identity, relu};
    graph.topological_order = {0, 1};

    aiinfra::onnx::cuda::OperatorRegistry registry;
    aiinfra::onnx::cuda::register_builtin_operators(registry);
    aiinfra::onnx::cuda::Executor executor(registry);
    ASSERT_NO_THROW(executor.compile(graph));
    const std::array<float, 4> input = {-2.0F, -1.0F, 2.0F, 3.0F};
    std::array<float, 4> output{};
    executor.copy_tensor("input", input.data(), sizeof(input));
    executor.run();
    executor.copy_output("output", output.data(), sizeof(output));
    executor.synchronize();

    EXPECT_FLOAT_EQ(output[0], 0.0F);
    EXPECT_FLOAT_EQ(output[1], 0.0F);
    EXPECT_FLOAT_EQ(output[2], 2.0F);
    EXPECT_FLOAT_EQ(output[3], 3.0F);
}

}  // namespace
