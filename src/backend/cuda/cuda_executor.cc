#include "onnx_parser/backend/cuda/cuda_executor.h"

#include <stdexcept>
#include <utility>

#include "onnx_parser/backend/cuda/cuda_operator_registry.h"
#include "onnx_parser/backend/cuda/cuda_utils.h"

namespace aiinfra::onnx::cuda {

Executor::Executor(const OperatorRegistry& registry) : registry_(registry) {
    AIINFRA_CUDA_CHECK(cudaStreamCreate(&stream_));
}

Executor::~Executor() {
    tensors_.clear();
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

Tensor* Executor::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::runtime_error("CUDA tensor is not found: " + name);
    return found->second.get();
}

void Executor::compile(const GraphInfo& graph) {
    tensors_.clear();
    steps_.clear();
    for (const auto& info : graph.tensors) {
        if (info.name.empty()) continue;
        std::vector<int64_t> shape;
        shape.reserve(info.shape.size());
        for (const auto& dimension : info.shape) {
            if (!dimension.has_value) throw std::runtime_error("dynamic CUDA tensor shape: " + info.name);
            shape.push_back(dimension.value);
        }
        if (!tensors_.emplace(info.name, std::make_unique<Tensor>(TensorDesc{info.name, info.dtype, shape})).second) {
            throw std::runtime_error("duplicate CUDA tensor: " + info.name);
        }
    }
    for (const auto node_index : graph.topological_order) {
        const auto& node = graph.nodes.at(static_cast<std::size_t>(node_index));
        Step step;
        step.operation = registry_.create(node);
        step.inputs = node.inputs;
        step.outputs = node.outputs;
        for (const auto& name : step.inputs) if (!name.empty()) tensor(name);
        for (const auto& name : step.outputs) if (!name.empty()) tensor(name);
        steps_.push_back(std::move(step));
    }
}

void Executor::copy_tensor(const std::string& name, const void* data, std::size_t bytes) {
    tensor(name)->copy_from_host(data, bytes, stream_);
}

void Executor::run() {
    for (const auto& step : steps_) {
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        for (const auto& name : step.inputs) if (!name.empty()) inputs.push_back(tensor(name));
        for (const auto& name : step.outputs) if (!name.empty()) outputs.push_back(tensor(name));
        step.operation->run(inputs, outputs, stream_);
    }
}

void Executor::copy_output(const std::string& name, void* data, std::size_t bytes) {
    tensor(name)->copy_to_host(data, bytes, stream_);
}

void Executor::synchronize() const { AIINFRA_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

}  // namespace aiinfra::onnx::cuda
