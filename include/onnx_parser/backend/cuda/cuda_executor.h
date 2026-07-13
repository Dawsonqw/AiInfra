#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

#include "onnx_parser/backend/cuda/cuda_operator.h"
#include "onnx_parser/backend/cuda/cuda_operator_registry.h"
#include "onnx_parser/backend/cuda/cuda_tensor.h"

namespace aiinfra::onnx::cuda {

class Executor {
public:
    explicit Executor(const OperatorRegistry& registry = OperatorRegistry::global());
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void compile(const GraphInfo& graph);
    void copy_tensor(const std::string& name, const void* data, std::size_t bytes);
    void copy_input(const std::string& name, const void* data, std::size_t bytes) {
        copy_tensor(name, data, bytes);
    }
    void run();
    void copy_output(const std::string& name, void* data, std::size_t bytes);
    void synchronize() const;

private:
    struct Step {
        std::unique_ptr<Operator> operation;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    Tensor* tensor(const std::string& name) const;

    cudaStream_t stream_ = nullptr;
    const OperatorRegistry& registry_;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> tensors_;
    std::vector<Step> steps_;
};

}  // namespace aiinfra::onnx::cuda
