#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "onnx_parser/onnx_parser.h"

namespace aiinfra::onnx::cuda {

struct TensorDesc {
    std::string name;
    DataType dtype = DataType::Unknown;
    std::vector<int64_t> shape;

    std::size_t numel() const;
    std::size_t bytes() const;
};

class Tensor {
public:
    explicit Tensor(TensorDesc desc);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const TensorDesc& desc() const noexcept { return desc_; }
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    void copy_from_host(const void* source, std::size_t bytes, cudaStream_t stream);
    void copy_to_host(void* destination, std::size_t bytes, cudaStream_t stream) const;

private:
    void release() noexcept;

    TensorDesc desc_;
    void* data_ = nullptr;
};

}  // namespace aiinfra::onnx::cuda
