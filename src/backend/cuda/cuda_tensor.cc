#include "onnx_parser/backend/cuda/cuda_tensor.h"

#include <limits>
#include <stdexcept>

#include "onnx_parser/backend/cuda/cuda_utils.h"

namespace aiinfra::onnx::cuda {

std::size_t TensorDesc::numel() const {
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("dynamic CUDA tensor shape is unsupported");
        if (dimension != 0 && result > std::numeric_limits<std::size_t>::max() /
                                      static_cast<std::size_t>(dimension)) {
            throw std::overflow_error("CUDA tensor element count overflow");
        }
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

std::size_t TensorDesc::bytes() const {
    return numel() * data_type_size(dtype);
}

Tensor::Tensor(TensorDesc desc) : desc_(std::move(desc)) {
    if (desc_.dtype == DataType::Unknown) throw std::invalid_argument("CUDA tensor has unknown dtype");
    AIINFRA_CUDA_CHECK(cudaMalloc(&data_, desc_.bytes()));
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : desc_(std::move(other.desc_)), data_(other.data_) {
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    release();
    desc_ = std::move(other.desc_);
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
}

void Tensor::copy_from_host(const void* source, std::size_t bytes, cudaStream_t stream) {
    if (bytes != desc_.bytes()) throw std::invalid_argument("input byte size does not match tensor");
    AIINFRA_CUDA_CHECK(cudaMemcpyAsync(data_, source, bytes, cudaMemcpyHostToDevice, stream));
}

void Tensor::copy_to_host(void* destination, std::size_t bytes, cudaStream_t stream) const {
    if (bytes != desc_.bytes()) throw std::invalid_argument("output byte size does not match tensor");
    AIINFRA_CUDA_CHECK(cudaMemcpyAsync(destination, data_, bytes, cudaMemcpyDeviceToHost, stream));
}

void Tensor::release() noexcept {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
}

}  // namespace aiinfra::onnx::cuda
