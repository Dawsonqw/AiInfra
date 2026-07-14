#include "onnx_parser/backend/cuda/operators/add.h"
#include "onnx_parser/backend/cuda/cuda_utils.h"

#include <stdexcept>
#include <spdlog/spdlog.h>

namespace aiinfra::onnx::cuda::operators{
namespace {
    void __global__ add_kernel(const float* left,const float* right,
                            float* result,std::size_t count){
        int idx=blockDim.x*blockIdx.x+threadIdx.x;
        if(idx<count){
            result[idx]=left[idx]+right[idx];
        }
    }
}

    void AddOperator::run(const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs,
                    cudaStream_t stream) const{
        if(inputs.size()!=2||outputs.size()!=1){
            throw std::invalid_argument("CUDA Add Tensor Error");
        }
        std::size_t count=inputs[0]->desc().numel();
        std::size_t grid_size=(count+255)/255;
        std::size_t block_size=256;
        add_kernel<<<grid_size,block_size,0,stream>>>(static_cast<float*>(inputs[0]->data()),
                                    static_cast<const float*>(inputs[1]->data()),
                                    static_cast<float*>(outputs[0]->data()),count);
        AIINFRA_CUDA_CHECK(cudaGetLastError());
    }
}