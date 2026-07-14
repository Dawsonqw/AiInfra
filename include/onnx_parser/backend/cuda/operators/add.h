#pragma once

#include "onnx_parser/backend/cuda/cuda_operator.h"


namespace aiinfra::onnx::cuda::operators{
class AddOperator final:public Operator{
    public:
    void run(const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs,
                    cudaStream_t stream) const override;

    OpKind type() const noexcept override{
        return OpKind::Add;
    }
};
}