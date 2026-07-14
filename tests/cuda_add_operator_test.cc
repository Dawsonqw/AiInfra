#include "onnx_parser/backend/cuda/cuda_tensor.h"
#include "onnx_parser/backend/cuda/cuda_utils.h"
#include "onnx_parser/backend/cuda/operators/add.h"

#include <gtest/gtest.h>
#include <random>

namespace{
    class CudaStream{
        public:
            CudaStream(){
                AIINFRA_CUDA_CHECK(cudaStreamCreate(&_stream));
            }

            ~CudaStream(){
                if(!_stream){
                    AIINFRA_CUDA_CHECK(cudaStreamDestroy(_stream));
                }
            }

            CudaStream(const CudaStream&)=delete;
            CudaStream& operator=(const CudaStream&)=delete;

            operator cudaStream_t(){
                return _stream;
            }
    
        private:
            cudaStream_t _stream=nullptr;;
    };
}

TEST(CudaAddOperatorTest,ComputesCorrectResultWithoutExecutor1D){
    CudaStream cudaStream;

    constexpr int Kelement=1024;
    constexpr size_t elementSize=sizeof(float)*Kelement;

    aiinfra::onnx::cuda::Tensor leftInput(
        {"left",aiinfra::onnx::DataType::Float32,{Kelement}}
    );

    aiinfra::onnx::cuda::Tensor rightInput(
        {"right",aiinfra::onnx::DataType::Float32,{Kelement}}
    );

    aiinfra::onnx::cuda::Tensor resultOutput(
        {"result",aiinfra::onnx::DataType::Float32,{Kelement}}
    );

    aiinfra::onnx::cuda::operators::AddOperator operation;

    std::vector<float> host_input_left(Kelement);
    std::vector<float> host_input_right(Kelement);
    std::vector<float> host_output(Kelement);

    constexpr std::uint32_t seed=20260714u;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float>distribution(0.0f,1.0f);

    for(int idx=0;idx<Kelement;idx++){
        host_input_left[idx]=distribution(generator);
        host_input_right[idx]=distribution(generator);
    }

    leftInput.copy_from_host(host_input_left.data(),elementSize,cudaStream);
    rightInput.copy_from_host(host_input_right.data(),elementSize,cudaStream);

    operation.run(
        {&leftInput,&rightInput},{&resultOutput},cudaStream
    );

    resultOutput.copy_to_host(host_output.data(),elementSize,cudaStream);

    AIINFRA_CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    for(std::size_t idx=0;idx<Kelement;idx++){
        EXPECT_FLOAT_EQ(host_output[idx],host_input_left[idx]+host_input_right[idx]);
    }
}

TEST(CudaAddOperatorTest,ComputesCorrectResultWithoutExecutor4D){

}