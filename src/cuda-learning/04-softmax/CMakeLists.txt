add_executable(softmax my_kernel.cu)
set_target_properties(softmax PROPERTIES CUDA_ARCHITECTURES 89)
