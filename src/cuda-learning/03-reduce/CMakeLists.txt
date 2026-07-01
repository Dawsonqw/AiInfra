add_executable(reduce my_kernel.cu)
set_target_properties(reduce PROPERTIES CUDA_ARCHITECTURES 89)
