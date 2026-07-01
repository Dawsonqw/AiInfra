add_executable(sgemm my_kernel.cu)
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES 89)
