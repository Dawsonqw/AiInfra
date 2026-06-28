add_executable(vec_add my_kernel.cu)

set_target_properties(vec_add PROPERTIES CUDA_ARCHITECTURES 89)