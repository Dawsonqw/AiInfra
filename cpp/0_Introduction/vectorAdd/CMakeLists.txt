add_executable(vectorAdd vec_add_0.cu)

target_compile_options(vectorAdd PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>: 
        -Xptxas=-v # 编译时展示静态信息
        -Xptxas=-warn-spills # 检测是否溢出到local memory
    >
)