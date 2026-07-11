add_executable(vectorAdd0 vec_add_0.cu)
target_compile_options(vectorAdd0 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>: 
        -Xptxas=-v # 编译时展示静态信息
        -Xptxas=-warn-spills # 检测是否溢出到local memory
    >
)

add_executable(vectorAdd1 vec_add_1.cu)
target_compile_options(vectorAdd1 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>: 
        -Xptxas=-v # 编译时展示静态信息
        -Xptxas=-warn-spills # 检测是否溢出到local memory
    >
)


