#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32

// TODO: 实现 warp_reduce_sum_f32（device 函数）
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    // Hint: 用 __shfl_xor_sync(0xffffffff, val, mask) 做蝴蝶归约
    return val;
}

// TODO: 实现 block_all_reduce_sum_f32 kernel
__global__ void block_all_reduce_sum_f32(const float* A, float* y, int N) {
    // 你的代码
}

int main() {
    // TODO: 参考 kernel.cu 验证 reduce 结果
    return 0;
}
