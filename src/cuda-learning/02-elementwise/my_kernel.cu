#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---- float4 辅助宏（提示：向量化时会用到）----
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// TODO: 实现 elementwise_add_f32 kernel
// 要求：1 个线程处理 1 个元素，超出 N 的线程不做任何操作
__global__ void elementwise_add_f32(const float *A, const float *B,
                                     float *C, int N) {
    // 你的代码
}

// TODO: 实现 elementwise_add_f32x4 kernel（float4 向量化）
// 要求：1 个线程处理 4 个连续元素
//   主路径：i, i+1, i+2, i+3 全在 [0, N) 内时，用 FLOAT4 宏一次读写 4 个
//   尾部处理：i+3 >= N 时，逐元素处理剩余元素（防止越界）
// 提示：grid 大小 = (N/4 + 255) / 256
__global__ void elementwise_add_f32x4(const float *A, const float *B,
                                       float *C, int N) {
    // 你的代码
}

int main() {
    // TODO: 参考 kernel.cu 的 main()，完成以下步骤：
    //   1. 设置 N = 1 << 24，分配 CPU/GPU 内存
    //   2. 随机初始化输入数据
    //   3. 将数据拷贝到 GPU
    //   4. 启动 elementwise_add_f32，拷回结果，与 CPU 参考值比较，打印 PASS/FAIL
    //   5. 启动 elementwise_add_f32x4，同样验证
    //   6. （可选）用 CUDA Events 计时，比较两个 kernel 的速度差异
    //   7. 释放所有内存
    return 0;
}
