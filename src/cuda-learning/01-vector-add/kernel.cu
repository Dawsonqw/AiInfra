#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================
// GPU Kernel: 每个线程处理一个元素
// __global__ 表示这是 GPU 函数，由 CPU 调用，在 GPU 上执行
// ============================================================
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // 计算"我是第几个线程"——这是 CUDA 最重要的公式
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止越界（线程总数可能大于 N）
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// CPU 入口
// ============================================================
int main() {
    int N = 256;                               // 数据量
    size_t bytes = N * sizeof(float);

    // ---- 1. CPU 端分配内存并初始化 ----
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;                    // A = [0, 1, 2, 3, ...]
        h_B[i] = i * 2.0f;                    // B = [0, 2, 4, 6, ...]
    }

    // ---- 2. GPU 端分配显存 ----
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);                  // 在 GPU 上申请内存
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // ---- 3. 把数据从 CPU 拷贝到 GPU ----
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ---- 4. 启动 Kernel ----
    int threads = 256;                        // 每个 block 256 个线程
    int blocks = (N + threads - 1) / threads; // 向上取整：需要几个 block？
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // ---- 5. 把 GPU 结果拷回 CPU ----
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // ---- 6. 验证 ----
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = i * 1.0f + i * 2.0f; // A[i] + B[i]
        if (fabs(h_C[i] - expected) > 1e-6) {
            printf("Mismatch at [%d]: got %f, expected %f\n", i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("PASS: all %d elements correct\n", N);
    else
        printf("FAIL: %d errors\n", errors);

    // ---- 7. 释放内存 ----
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
