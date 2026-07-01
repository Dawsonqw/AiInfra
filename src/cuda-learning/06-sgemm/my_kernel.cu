/*
 * 练习：从零实现 SGEMM
 *
 * 目标：理解 shared memory tiling 对全局内存访问的优化效果。
 *
 * 运行参考实现看输出：
 *   nvcc -std=c++17 -O2 -arch=sm_89 kernel.cu -o kernel && ./kernel
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BM 32
#define BN 32
#define BK 32

// -------------------------------------------------------------------------
// TODO: sgemm_naive
// 3层循环，每线程负责一个 C[row][col]，直接读全局内存。
// grid(ceil(N/16), ceil(M/16)), block(16, 16)
// -------------------------------------------------------------------------
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    // TODO
}

// -------------------------------------------------------------------------
// TODO: sgemm_tiled
// 使用 shared memory 分块（BM×BK 和 BK×BN），减少全局内存访问。
// 关键点：
//   1. 每线程加载 sA 和 sB 各一个元素
//   2. 第一个 __syncthreads()：确保加载完成再计算
//   3. 累加 BK 次（从 shared memory 读）
//   4. 第二个 __syncthreads()：确保计算完成再加载下一块
// grid(ceil(N/BN), ceil(M/BM)), block(BN, BM)
// -------------------------------------------------------------------------
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    // TODO
}

// -------------------------------------------------------------------------
// CPU 参考实现（用于验证正确性，不需要修改）
// -------------------------------------------------------------------------
void sgemm_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

float max_error(const float* ref, const float* out, int n) {
    float err = 0.f;
    for (int i = 0; i < n; i++)
        err = fmaxf(err, fabsf(ref[i] - out[i]));
    return err;
}

int main() {
    const int M = 512, K = 512, N = 512;
    const float threshold = 1e-3f;

    float* hA     = (float*)malloc(M * K * sizeof(float));
    float* hB     = (float*)malloc(K * N * sizeof(float));
    float* hC_ref = (float*)malloc(M * N * sizeof(float));
    float* hC_gpu = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (int i = 0; i < K * N; i++) hB[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;

    sgemm_cpu(hA, hB, hC_ref, M, K, N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_naive, ms_tiled;

    // sgemm_naive
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        cudaMemset(dC, 0, M * N * sizeof(float));
        cudaEventRecord(t0);
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_naive, t0, t1);
        cudaMemcpy(hC_gpu, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        float err = max_error(hC_ref, hC_gpu, M * N);
        printf("sgemm_naive : time=%.3fms  %s\n",
               ms_naive, err < threshold ? "PASS" : "FAIL");
    }

    // sgemm_tiled
    {
        dim3 block(BN, BM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        cudaMemset(dC, 0, M * N * sizeof(float));
        cudaEventRecord(t0);
        sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_tiled, t0, t1);
        cudaMemcpy(hC_gpu, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        float err = max_error(hC_ref, hC_gpu, M * N);
        printf("sgemm_tiled : time=%.3fms  %s\n",
               ms_tiled, err < threshold ? "PASS" : "FAIL");
    }

    printf("speedup: %.1fx\n", ms_naive / ms_tiled);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_ref); free(hC_gpu);
    return 0;
}
