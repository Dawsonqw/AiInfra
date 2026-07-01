/*
 * SGEMM: C = A @ B (FP32), Naive vs Tiled (shared memory)
 *
 * === Naive GEMM 内存访问瓶颈 ===
 * 每个输出元素 C[row][col] 需要：
 *   - 从全局内存读取 A 的第 row 行：K 次读取
 *   - 从全局内存读取 B 的第 col 列：K 次读取
 * 共计 2K 次全局内存访问（延迟 ~400-800 cycles 每次）
 * 整个 C 矩阵（M×N 个元素）总计 2*M*N*K 次全局内存访问。
 * Global memory bandwidth 是主要瓶颈，ALU 大量空转。
 *
 * === Tiled GEMM 优化原理 ===
 * 将 A、B 分块（tile size: BM×BK 和 BK×BN），每次把一块加载到
 * shared memory（延迟 ~5 cycles），块内所有线程复用这块数据。
 *
 * 具体分析（以 BK=32 为例）：
 *   - 一个 block（BM×BN 个线程）负责 C 的 BM×BN 子矩阵
 *   - 外循环共 K/BK 轮，每轮只从全局内存读取：
 *       sA: BM×BK 个元素，sB: BK×BN 个元素
 *   - 这 BM×BN 个线程各自读自己负责的元素（1 次全局访问）
 *     然后对 sA/sB 共享数据做 BK 次乘加（从 shared memory 读取）
 *
 * 全局访问量：原来每线程 2K 次，现在每线程 2K/BK 次（降低 BK 倍）
 * 当 BK=32，理论上全局访问减少 32x，带宽利用率大幅提升。
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BM 32
#define BN 32
#define BK 32

// -------------------------------------------------------------------------
// Kernel 1: Naive GEMM
// grid(ceil(N/16), ceil(M/16)), block(16, 16)
// thread (tx, ty) 负责计算 C[row][col]
// -------------------------------------------------------------------------
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Tiled GEMM with shared memory
// grid(ceil(N/BN), ceil(M/BM)), block(BN, BM)
// 每个 thread (tx, ty) 负责 C[row][col] 一个元素
// -------------------------------------------------------------------------
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    // 每个 block 负责的 C 子矩阵左上角 (row_start, col_start)
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    // shared memory tiles：sA[BM][BK] 存 A 的一个列块，sB[BK][BN] 存 B 的一个行块
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float sum = 0.f;

    // 外循环：沿 K 维度按步长 BK 迭代
    for (int bk = 0; bk < K; bk += BK) {
        // --- 步骤 1：协作加载 sA 和 sB ---
        // 每个线程负责加载自己对应的一个元素（1 次全局读取）
        // sA[ty][tx] <- A[row][bk + tx]
        if (row < M && (bk + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.f;

        // sB[ty][tx] <- B[bk + ty][col]
        if ((bk + threadIdx.y) < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.f;

        // --- 步骤 2：第一个 __syncthreads()，确保加载完成再计算 ---
        // 作用：保证 block 内所有线程都完成了 sA/sB 的写入，
        //       没有线程在其他线程还没写完就开始读 shared memory。
        __syncthreads();

        // --- 步骤 3：tile 内累加（全从 shared memory 读，~5 cycles）---
        for (int k = 0; k < BK; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        // --- 步骤 4：第二个 __syncthreads()，确保计算完成再加载下一块 ---
        // 作用：保证 block 内所有线程都完成了对 sA/sB 的读取，
        //       没有线程在下一轮覆盖 shared memory 时，其他线程还在用旧数据。
        __syncthreads();
    }

    // 写回结果（边界保护）
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// -------------------------------------------------------------------------
// CPU 参考实现（3 层循环，仅用于正确性验证）
// M=K=N=512 时约 20~80ms，只运行一次
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

// -------------------------------------------------------------------------
// 辅助：验证 GPU 结果与 CPU 结果的最大误差
// -------------------------------------------------------------------------
float max_error(const float* ref, const float* out, int n) {
    float err = 0.f;
    for (int i = 0; i < n; i++)
        err = fmaxf(err, fabsf(ref[i] - out[i]));
    return err;
}

int main() {
    const int M = 512, K = 512, N = 512;
    const float threshold = 1e-3f;

    // --- 分配 Host 内存 ---
    float* hA  = (float*)malloc(M * K * sizeof(float));
    float* hB  = (float*)malloc(K * N * sizeof(float));
    float* hC_ref = (float*)malloc(M * N * sizeof(float));
    float* hC_gpu = (float*)malloc(M * N * sizeof(float));

    // --- 随机初始化 A, B（值域 [-1, 1]）---
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (int i = 0; i < K * N; i++) hB[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;

    // --- CPU 参考（只跑一次）---
    sgemm_cpu(hA, hB, hC_ref, M, K, N);

    // --- 分配 Device 内存 ---
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // --- CUDA Event 计时工具 ---
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    float ms_naive, ms_tiled;

    // =====================================================================
    // Kernel 1: sgemm_naive
    // =====================================================================
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
        printf("sgemm_naive : time=%.3fms  %s  (max_err=%.2e)\n",
               ms_naive, err < threshold ? "PASS" : "FAIL", err);
    }

    // =====================================================================
    // Kernel 2: sgemm_tiled
    // =====================================================================
    {
        // block(BN, BM) => threadIdx.x 对应列，threadIdx.y 对应行
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
        printf("sgemm_tiled : time=%.3fms  %s  (max_err=%.2e)\n",
               ms_tiled, err < threshold ? "PASS" : "FAIL", err);
    }

    printf("speedup: %.1fx\n", ms_naive / ms_tiled);

    // --- 清理 ---
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_ref); free(hC_gpu);
    return 0;
}
