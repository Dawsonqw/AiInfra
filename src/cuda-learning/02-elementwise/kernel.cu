// ============================================================
// 第 02 章：Elementwise 操作与 float4 向量化
//
// 学习目标：
//   1. float4 向量化（LDG.128）：一条指令读 4 个 float，
//      减少内存事务次数，提升带宽利用率
//   2. Memory coalescing：相邻线程访问相邻地址，
//      使硬件能合并成 128-byte 的宽事务（L2/DRAM 对齐）
//   3. "改操作不改框架"：把加法换成加法+ReLU，
//      框架代码几乎不变，只需改计算逻辑
//
// 三个 kernel：
//   elementwise_add_f32      — 标量版，baseline，1 线程 1 元素
//   elementwise_add_f32x4    — float4 向量化，1 线程 4 元素
//   elementwise_add_relu_f32 — 标量加法 + ReLU，演示"换操作"
// ============================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ---- 错误检查宏 ----
// 任何 CUDA API 调用失败立即打印位置并退出，避免静默错误
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- float4 辅助宏 ----
// 写入版：把 float* 重解释为 float4*（用于输出或可变数组）
#define FLOAT4(x)  (reinterpret_cast<float4       *>(&(x))[0])
// 只读版：把 const float* 重解释为 const float4*（用于输入数组）
#define CFLOAT4(x) (reinterpret_cast<const float4 *>(&(x))[0])

// ============================================================
// Kernel 1: 标量版（baseline）
//   grid  = (N + 255) / 256
//   block = 256
//   每个线程处理 1 个 float
// ============================================================
__global__ void elementwise_add_f32(const float *A, const float *B,
                                     float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// Kernel 2: float4 向量化版
//   grid  = (N/4 + 255) / 256   ← 线程数减少为 N/4
//   block = 256
//   每个线程处理 4 个连续 float（一次 LDG.128 / STG.128）
//
//   为什么快？
//   - 标量版：N 次 4-byte 加载 → N 个内存事务（最坏情况）
//   - 向量版：N/4 次 16-byte 加载 → 带宽等价但事务数减少，
//     减轻了地址计算与调度开销；同时对 L1/L2 cache line 更友好
//
//   tail 处理：
//   N 不一定是 4 的倍数，最后不足 4 个元素的尾巴
//   用标量循环处理（通常只有 0~3 次，开销可忽略）
// ============================================================
__global__ void elementwise_add_f32x4(const float *A, const float *B,
                                       float *C, int N) {
    // 以 4 个元素为步长编号线程
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (i + 3 < N) {
        // ---- 主路径：4 个元素全在边界内，使用向量化访问 ----
        // FLOAT4 宏把 float* 变成 float4*，一次读 128 bits
        float4 a4 = CFLOAT4(A[i]);
        float4 b4 = CFLOAT4(B[i]);
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        FLOAT4(C[i]) = c4;
    } else {
        // ---- 尾部处理：逐元素写，防止越界 ----
        for (int j = i; j < N; j++) {
            C[j] = A[j] + B[j];
        }
    }
}

// ============================================================
// Kernel 3: 加法 + ReLU（在 f32 框架上换操作）
//   ReLU(x) = max(0, x)
//   用 fmaxf(0.f, x) 实现，编译器会生成一条 FMAX 指令
//
//   注意：框架与 elementwise_add_f32 完全相同，
//   唯一的变化是把 "A+B" 换成 "max(0, A+B)"
// ============================================================
__global__ void elementwise_add_relu_f32(const float *A, const float *B,
                                          float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = fmaxf(0.f, A[i] + B[i]);
    }
}

// ============================================================
// 计时辅助：用 CUDA Events 精确测量 kernel 时间
//   warmup    次先跑，让 GPU 进入稳定状态（避免首次 JIT/缓存冷启动）
//   bench_run 次正式计时，取平均
// ============================================================
static float bench_ms(void (*launch)(const float *, const float *, float *, int,
                                     int, int),
                       const float *d_A, const float *d_B, float *d_C, int N,
                       int blocks, int threads,
                       int warmup = 5, int bench_run = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++)
        launch(d_A, d_B, d_C, N, blocks, threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_run; i++)
        launch(d_A, d_B, d_C, N, blocks, threads);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / bench_run;
}

// ---- launch 包装（统一签名给 bench_ms 使用）----
static void launch_f32(const float *A, const float *B, float *C, int N,
                        int blocks, int threads) {
    elementwise_add_f32<<<blocks, threads>>>(A, B, C, N);
}

static void launch_f32x4(const float *A, const float *B, float *C, int N,
                          int blocks, int threads) {
    elementwise_add_f32x4<<<blocks, threads>>>(A, B, C, N);
}

static void launch_relu_f32(const float *A, const float *B, float *C, int N,
                              int blocks, int threads) {
    elementwise_add_relu_f32<<<blocks, threads>>>(A, B, C, N);
}

// ============================================================
// main
// ============================================================
int main() {
    const int N = 1 << 24;  // 16M 个 float ≈ 64 MB × 3 arrays ≈ 192 MB
    const int THREADS = 256;
    size_t bytes = (size_t)N * sizeof(float);

    // ---- CPU 端分配 + 初始化 ----
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        // 随机初始化到 [-1, 1]，部分值为负（方便测试 ReLU）
        h_A[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
        h_B[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
    }

    // ---- GPU 端分配 ----
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ==== 测试 kernel 1: elementwise_add_f32 ====
    {
        int blocks = (N + THREADS - 1) / THREADS;
        float t = bench_ms(launch_f32, d_A, d_B, d_C, N, blocks, THREADS);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        // CPU 参考
        float max_err = 0.f;
        for (int i = 0; i < N; i++) {
            h_ref[i] = h_A[i] + h_B[i];
            float e = fabsf(h_C[i] - h_ref[i]);
            if (e > max_err) max_err = e;
        }
        printf("elementwise_add_f32     : time=%.3fms  %s\n",
               t, max_err < 1e-6f ? "PASS" : "FAIL");
    }

    // ==== 测试 kernel 2: elementwise_add_f32x4 ====
    {
        // 线程数按 N/4 计算，grid 向上取整
        int blocks4 = (N / 4 + THREADS - 1) / THREADS;
        float t = bench_ms(launch_f32x4, d_A, d_B, d_C, N, blocks4, THREADS);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        float max_err = 0.f;
        for (int i = 0; i < N; i++) {
            float expected = h_A[i] + h_B[i];
            float e = fabsf(h_C[i] - expected);
            if (e > max_err) max_err = e;
        }
        printf("elementwise_add_f32x4   : time=%.3fms  %s\n",
               t, max_err < 1e-6f ? "PASS" : "FAIL");
    }

    // ==== 测试 kernel 3: elementwise_add_relu_f32 ====
    {
        int blocks = (N + THREADS - 1) / THREADS;
        float t = bench_ms(launch_relu_f32, d_A, d_B, d_C, N, blocks, THREADS);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        float max_err = 0.f;
        for (int i = 0; i < N; i++) {
            float expected = fmaxf(0.f, h_A[i] + h_B[i]);
            float e = fabsf(h_C[i] - expected);
            if (e > max_err) max_err = e;
        }
        printf("elementwise_add_relu_f32: time=%.3fms  %s\n",
               t, max_err < 1e-6f ? "PASS" : "FAIL");
    }

    // ---- 释放 ----
    free(h_A); free(h_B); free(h_C); free(h_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
