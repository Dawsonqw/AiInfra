/*
 * Online Safe Softmax — CUDA 参考实现
 * ====================================
 *
 * 三种算法对比
 * -----------
 * 1. Naive Softmax
 *    y_i = exp(x_i) / sum_j(exp(x_j))
 *    问题：x_i 较大时 exp(x_i) 溢出为 INF，导致 NaN。
 *
 * 2. Safe Softmax（数值稳定版）
 *    先求最大值 m = max(x)，再计算 y_i = exp(x_i - m) / sum_j(exp(x_j - m))
 *    减去 m 后 exp 不会溢出；需要 2 次遍历（pass1: max, pass2: exp+sum）
 *    + 一次最终写出 = 共 3 次 pass（或 2 pass + 一次重算）。
 *
 * 3. Online Safe Softmax（Flash Attention 核心思想）
 *    维护 (m, d) 对：m = 当前已见元素的最大值，d = 归一化后的 exp 之和。
 *    每步合并规则：
 *        m_new = max(m, x_i)
 *        d_new = d * exp(m - m_new) + exp(x_i - m_new)
 *    只需一次遍历即可同时得到 max 和归一化 sum，大幅减少访存。
 *    FlashAttention 把这个思想用于 Q·K^T 的 softmax，在 SRAM 内分块完成
 *    attention，避免把整张 N×N attention matrix 写回 HBM，从而将访存复杂
 *    度从 O(N^2) 降至 O(N)。
 *
 * 本文件限制
 * ---------
 *  - D <= 1024（单 block 内用 shared memory 缓存整行）
 *  - grid(M), block(D)，每行一个 block
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

// ============================================================
// Warp-level reduce（蝴蝶归约）
// ============================================================
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

// ============================================================
// Block-level reduce
// 思路：先 warp reduce，warp 0 号线程写 smem；再对 smem 做一次 warp reduce。
// 假设 blockDim.x <= 1024（最多 32 个 warp）。
// ============================================================
__device__ float block_reduce_sum_f32(float val) {
    __shared__ float smem[WARP_SIZE];
    int lane   = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum_f32(val);

    if (lane == 0)
        smem[warp_id] = val;
    __syncthreads();

    // 只有第一个 warp 参与最终归约
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < n_warps) ? smem[lane] : 0.0f;
    if (warp_id == 0)
        val = warp_reduce_sum_f32(val);

    // 广播结果到所有线程（通过 __shfl_sync from lane 0 of warp 0）
    val = __shfl_sync(0xffffffff, val, 0);
    return val;
}

__device__ float block_reduce_max_f32(float val) {
    __shared__ float smem[WARP_SIZE];
    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max_f32(val);

    if (lane == 0)
        smem[warp_id] = val;
    __syncthreads();

    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < n_warps) ? smem[lane] : -1e38f;
    if (warp_id == 0)
        val = warp_reduce_max_f32(val);

    val = __shfl_sync(0xffffffff, val, 0);
    return val;
}

// ============================================================
// (m, d) pair 及其合并操作（Online Softmax 专用）
// ============================================================
struct MD {
    float m;  // 当前最大值
    float d;  // 归一化 exp 之和
};

__device__ __forceinline__ MD md_merge(MD a, MD b) {
    MD res;
    res.m = fmaxf(a.m, b.m);
    res.d = a.d * expf(a.m - res.m) + b.d * expf(b.m - res.m);
    return res;
}

// Warp-level MD reduce
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
        other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
        val = md_merge(val, other);
    }
    return val;
}

// Block-level MD reduce
__device__ MD block_reduce_md(MD val) {
    __shared__ float smem_m[WARP_SIZE];
    __shared__ float smem_d[WARP_SIZE];
    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_md(val);

    if (lane == 0) {
        smem_m[warp_id] = val.m;
        smem_d[warp_id] = val.d;
    }
    __syncthreads();

    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    MD warp_val;
    warp_val.m = (lane < n_warps) ? smem_m[lane] : -1e38f;
    warp_val.d = (lane < n_warps) ? smem_d[lane] : 0.0f;
    if (warp_id == 0)
        warp_val = warp_reduce_md(warp_val);

    // 广播
    warp_val.m = __shfl_sync(0xffffffff, warp_val.m, 0);
    warp_val.d = __shfl_sync(0xffffffff, warp_val.d, 0);
    return warp_val;
}

// ============================================================
// Kernel 1: Naive Softmax（无数值稳定处理）
// grid(M), block(D)；D == blockDim.x <= 1024
// ============================================================
__global__ void naive_softmax_f32(const float* X, float* Y, int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* x = X + row * D;
    float*       y = Y + row * D;

    float val     = x[threadIdx.x];
    float exp_val = expf(val);  // 无 max 减法 → 大值时可能 overflow → NaN

    float sum = block_reduce_sum_f32(exp_val);  // 全 block 求 sum

    y[threadIdx.x] = exp_val / sum;
}

// ============================================================
// Kernel 2: Safe Softmax（3-pass：max → exp+sum → 写出）
// grid(M), block(D)；D == blockDim.x <= 1024
// 用动态 shared memory 缓存 x 和 exp 值，避免重复访问全局内存。
// ============================================================
__global__ void safe_softmax_f32(const float* X, float* Y, int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];  // 大小 = D * sizeof(float)

    float val = X[row * D + threadIdx.x];
    smem[threadIdx.x] = val;
    __syncthreads();

    // pass 1: reduce max（block_reduce_max 会把结果广播给所有线程）
    float max_val = block_reduce_max_f32(val);

    // pass 2: exp(x - max) + reduce sum
    float exp_val = expf(smem[threadIdx.x] - max_val);
    __syncthreads();
    smem[threadIdx.x] = exp_val;  // 复用 smem 存 exp
    __syncthreads();

    float sum = block_reduce_sum_f32(exp_val);

    // pass 3: 写出
    Y[row * D + threadIdx.x] = smem[threadIdx.x] / sum;
}

// ============================================================
// Kernel 3: Online Safe Softmax（单次遍历求 (m,d)，再写出）
// grid(M), block(D)；D == blockDim.x <= 1024
// ============================================================
__global__ void online_safe_softmax_f32(const float* X, float* Y, int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];  // 缓存 x 值供第二次使用

    float xi = X[row * D + threadIdx.x];
    smem[threadIdx.x] = xi;
    __syncthreads();

    // 每个线程持有自己元素的初始 (m, d)
    MD local;
    local.m = xi;
    local.d = 1.0f;  // exp(xi - xi) = 1

    // Block-level merge → 得到全局 (m, d)
    MD global_md = block_reduce_md(local);

    // 利用全局 m 和 d 写出结果
    Y[row * D + threadIdx.x] = expf(smem[threadIdx.x] - global_md.m) / global_md.d;
}

// ============================================================
// CPU 参考实现（Safe Softmax）
// ============================================================
void cpu_safe_softmax(const float* X, float* Y, int M, int D) {
    for (int row = 0; row < M; row++) {
        const float* x = X + row * D;
        float*       y = Y + row * D;

        float max_val = x[0];
        for (int j = 1; j < D; j++)
            if (x[j] > max_val) max_val = x[j];

        float sum = 0.0f;
        for (int j = 0; j < D; j++)
            sum += expf(x[j] - max_val);

        for (int j = 0; j < D; j++)
            y[j] = expf(x[j] - max_val) / sum;
    }
}

// ============================================================
// 工具：检查 CUDA 错误
// ============================================================
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ============================================================
// main
// ============================================================
int main() {
    // ----------------------------------------------------------
    // 测试 1：小矩阵数值正确性（M=4, D=8）
    // ----------------------------------------------------------
    {
        int M = 4, D = 8;
        size_t bytes = (size_t)M * D * sizeof(float);

        float* h_X     = (float*)malloc(bytes);
        float* h_Y_cpu = (float*)malloc(bytes);
        float* h_Y_gpu = (float*)malloc(bytes);

        // 随机初始化
        for (int i = 0; i < M * D; i++)
            h_X[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;

        cpu_safe_softmax(h_X, h_Y_cpu, M, D);

        float *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_X, bytes));
        CUDA_CHECK(cudaMalloc(&d_Y, bytes));
        CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

        size_t smem = D * sizeof(float);

        // --- Naive ---
        naive_softmax_f32<<<M, D>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("[Small M=%d D=%d] Naive   max_err = %.2e  %s\n",
               M, D, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        // --- Safe ---
        safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("[Small M=%d D=%d] Safe    max_err = %.2e  %s\n",
               M, D, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        // --- Online Safe ---
        online_safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("[Small M=%d D=%d] Online  max_err = %.2e  %s\n",
               M, D, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        free(h_X); free(h_Y_cpu); free(h_Y_gpu);
    }

    // ----------------------------------------------------------
    // 测试 2：数值稳定性测试
    // 构造一行 = [0, 0, ..., 0, 80]（最后元素极大）
    // naive 版 exp(80) ≈ 5.5e34，单精度溢出 → NaN；safe/online 版正确
    // ----------------------------------------------------------
    printf("\n--- 数值稳定性测试（行最后一元素 = 80）---\n");
    {
        int M = 1, D = 8;
        size_t bytes = (size_t)M * D * sizeof(float);

        float h_X[8]       = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 80.f};
        float h_Y_cpu[8]   = {0};
        float h_Y_gpu[8]   = {0};

        cpu_safe_softmax(h_X, h_Y_cpu, M, D);

        float *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_X, bytes));
        CUDA_CHECK(cudaMalloc(&d_Y, bytes));
        CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

        size_t smem = D * sizeof(float);

        // Naive
        naive_softmax_f32<<<M, D>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        printf("Naive   output[7] = %f  (expected ~1.0, NaN if overflow)\n",
               h_Y_gpu[D - 1]);

        // Safe
        safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        printf("Safe    output[7] = %f  (expected ~%f)\n",
               h_Y_gpu[D - 1], h_Y_cpu[D - 1]);

        // Online
        online_safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        printf("Online  output[7] = %f  (expected ~%f)\n",
               h_Y_gpu[D - 1], h_Y_cpu[D - 1]);

        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
    }

    // ----------------------------------------------------------
    // 测试 3：大矩阵计时（M=1024, D=256）
    // ----------------------------------------------------------
    printf("\n--- 大矩阵计时 M=1024 D=256 ---\n");
    {
        int M = 1024, D = 256;
        size_t bytes = (size_t)M * D * sizeof(float);

        float* h_X     = (float*)malloc(bytes);
        float* h_Y_cpu = (float*)malloc(bytes);
        float* h_Y_gpu = (float*)malloc(bytes);

        for (int i = 0; i < M * D; i++)
            h_X[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;

        cpu_safe_softmax(h_X, h_Y_cpu, M, D);

        float *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_X, bytes));
        CUDA_CHECK(cudaMalloc(&d_Y, bytes));
        CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

        size_t smem = D * sizeof(float);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int ITERS = 100;

        // --- Naive ---
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < ITERS; t++)
            naive_softmax_f32<<<M, D>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_naive = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("Naive   avg %.3f us  max_err=%.2e  %s\n",
               ms_naive / ITERS * 1000.0f, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        // --- Safe ---
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < ITERS; t++)
            safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_safe = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_safe, start, stop));
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("Safe    avg %.3f us  max_err=%.2e  %s\n",
               ms_safe / ITERS * 1000.0f, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        // --- Online Safe ---
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < ITERS; t++)
            online_safe_softmax_f32<<<M, D, smem>>>(d_X, d_Y, M, D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_online = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_online, start, stop));
        CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));
        max_err = 0.0f;
        for (int i = 0; i < M * D; i++)
            max_err = fmaxf(max_err, fabsf(h_Y_gpu[i] - h_Y_cpu[i]));
        printf("Online  avg %.3f us  max_err=%.2e  %s\n",
               ms_online / ITERS * 1000.0f, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        free(h_X); free(h_Y_cpu); free(h_Y_gpu);
    }

    return 0;
}
