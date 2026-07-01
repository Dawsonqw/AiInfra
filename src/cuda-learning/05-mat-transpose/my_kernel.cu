/*
 * 矩阵转置练习：在此实现三个 kernel
 * ===================================
 * 参考 kernel.cu 的注释理解原理，然后独立实现。
 * 编译：在项目 build 目录执行 cmake --build . 即可生成 mat_transpose 可执行文件。
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

// ─────────────────────────────────────────────────────────────────────────────
// TODO 1: 朴素转置（无 shared memory）
//
// 思路：
//   - 用 2D block(TILE, TILE)，grid 覆盖整个矩阵
//   - 每个线程：col = blockIdx.x*TILE + threadIdx.x（A 的列）
//               row = blockIdx.y*TILE + threadIdx.y（A 的行）
//   - 读 A[row][col]（coalesced），写 B[col][row]（strided，非 coalesced）
//   - 注意边界检查：row < M && col < N
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_naive_f32(const float* __restrict__ A,
                                        float* __restrict__ B,
                                        int M, int N) {
    // TODO: 实现朴素转置
    // int col = ...;
    // int row = ...;
    // if (...) {
    //     B[...] = A[...];
    // }
}

// ─────────────────────────────────────────────────────────────────────────────
// TODO 2: Shared Memory 转置（有 bank conflict）
//
// 思路：
//   __shared__ float smem[TILE][TILE];
//
//   步骤 1：coalesced 读 A → smem[ty][tx]
//     int col = blockIdx.x * TILE + tx;  // A 的列
//     int row = blockIdx.y * TILE + ty;  // A 的行
//     if (row < M && col < N) smem[ty][tx] = A[row*N + col];
//
//   __syncthreads();  // 必须！等待整个 tile 写入完成
//
//   步骤 2：读 smem[tx][ty] → coalesced 写 B
//     int out_col = blockIdx.y * TILE + tx;  // B 的列（对应 A 的行块）
//     int out_row = blockIdx.x * TILE + ty;  // B 的行（对应 A 的列块）
//     if (out_row < N && out_col < M) B[out_row*M + out_col] = smem[tx][ty];
//
//   注意：步骤 2 读 smem[tx][ty] 时同 warp 会触发 32-way bank conflict
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_shared_f32(const float* __restrict__ A,
                                         float* __restrict__ B,
                                         int M, int N) {
    // TODO: 声明 smem，实现两步转置
    // __shared__ float smem[TILE][TILE];
    // ...
}

// ─────────────────────────────────────────────────────────────────────────────
// TODO 3: Bank Conflict Free 转置（+1 padding）
//
// 只需在 TODO 2 的基础上做一处改动：
//   将 __shared__ float smem[TILE][TILE];
//   改为 __shared__ float smem[TILE][TILE + 1];
//
// 其余代码与 TODO 2 完全相同。
// +1 的作用：让同 warp 在步骤 2 读取时落在不同 bank，消除 32-way bank conflict。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_shared_bcf_f32(const float* __restrict__ A,
                                              float* __restrict__ B,
                                              int M, int N) {
    // TODO: 声明 smem[TILE][TILE+1]，其余与 shared 版相同
    // __shared__ float smem[TILE][TILE + 1];
    // ...
}

// ─────────────────────────────────────────────────────────────────────────────
// 工具函数（勿修改）
// ─────────────────────────────────────────────────────────────────────────────
static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static bool verify(const float* A, const float* B, int M, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (A[i * N + j] != B[j * M + i]) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// main（勿修改，实现好三个 kernel 后直接编译运行）
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    const int M = 1024, N = 1024;

    size_t bytes_A = (size_t)M * N * sizeof(float);
    size_t bytes_B = (size_t)N * M * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            h_A[i * N + j] = (float)(i * N + j);

    float *d_A, *d_B;
    check(cudaMalloc(&d_A, bytes_A), "malloc d_A");
    check(cudaMalloc(&d_B, bytes_B), "malloc d_B");
    check(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice), "copy A");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // 测试 naive
    check(cudaMemset(d_B, 0, bytes_B), "memset");
    mat_transpose_naive_f32<<<grid, block>>>(d_A, d_B, M, N);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost), "copy B");
    printf("naive     : %s\n", verify(h_A, h_B, M, N) ? "PASS" : "FAIL");

    // 测试 shared
    check(cudaMemset(d_B, 0, bytes_B), "memset");
    mat_transpose_shared_f32<<<grid, block>>>(d_A, d_B, M, N);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost), "copy B");
    printf("shared    : %s\n", verify(h_A, h_B, M, N) ? "PASS" : "FAIL");

    // 测试 bcf
    check(cudaMemset(d_B, 0, bytes_B), "memset");
    mat_transpose_shared_bcf_f32<<<grid, block>>>(d_A, d_B, M, N);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost), "copy B");
    printf("shared_bcf: %s\n", verify(h_A, h_B, M, N) ? "PASS" : "FAIL");

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
