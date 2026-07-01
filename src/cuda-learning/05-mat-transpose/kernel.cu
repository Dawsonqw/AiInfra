/*
 * 矩阵转置：shared memory tile 与 bank conflict
 * =============================================
 *
 * 问题背景
 * --------
 * 对于 M×N 矩阵 A，朴素转置是：B[j][i] = A[i][j]
 * - 读 A：row-major 逐行读，同一 warp 的 32 个线程读连续地址 → coalesced read
 * - 写 B：同一 warp 的 32 个线程写 B[0][i], B[1][i], ..., B[31][i]
 *         这些地址间距为 M 个 float，步幅为 M*4 字节 → strided (non-coalesced) write
 * 结果：写操作每次需要 32 个独立内存事务，带宽利用率极低。
 *
 * Shared Memory 解决方案
 * ----------------------
 * 将全局内存访问分为两步，均做 coalesced 访问：
 *   步骤 1：coalesced 读全局内存 A → 写入 shared memory（smem）
 *   步骤 2：从 smem 读（转置后方向）→ coalesced 写全局内存 B
 * shared memory 带宽远高于全局内存，可吸收非 coalesced 访问的代价。
 *
 * Bank Conflict
 * -------------
 * Shared memory 分为 32 个 bank，每个 bank 宽度为 4 字节（1 个 float）。
 * 同一 warp 中的线程若访问同一 bank 的不同地址，会产生 bank conflict，
 * 串行化访问，降低 shared memory 带宽。
 *
 * 在 smem[TILE][TILE]（TILE=32）中：
 *   - 步骤 1 写入：smem[threadIdx.y][threadIdx.x] = A[row][col]
 *     同一 warp（threadIdx.y 相同）写不同 threadIdx.x → 不同 bank → 无 conflict
 *   - 步骤 2 读取：smem[threadIdx.x][threadIdx.y]
 *     同一 warp（threadIdx.y 相同）读 smem[0][ty], smem[1][ty], ..., smem[31][ty]
 *     地址为 0*32+ty, 1*32+ty, ..., 31*32+ty
 *     对应 bank = (地址 % 32)，全部 = ty → 同一 bank → 32-way bank conflict!
 *
 * +1 Padding 消除 bank conflict
 * ------------------------------
 * 将 smem 声明为 smem[TILE][TILE+1]：
 *   - 步骤 2 读取：smem[threadIdx.x][threadIdx.y]
 *     地址为 0*(32+1)+ty, 1*(32+1)+ty, ..., 31*(32+1)+ty
 *     对应 bank = (地址 % 32) = ty, ty+1, ty+2, ..., ty+31 (mod 32)
 *     → 32 个不同 bank → 无 conflict!
 * 只增加少量 shared memory（每行 1 个 float），消除了所有 bank conflict。
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32
#define FLOAT4(x)  (reinterpret_cast<float4*>(&(x))[0])
#define CFLOAT4(x) (reinterpret_cast<const float4*>(&(x))[0])

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: 朴素转置（无 shared memory）
//   读 A：coalesced（同一 warp 读连续行元素）
//   写 B：strided（同一 warp 写不连续列元素，步幅 = M）→ 非 coalesced
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_naive_f32(const float* __restrict__ A,
                                        float* __restrict__ B,
                                        int M, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;  // A 的列索引
    int row = blockIdx.y * TILE + threadIdx.y;  // A 的行索引

    if (row < M && col < N) {
        // 读 A[row][col]：同 warp 中 threadIdx.x 连续 → coalesced read
        float val = A[row * N + col];
        // 写 B[col][row]：同 warp 中 threadIdx.x 连续，但写入 B 的行 col 各不相同
        //   → B[col][row] = B + col*M + row，col 因 threadIdx.x 不同而不同
        //   → strided write，步幅 = M，非 coalesced
        B[col * M + row] = val;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Shared Memory 转置（有 bank conflict）
//   步骤 1：coalesced 读全局 A → 写 smem[ty][tx]（同 warp tx 各异，不同列 → 无 read conflict）
//           但写 smem 列方向时，步骤 2 读取会有 bank conflict（见顶部注释）
//   步骤 2：读 smem[tx][ty] → coalesced 写全局 B
//           读 smem 时：同 warp (ty 相同) 读 smem[0..31][ty] → 32-way bank conflict
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_shared_f32(const float* __restrict__ A,
                                         float* __restrict__ B,
                                         int M, int N) {
    __shared__ float smem[TILE][TILE];  // 32×32 shared memory tile

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 步骤 1：从全局内存 coalesced 读入 smem
    int col = blockIdx.x * TILE + tx;  // A 的列（全局）
    int row = blockIdx.y * TILE + ty;  // A 的行（全局）

    if (row < M && col < N) {
        smem[ty][tx] = A[row * N + col];  // coalesced read from A
        // 写 smem[ty][tx]：同 warp tx 各异，写不同 bank → 无 bank conflict（写入阶段）
    }

    __syncthreads();  // 等待整个 tile 写入 smem 完成

    // 步骤 2：从 smem 读出（转置索引），coalesced 写入全局内存 B
    // 转置后的 tile 在 B 中的位置：B 的行 = blockIdx.x * TILE，B 的列 = blockIdx.y * TILE
    int out_col = blockIdx.y * TILE + tx;  // B 的列索引（对应 A 的行块）
    int out_row = blockIdx.x * TILE + ty;  // B 的行索引（对应 A 的列块）

    if (out_row < N && out_col < M) {
        // 读 smem[tx][ty]：同 warp (ty 相同) 读 smem[0..31][ty]
        // → 地址 0*32+ty, 1*32+ty, ..., 31*32+ty，bank = 地址%32 = ty（全相同）
        // → 32-way bank conflict！
        B[out_row * M + out_col] = smem[tx][ty];  // strided smem read (bank conflict)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Bank Conflict Free 转置（+1 padding）
//   smem[TILE][TILE+1]：每行多 1 个 float，使同 warp 读取落在不同 bank
//   步骤 2 读 smem[tx][ty]：地址 = tx*(TILE+1)+ty
//   bank = (tx*(33)+ty) % 32，不同 tx 对应不同 bank → 无 conflict
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mat_transpose_shared_bcf_f32(const float* __restrict__ A,
                                              float* __restrict__ B,
                                              int M, int N) {
    __shared__ float smem[TILE][TILE + 1];  // +1 padding 消除 bank conflict

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 步骤 1：coalesced 读全局 A → smem（与 shared 版相同）
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    if (row < M && col < N) {
        smem[ty][tx] = A[row * N + col];  // coalesced read from A
    }

    __syncthreads();

    // 步骤 2：读 smem（+1 padding → 无 bank conflict）→ coalesced 写 B
    int out_col = blockIdx.y * TILE + tx;
    int out_row = blockIdx.x * TILE + ty;

    if (out_row < N && out_col < M) {
        // 读 smem[tx][ty]：地址 = tx*(TILE+1)+ty = tx*33+ty
        // 同 warp (ty 相同)：bank = (tx*33+ty) % 32
        //   tx=0: ty%32, tx=1: (33+ty)%32=(1+ty)%32, tx=2: (66+ty)%32=(2+ty)%32, ...
        //   → bank 依次递增 1，32 个线程落在 32 个不同 bank → 无 conflict！
        B[out_row * M + out_col] = smem[tx][ty];  // bank conflict free smem read
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────────────────────────────────────
static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// 验证转置结果：B[j][i] == A[i][j]（精确匹配，因为数值是整数以 float 存储）
static bool verify(const float* A, const float* B, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i * N + j] != B[j * M + i]) {
                fprintf(stderr, "  MISMATCH at [%d][%d]: A=%f B=%f\n",
                        i, j, A[i * N + j], B[j * M + i]);
                return false;
            }
        }
    }
    return true;
}

// 用 CUDA events 对 kernel 计时，warmup + iters 取平均
static float bench(void (*launch)(const float*, float*, int, int,
                                  dim3, dim3),
                   const float* d_A, float* d_B,
                   int M, int N,
                   dim3 grid, dim3 block,
                   int warmup, int iters) {
    // warmup
    for (int i = 0; i < warmup; i++) {
        launch(d_A, d_B, M, N, grid, block);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launch(d_A, d_B, M, N, grid, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// 每个 kernel 的启动封装（统一函数签名供 bench 使用）
static void launch_naive(const float* A, float* B, int M, int N,
                         dim3 grid, dim3 block) {
    mat_transpose_naive_f32<<<grid, block>>>(A, B, M, N);
}
static void launch_shared(const float* A, float* B, int M, int N,
                          dim3 grid, dim3 block) {
    mat_transpose_shared_f32<<<grid, block>>>(A, B, M, N);
}
static void launch_bcf(const float* A, float* B, int M, int N,
                       dim3 grid, dim3 block) {
    mat_transpose_shared_bcf_f32<<<grid, block>>>(A, B, M, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    const int M = 1024, N = 1024;
    const int WARMUP = 5, ITERS = 20;

    size_t bytes_A = (size_t)M * N * sizeof(float);
    size_t bytes_B = (size_t)N * M * sizeof(float);  // 转置后 N×M

    // 分配并初始化主机内存
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            h_A[i * N + j] = (float)(i * N + j);  // 整数值，便于精确验证

    // 分配设备内存
    float *d_A, *d_B;
    check(cudaMalloc(&d_A, bytes_A), "malloc d_A");
    check(cudaMalloc(&d_B, bytes_B), "malloc d_B");
    check(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice), "copy A");

    // grid / block 配置
    // block(TILE, TILE) = (32, 32) = 1024 线程，合法最大 block size
    // grid.x 对应列方向（N），grid.y 对应行方向（M）
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    struct {
        const char* name;
        void (*fn)(const float*, float*, int, int, dim3, dim3);
    } kernels[] = {
        {"mat_transpose_naive     ", launch_naive},
        {"mat_transpose_shared    ", launch_shared},
        {"mat_transpose_shared_bcf", launch_bcf},
    };

    for (auto& k : kernels) {
        // 清零输出，避免上次结果干扰验证
        check(cudaMemset(d_B, 0, bytes_B), "memset B");

        // 先跑一次获取结果用于验证
        k.fn(d_A, d_B, M, N, grid, block);
        check(cudaDeviceSynchronize(), "sync");

        check(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost), "copy B");
        bool ok = verify(h_A, h_B, M, N);

        // 计时
        float ms = bench(k.fn, d_A, d_B, M, N, grid, block, WARMUP, ITERS);

        printf("%-28s : time=%6.3fms  %s\n", k.name, ms, ok ? "PASS" : "FAIL");
    }

    // 性能说明
    printf("\n注：bcf 版理论上比 shared 版快（消除 32-way bank conflict），"
           "naive 版最慢（strided 全局内存写）。\n");

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
