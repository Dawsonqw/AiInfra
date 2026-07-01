// 03-reduce/kernel.cu
// 主题：Block All Reduce Sum
//
// 核心概念：
//   1. Warp Shuffle 蝴蝶归约（butterfly reduce）
//      - __shfl_xor_sync(mask, val, offset)：让 lane_id 与 lane_id^offset 的线程交换 val
//      - 每轮步长减半（32→16→8→4→2→1），共 log2(32)=5 步
//      - 结束后每个 lane 都持有 warp 内所有元素的 sum
//   2. Shared Memory 跨 warp 汇总
//      - 每个 warp 的 lane 0 将本 warp 的小计写入 smem
//      - __syncthreads() 确保所有 warp 都写完后，warp 0 再对 smem 做一次 reduce
//   3. atomicAdd 跨 block 汇总到全局输出

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 32
#define FLOAT4(x)  (reinterpret_cast<float4*>(&(x))[0])
#define CFLOAT4(x) (reinterpret_cast<const float4*>(&(x))[0])

// ─────────────────────────────────────────────────────────────
// Device 函数：warp 内 sum 归约
//   输入：val —— 本 lane 的值
//   输出：每个 lane 都返回 warp 内 32 个 val 的总和
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    // 蝴蝶归约：O(log WARP_SIZE) 步
    // mask = 16: lane 与 lane^16 交换，相加 → 16 对相邻 lane 各持有 2 个数的和
    // mask =  8: 再交换 → 每 lane 持有 4 个数的和
    // ...
    // mask =  1: 最后一步 → 每 lane 持有全部 32 个数的和
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ─────────────────────────────────────────────────────────────
// Device 函数：warp 内 max 归约
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

// ─────────────────────────────────────────────────────────────
// Kernel 1：block_all_reduce_sum_f32
//   grid(N / blockDim.x), block(256)
//   每线程加载 1 个 float，通过三级归约：
//     warp reduce → smem 汇总 → warp 0 再 reduce → atomicAdd
// ─────────────────────────────────────────────────────────────
__global__ void block_all_reduce_sum_f32(const float* __restrict__ A,
                                          float* __restrict__ y,
                                          int N) {
    // ── 第一步：每线程读取自己负责的元素 ──────────────────────
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? A[idx] : 0.0f;

    // ── 第二步：warp 内归约 ────────────────────────────────────
    // 每个 warp 的 32 条 lane 协作，结果每条 lane 都相同
    val = warp_reduce_sum_f32(val);

    // ── 第三步：lane 0 把本 warp 的小计写入 shared memory ──────
    // smem 大小 = WARP_SIZE = 32，最多能容纳 256/32 = 8 个 warp 的小计
    __shared__ float smem[WARP_SIZE];
    int lane   = threadIdx.x % WARP_SIZE;  // 在本 warp 中的位置（0~31）
    int warpId = threadIdx.x / WARP_SIZE;  // 本 block 内的 warp 编号

    if (lane == 0)
        smem[warpId] = val;

    // ── 第四步：等所有 warp 都写完 smem ───────────────────────
    __syncthreads();

    // ── 第五步：warp 0 对 smem 做最后一次 warp reduce ─────────
    // 此时有效数据只有 blockDim.x/WARP_SIZE 个（如 256/32=8 个）
    // 超出部分用 0 填充，保证 warp_reduce_sum_f32 行为正确
    int numWarps = blockDim.x / WARP_SIZE;
    val = (lane < numWarps) ? smem[lane] : 0.0f;

    if (warpId == 0)
        val = warp_reduce_sum_f32(val);

    // ── 第六步：thread 0 将本 block 的 sum 原子累加到输出 ──────
    if (threadIdx.x == 0)
        atomicAdd(y, val);
}

// ─────────────────────────────────────────────────────────────
// Kernel 2：block_all_reduce_sum_f32x4
//   每线程用 float4 一次加载 4 个元素，本地先累加再走同样三级归约
//   grid(N / 4 / blockDim.x), block(256)
// ─────────────────────────────────────────────────────────────
__global__ void block_all_reduce_sum_f32x4(const float* __restrict__ A,
                                            float* __restrict__ y,
                                            int N) {
    // 每线程负责 4 个元素
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // ── 第一步：float4 加载并本地累加 ─────────────────────────
    float val = 0.0f;
    if (idx4 + 3 < N) {
        float4 data = CFLOAT4(A[idx4]);
        val = data.x + data.y + data.z + data.w;
    } else {
        // 边界处理：逐个读取
        for (int i = 0; i < 4 && idx4 + i < N; i++)
            val += A[idx4 + i];
    }

    // ── 第二步以后与 f32 版本完全相同 ─────────────────────────
    val = warp_reduce_sum_f32(val);

    __shared__ float smem[WARP_SIZE];
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (lane == 0)
        smem[warpId] = val;

    __syncthreads();

    int numWarps = blockDim.x / WARP_SIZE;
    val = (lane < numWarps) ? smem[lane] : 0.0f;

    if (warpId == 0)
        val = warp_reduce_sum_f32(val);

    if (threadIdx.x == 0)
        atomicAdd(y, val);
}

// ─────────────────────────────────────────────────────────────
// Kernel 3（Bonus）：block_all_reduce_max_f32
//   与 sum 版本结构相同，归约操作换成 fmaxf
// ─────────────────────────────────────────────────────────────
__global__ void block_all_reduce_max_f32(const float* __restrict__ A,
                                          float* __restrict__ y,
                                          int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 不参与的线程用 -FLT_MAX，不影响 max 结果
    float val = (idx < N) ? A[idx] : -3.402823466e+38f;

    val = warp_reduce_max_f32(val);

    __shared__ float smem[WARP_SIZE];
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (lane == 0)
        smem[warpId] = val;

    __syncthreads();

    int numWarps = blockDim.x / WARP_SIZE;
    val = (lane < numWarps) ? smem[lane] : -3.402823466e+38f;

    if (warpId == 0)
        val = warp_reduce_max_f32(val);

    // atomicMax 不直接支持 float，用 int 位模拟（正数安全）
    if (threadIdx.x == 0) {
        // 对于非负 float，int 比较等价于 float 比较（IEEE 754 保证）
        atomicMax(reinterpret_cast<int*>(y),
                  __float_as_int(val));
    }
}

// ─────────────────────────────────────────────────────────────
// 辅助：CUDA Events 计时
// ─────────────────────────────────────────────────────────────
static float benchmark(void (*launcher)(void*), void* args,
                       int warmup, int iters) {
    // 此处用宏包装，实际计时见 main() 内联
    (void)launcher; (void)args; (void)warmup; (void)iters;
    return 0.0f;
}

int main() {
    const int N       = 1 << 24;  // 16M elements
    const int BLOCK   = 256;
    const int WARMUP  = 5;
    const int ITERS   = 20;

    // ── 主机端初始化：全 1 数组 ─────────────────────────────────
    float* h_A = new float[N];
    for (int i = 0; i < N; i++) h_A[i] = 1.0f;

    // ── 设备端分配 ──────────────────────────────────────────────
    float *d_A, *d_y;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    float result = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ═══════════════════════════════════════════════════════════
    // Benchmark: block_all_reduce_sum_f32
    // ═══════════════════════════════════════════════════════════
    {
        dim3 grid((N + BLOCK - 1) / BLOCK);
        dim3 block(BLOCK);

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            cudaMemset(d_y, 0, sizeof(float));
            block_all_reduce_sum_f32<<<grid, block>>>(d_A, d_y, N);
        }
        cudaDeviceSynchronize();

        // timed iters
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            cudaMemset(d_y, 0, sizeof(float));
            block_all_reduce_sum_f32<<<grid, block>>>(d_A, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= ITERS;

        cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost);
        bool pass = (fabsf(result - (float)N) < 1.0f);
        printf("block_all_reduce_sum_f32   : time=%.3fms  sum=%.0f  %s\n",
               ms, result, pass ? "PASS" : "FAIL");
    }

    // ═══════════════════════════════════════════════════════════
    // Benchmark: block_all_reduce_sum_f32x4
    // ═══════════════════════════════════════════════════════════
    {
        // 每线程处理 4 个元素
        dim3 grid((N / 4 + BLOCK - 1) / BLOCK);
        dim3 block(BLOCK);

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            cudaMemset(d_y, 0, sizeof(float));
            block_all_reduce_sum_f32x4<<<grid, block>>>(d_A, d_y, N);
        }
        cudaDeviceSynchronize();

        // timed iters
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            cudaMemset(d_y, 0, sizeof(float));
            block_all_reduce_sum_f32x4<<<grid, block>>>(d_A, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= ITERS;

        cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost);
        bool pass = (fabsf(result - (float)N) < 1.0f);
        printf("block_all_reduce_sum_f32x4 : time=%.3fms  sum=%.0f  %s\n",
               ms, result, pass ? "PASS" : "FAIL");
    }

    // ═══════════════════════════════════════════════════════════
    // Benchmark: block_all_reduce_max_f32 (Bonus)
    // ═══════════════════════════════════════════════════════════
    {
        dim3 grid((N + BLOCK - 1) / BLOCK);
        dim3 block(BLOCK);

        // 初始化为一个极小值（float 位模式），让 atomicMax(int) 正确工作
        float neg_inf = -3.402823466e+38f;
        cudaMemcpy(d_y, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            cudaMemcpy(d_y, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
            block_all_reduce_max_f32<<<grid, block>>>(d_A, d_y, N);
        }
        cudaDeviceSynchronize();

        // timed iters
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            cudaMemcpy(d_y, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
            block_all_reduce_max_f32<<<grid, block>>>(d_A, d_y, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= ITERS;

        cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost);
        bool pass = (fabsf(result - 1.0f) < 1e-5f);
        printf("block_all_reduce_max_f32   : time=%.3fms  max=%.0f  %s\n",
               ms, result, pass ? "PASS" : "FAIL");
    }

    // ── 清理 ────────────────────────────────────────────────────
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_y);
    delete[] h_A;

    return 0;
}
