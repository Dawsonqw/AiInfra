#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32

// ============================================================
// 设备辅助函数（可直接复制 kernel.cu 中的实现，或自己写）
// ============================================================

// TODO: 实现 warp_reduce_sum_f32
// Hint: 用 __shfl_xor_sync(0xffffffff, val, mask) 蝴蝶归约
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    return val;
}

// TODO: 实现 warp_reduce_max_f32
// Hint: 用 fmaxf + __shfl_xor_sync 蝴蝶归约
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    return val;
}

// TODO: 实现 block_reduce_sum_f32
// Hint: 先 warp reduce → warp leader 写 smem → 第一个 warp 再 reduce → 广播
__device__ float block_reduce_sum_f32(float val) {
    return val;
}

// TODO: 实现 block_reduce_max_f32
// Hint: 同上，但 reduce 操作换成 fmaxf
__device__ float block_reduce_max_f32(float val) {
    return val;
}

// ============================================================
// Online Softmax 专用：(m, d) pair 及合并操作
// ============================================================

struct MD {
    float m;  // 当前最大值
    float d;  // 归一化 exp 之和
};

// TODO: 实现 md_merge
// Hint: m_new = max(a.m, b.m)
//       d_new = a.d * exp(a.m - m_new) + b.d * exp(b.m - m_new)
__device__ __forceinline__ MD md_merge(MD a, MD b) {
    MD res;
    res.m = a.m;
    res.d = a.d;
    return res;
}

// TODO: 实现 warp_reduce_md（对 MD 做蝴蝶归约）
// Hint: 用 __shfl_xor_sync 分别交换 m 和 d，再调用 md_merge
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    return val;
}

// TODO: 实现 block_reduce_md
// Hint: 和 block_reduce_sum/max 类似，但存储 MD（需要两个 smem 数组）
__device__ MD block_reduce_md(MD val) {
    return val;
}

// ============================================================
// Kernel 1: TODO — 实现 naive_softmax_f32
//
// 算法：y_i = exp(x_i) / sum_j(exp(x_j))
// 无 max 减法，大值时 exp 溢出 → NaN（预期行为，用于对比）
//
// 参数：grid(M), block(D)；D == blockDim.x <= 1024
// ============================================================
__global__ void naive_softmax_f32(const float* X, float* Y, int M, int D) {
    // 你的代码
}

// ============================================================
// Kernel 2: TODO — 实现 safe_softmax_f32
//
// 算法：先 reduce max，再 exp(x - max) / sum(exp(x - max))
// 需要 3 次 block reduce + shared memory 缓存 x 值
//
// Hint:
//   1. 用 extern __shared__ float smem[] 缓存当前行
//   2. block_reduce_max_f32 → 得到 max_val（所有线程都有）
//   3. 每线程计算 exp_val = expf(smem[tid] - max_val)
//   4. block_reduce_sum_f32 → 得到 sum
//   5. 写出 exp_val / sum
//
// 启动时：safe_softmax_f32<<<M, D, D*sizeof(float)>>>(...)
// ============================================================
__global__ void safe_softmax_f32(const float* X, float* Y, int M, int D) {
    extern __shared__ float smem[];
    // 你的代码
}

// ============================================================
// Kernel 3: TODO — 实现 online_safe_softmax_f32
//
// 算法：维护 (m, d) 对，一次遍历同时求 max 和归一化 sum
//   初始化：local.m = x[tid], local.d = 1.0f
//   block_reduce_md → 得到全局 (global_m, global_d)
//   写出：exp(x[tid] - global_m) / global_d
//
// Hint: 用 smem 缓存 x，避免第二次访问全局内存
//
// 启动时：online_safe_softmax_f32<<<M, D, D*sizeof(float)>>>(...)
// ============================================================
__global__ void online_safe_softmax_f32(const float* X, float* Y, int M, int D) {
    extern __shared__ float smem[];
    // 你的代码
}

// ============================================================
// CPU 参考实现（勿修改，用于验证）
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

int main() {
    // TODO: 参考 kernel.cu 中的 main() 进行验证
    // 建议步骤：
    //   1. 小矩阵（M=4, D=8）验证三个 kernel 的数值正确性
    //   2. 数值稳定性测试：构造行 [0,...,0,80]，观察 naive 是否 NaN
    //   3. 大矩阵（M=1024, D=256）计时，比较三个版本的性能
    printf("TODO: 完成 kernel 实现后在此验证\n");
    return 0;
}
