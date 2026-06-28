# Chapter 4: SGEMM — 矩阵乘法与 Shared Memory Tiling

## 目标

实现 `C = A @ B`（FP32），从 Naive → Tiled，理解 Shared Memory 为什么是 GPU 性能的关键。

## 要解决的问题

- Naive GEMM 每个输出元素读 K 次全局内存，M=N=K=1024 时 ≈ 20 亿次访问 — 怎么降？
- Shared Memory 比 Global Memory 快 10 倍，但只有 ~100KB — 怎么用？

## 核心概念

- **Tiling（分块）**：把 K 维度切成 BK 大小的块，每块加载到 shared memory，片上计算
- **内存节省**：全局访问从 `2K` 降到 `2K/BK`（BK=32 时降到 1/16）
- **`__shared__`**：block 内所有线程共享，生命周期 = kernel
- **`__syncthreads()` 的双重用途**：
  1. 加载后：确保所有数据都进 smem
  2. 计算后：确保所有线程用完 smem 再覆写

## CUDA C++ 要点

**Naive 版**（无 shared memory）：
- `grid=(N/16, M/16), block=(16, 16)`
- 每个线程直接读全局内存，做 K 次乘加

**Tiled 版**（shared memory）：
- `grid=(N/BN, M/BM), block=(BN, BM)` — 每个 block 负责 C 的一个 BM×BN 块
- `__shared__ float sA[BM][BK], sB[BK][BN]` — 两个 tile buffer
- 外循环 `for (bk = 0; bk < K; bk += BK)` — 遍历 K 方向
- 加载：整个 block 协同，每个线程加载一个 A 和一个 B 元素
- 计算：`for (k = 0; k < BK; k++) sum += sA[ty][k] * sB[k][tx]`
- **两个 `__syncthreads()`** 夹住计算部分

## Triton 要点

- `tl.dot(a, b)` 一行搞定，且自动用 Tensor Cores（FP16/BF16 下）
- 2D grid：`pid_m = tl.program_id(0)`, `pid_n = tl.program_id(1)`
- Block pointers 用 broadcasting 构建：`rm[:, None] * K + rk[None, :]`

## 验收标准

- 与 `torch.matmul` 误差 < 1e-4（FP32）
- Tiled 版比 Naive 版快 > 5x（M=N=K=1024）
- 小矩阵（M=N=K=128）验证无越界
