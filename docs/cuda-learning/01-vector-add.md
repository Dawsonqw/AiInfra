# Chapter 1: Vector Add — Hello World

## 目标

实现 `C[i] = A[i] + B[i]`，跑通 CUDA 全流程。

## 要解决的问题

- 如何把代码从 CPU 发到 GPU 上执行？
- `__global__` 和 `<<<grid, block>>>` 是什么？
- 如何从 `blockIdx.x * blockDim.x + threadIdx.x` 算出全局索引？

## 核心概念

- **Kernel**：在 GPU 上执行的函数，用 `__global__` 标记
- **Launch Config**：`<<<blocks, threads>>>` 指定并行规模
- **线程索引**：`threadIdx`（block 内）、`blockIdx`、`blockDim` / `gridDim`

## CUDA C++ 要点

- 用 `nvcc -std=c++17 -arch=sm_89` 编译
- 用 `cudaMalloc` / `cudaMemcpy` 管理显存，或直接用 PyTorch Tensor（`x.data_ptr<float>()`）
- 用 `torch.utils.cpp_extension.load_inline` 做 JIT 编译 + Python 绑定
- `cudaDeviceSynchronize()` 等 GPU 跑完

## Triton 要点

- `tl.program_id(0)` = CUDA 的 blockIdx
- `tl.arange(0, BLOCK)` + `pid * BLOCK` 算出全局偏移
- `tl.load` / `tl.store` 自动做向量化和边界检查（mask）

## 验收标准

- 与 PyTorch `A + B` 对比，最大误差 < 1e-6
- 手动验证小 N（N=8）每个元素都正确
