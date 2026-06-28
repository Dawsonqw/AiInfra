# Chapter 5: Triton 入门

## 目标

理解 Triton 编程模型，实现 Fused Softmax 和 MatMul，感受比 CUDA 高一个抽象层的开发体验。

## 要解决的问题

- CUDA 要手写 shared memory / warp shuffle / 向量化 — Triton 能不能自动做？
- PyTorch 的 softmax 是 4 个 kernel 串起来 — 能不能一个 kernel 搞定（kernel fusion）？
- `tl.dot()` 是什么？它和 Tensor Cores 什么关系？

## 核心概念

| CUDA 概念 | Triton 对应 |
|-----------|------------|
| `blockIdx` | `tl.program_id(axis)` |
| `<<<grid, block>>>` | `kernel[grid](args)` |
| 手动 shared memory | 自动管理（双缓冲） |
| `__shfl_sync` | `tl.reduce` / `tl.max` / `tl.sum` |
| `float4` / `half2` | `tl.load` / `tl.store` 自动向量化 |
| MMA PTX 指令 | `tl.dot()` |

**关键思想**：你只写 tile 级逻辑，编译器帮你做底层优化。

## 两个练习

### Fused Softmax

- 复用 Ch3 的 Online Safe Softmax 算法
- 一个 `tl.arange` + `tl.load` 周期内完成 max + sum
- 与 PyTorch 对比：4 个 kernel → 1 个

### MatMul (Tiled)

- 2D grid：`grid = (cdiv(M, BM), cdiv(N, BN))`
- 用 broadcasting 构建 block pointers
- `tl.dot(a, b)` + FP16 dtype → Tensor Cores 加速

## 验收标准

- Fused Softmax：与 `torch.softmax` 误差 < 1e-5
- MatMul：与 `torch.matmul` 误差 < 1e-4（FP16）
- Triton MatMul 至少不比 Naive CUDA GEMM 慢（大概率更快）
