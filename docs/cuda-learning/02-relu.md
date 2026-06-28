# Chapter 2: ReLU — 向量化访存

## 目标

实现 `y = max(0, x)` (ReLU)，理解 GPU 内存访问模式。

## 要解决的问题

- 为什么 GPU 喜欢 "相邻线程访问相邻地址"？
- `float4`（128-bit）加载比 `float`（32-bit）快在哪？
- FP16 的 `half2` 如何一次处理 2 个元素？

## 核心概念

- **Memory Coalescing**：相邻线程访问相邻地址时，GPU 会把多次访问合并成一次总线传输
- **向量化访存**：用 `float4` / `half2` 一次读/写多个元素，减少指令数，提高带宽利用率
- **FP32 vs FP16**：半精度省一半带宽、一半显存，但需要处理精度问题

## CUDA C++ 要点

- 标量版（FP32）：`fmaxf(0.0f, x[idx])`
- 向量化 FP32×4：`reinterpret_cast<const float4*>` + 展开
- 向量化 FP16×8：`reinterpret_cast<const half2*>` × 4 组 + 边界检查
- 注意 `half`（CUDA 类型）和 `at::Half`（PyTorch C++ 类型）不一致，传参时用 `reinterpret_cast`

## Triton 要点

- Triton 自动向量化，不需要手写 `float4` / `half2`
- `tl.maximum(0.0, x)` 对应 ReLU
- BLOCK 大小设大一点（如 1024）让 Triton 有足够空间做向量化

## 验收标准

- FP32 与 `torch.relu` 误差 < 1e-6
- 向量化版本比标量版本快 > 2x（N=8M 时）
- 小 N（N=16 且不是 4/8 的倍数）验证边界处理正确
