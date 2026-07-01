# Chapter 2: Elementwise — 向量化访存

## 目标

在 Vector Add 的基础上，用同一个操作（逐元素加法 / ReLU）系统掌握 GPU 内存访问优化，从 f32 标量一路做到 f16x8 pack。

参考：`LeetCUDA/kernels/elementwise/`、`LeetCUDA/kernels/relu/`

## 要解决的问题

- Ch1 的 Vector Add 每个线程读 1 个 float（32-bit），带宽利用率很低——为什么？
- `float4`（128-bit LDG）比 `float` 快在哪？一次 128-bit 传输 vs 四次 32-bit 有什么区别？
- FP16 的 `half2` 如何一条指令处理 2 个元素？`f16x8_pack` 又是什么？

## 核心概念

- **Memory Coalescing**：相邻线程访问相邻地址时，GPU 把多次访问合并成一次 cache line 传输（128B）。标量版 1 线程读 4B，向量化版 1 线程读 16B，同样的线程数吃满带宽。
- **向量化访存（LDG.128）**：用 `float4` / `int4` 一次发出 128-bit 加载指令（`LDG.E.128`），减少指令数和 SM 调度开销。
- **FP32 vs FP16**：半精度省一半带宽。`half2` 是两个 fp16 打包成 32-bit，一条 SIMD 指令同时计算；`f16x8_pack` 用 `LDG.128` 一次加载 8 个 fp16。
- **宏 `FLOAT4(x)`**：`reinterpret_cast<float4*>(&x)[0]`，原地把 `float*` 变成 `float4*`，零开销。

## CUDA C++ 实现路径

| 版本 | 每线程处理 | 指令 | 说明 |
|------|-----------|------|------|
| `f32` | 1 × float | LDG.E.32 | 基准，对应 Ch1 |
| `f32x4` | 4 × float | LDG.E.128 | 用 `FLOAT4` 宏 + `float4` 展开 |
| `f16` | 1 × half | LDG.E.16 | `__hadd(a, b)` |
| `f16x2` | 2 × half | LDG.E.32 | `__hadd2(HALF2(a), HALF2(b))` |
| `f16x8_pack` | 8 × half | LDG.E.128 | `LDST128BITS` 宏 + `half2` × 4 展开 |

关键宏（LeetCUDA 风格）：

```cpp
#define FLOAT4(x)       (reinterpret_cast<float4*>(&(x))[0])
#define HALF2(x)        (reinterpret_cast<half2*>(&(x))[0])
#define LDST128BITS(x)  (reinterpret_cast<float4*>(&(x))[0])
```

边界处理：向量化版本 `idx` 步长是 4 或 8，最后一段不满时退回标量循环。

## Triton 要点

- Triton 自动向量化，不需要手写 `float4` / `half2`
- `tl.load(ptr + offsets, mask=mask)` 自动选择 128-bit 加载
- BLOCK 大小设大（如 1024）让编译器有足够空间做向量化

## 验收标准

- FP32 与 `torch` 参考结果误差 < 1e-6
- f32x4 版比 f32 标量版快 > 2x（N=8M 时）
- f16x8_pack 版比 f16 标量版快 > 3x
- 小 N（N=16，不是 4/8 的倍数）验证边界处理正确
