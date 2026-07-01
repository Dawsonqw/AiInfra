# CUDA 入门学习计划

> 环境：RTX 4060 Laptop (Ada Lovelace, sm_89), CUDA 13.2, PyTorch 2.12
> 工作流：物理机写代码 → 容器内编译运行
> 双路线：每个题目同时用 CUDA C++ 和 Triton 实现

## 路线图

```
Ch1       Ch2         Ch3         Ch4         Ch5          Ch6       Ch7
Vector →  Element- →  Block   →  Online  →  Mat      →  SGEMM → Triton
 Add      wise Op     Reduce      Softmax     Transpose           入门

(Hello    (float4,    (warp       (reduce     (shared     (tiling,  (Fused
 World)   fp16,       shuffle,    + online    memory,     smem,     Softmax
          pack)       smem)       algo)       bank        HGEMM)    + MatMul)
                                              conflict)

 ✅        📖           ○           ○           ○            ○         ○

 ⭐        ⭐⭐          ⭐⭐          ⭐⭐⭐        ⭐⭐⭐          ⭐⭐⭐⭐     ⭐⭐⭐
```

**每章学到的核心技能：**

| 章节 | 新增概念 | 与 AI 算子的关系 |
|------|---------|----------------|
| Ch1 Vector Add | grid/block/thread 索引 | 所有算子的基础 |
| Ch2 Elementwise | float4 / fp16 / 向量化 LDG.128 | 所有逐元素算子（ReLU、GELU、Add…） |
| Ch3 Block Reduce | warp shuffle / shared memory / syncthreads | LayerNorm、RMSNorm 的 reduce 部分 |
| Ch4 Softmax | online 算法 / 跨 warp merge | Attention score normalization |
| Ch5 Mat Transpose | smem tiling / bank conflict / padding | GEMM 内部 A/B 加载模式 |
| Ch6 SGEMM | 二维 tiling / 寄存器复用 | 最核心的矩阵乘（FC、Attn QK^T） |
| Ch7 Triton | program_id / tl.dot / kernel fusion | 快速原型、自定义算子 |

## 前置知识

| 概念 | 一句话 |
|------|--------|
| Host vs Device | CPU 是 host，GPU 是 device。Kernel 在 device 上跑，host 负责调度 |
| Grid → Block → Thread | 三级并行：Grid 分 Block，Block 分 Thread |
| Warp | 32 个 Thread 一组，同时执行同一条指令（SIMT） |
| Global Memory | 显存，容量大但慢（RTX 4060 约 256 GB/s） |
| Shared Memory | Block 内共享的片上 SRAM，快约 10 倍（~10 TB/s） |
| Register | 每个线程私有，最快，但数量有限 |

## GPU 参数速查

| GPU | Compute Capability | 编译参数 |
|-----|-------------------|----------|
| RTX 4060/4090 | 8.9 | `-arch=sm_89` |
| RTX 3080 | 8.6 | `-arch=sm_86` |
| A100 | 8.0 | `-arch=sm_80` |
| H100 | 9.0a | `-arch=sm_90a` |

## 参考项目

`LeetCUDA/kernels/` 下每个子目录对应一类算子，包含 `.cu` 实现和 `.py` 测试脚本。读代码时重点关注命名规律：`f32` → `f32x4` → `f16` → `f16x2` → `f16x8_pack`，这条线贯穿所有章节。

## 开始

从 [Chapter 1: Vector Add](./01-vector-add.md) 开始，按顺序递进。
