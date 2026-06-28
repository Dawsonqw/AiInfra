# CUDA 入门学习计划

> 环境：RTX 4060 Laptop (Ada Lovelace, sm_89), CUDA 13.2, PyTorch 2.12
> 工作流：物理机写代码 → 容器内编译运行
> 双路线：每个题目同时用 CUDA C++ 和 Triton 实现

## 路线图

```
Ch1 ─────── Ch2 ─────── Ch3 ─────── Ch4 ─────── Ch5
Vector       Element-    Warp         Matrix       Triton
 Add   →     wise Op →   Reduce  →   Multiply →   入门
(Hello        (ReLU)     (Softmax)    (SGEMM)      (Fused
 World)                                             Softmax
                                                   + MatMul)

 ⭐️          ⭐️          ⭐️⭐️        ⭐️⭐️⭐️      ⭐️⭐️⭐️
```

每章都包含：
1. 要解决什么问题
2. 核心概念
3. CUDA C++ 实现要点
4. Triton 实现要点
5. 验收标准

## 前置知识

| 概念 | 一句话 |
|------|--------|
| Host vs Device | CPU 是 host，GPU 是 device。Kernel 在 device 上跑，host 负责调度 |
| Grid → Block → Thread | 三级并行：Grid 分 Block，Block 分 Thread |
| Warp | 32 个 Thread 一组，同时执行同一条指令（SIMT） |
| Global Memory | 显存，容量大但慢（256 GB/s on 4060） |
| Shared Memory | Block 内共享的片上 SRAM，快 10 倍（~10 TB/s） |
| Register | 每个线程私有，最快 |

## GPU 参数速查

| GPU | Compute Capability | 编译参数 |
|-----|-------------------|----------|
| RTX 3080 | 8.6 | `-arch=sm_86` |
| RTX 4060/4090 | 8.9 | `-arch=sm_89` |
| A100 | 8.0 | `-arch=sm_80` |
| H100 | 9.0a | `-arch=sm_90a` |

## 开始

从 [Chapter 1: Vector Add](./01-vector-add.md) 开始，按顺序递进。
