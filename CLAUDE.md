# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

GPU: RTX 4060 Laptop (Ada Lovelace, sm_89, compute capability 8.9)
容器: nvcr.io/nvidia/pytorch:26.04-py3 (NGC)
工作流: 物理机写代码 → 容器内编译运行

## 容器

```bash
# 容器已就绪，挂载项目根目录到 /workspace
docker exec -it aiinfra bash

# 如果容器没起（重启后）
docker start aiinfra
```

项目目录通过 `-v $(pwd):/workspace` 挂载进容器，物理机上改代码，容器内立刻生效。

## 工具链（均在容器内）

| 工具 | 版本 | 路径 |
|------|------|------|
| nvcc | CUDA 13.2.78 | /usr/local/cuda/bin/nvcc |
| cmake | 3.31.6 | /usr/local/bin/cmake |
| gcc | 13.3.0 | /usr/bin/gcc |
| Python | 3.12.3 | 系统默认 |
| PyTorch | 2.12.0 | NGC 定制版 |
| Triton | 3.6.0 | NGC 定制版 |

CUDA 编译必须用 `-arch=sm_89`。

## 项目结构

```
src/cuda-learning/        # CUDA 学习代码，每章双语言实现
  CMakeLists.txt           # 顶层 cmake（不含 07-triton，该章纯 Python）
  01-vector-add/           # Ch1: grid/block/thread 索引
  02-elementwise/          # Ch2: float4 / fp16 / 向量化访存
  03-reduce/               # Ch3: warp shuffle / shared memory / syncthreads
  04-softmax/              # Ch4: online safe softmax（FlashAttention 前身）
  05-mat-transpose/        # Ch5: smem tiling / bank conflict / padding
  06-sgemm/                # Ch6: 二维 tiling / 寄存器复用
  07-triton/               # Ch7: Triton 综合（fused softmax + fp16 matmul）
docs/cuda-learning/        # 每章学习文档（.md），含概念讲解和路线图
```

## 文件约定（关键）

每章目录下有两组文件，角色不同：

| 文件 | 角色 | CMake 构建？ |
|------|------|-------------|
| `kernel.cu` / `kernel.py` | **参考实现**（完整，可直接运行） | ❌ 不构建 |
| `my_kernel.cu` / `my_kernel.py` | **练习文件**（学习时在此实现，CMake 只构建此文件） | ✅ 构建 |
| `CMakeLists.txt` | 只包含 `add_executable(xxx my_kernel.cu)` | — |

学习流程：读文档 → 理解 `kernel.cu`/`kernel.py` 参考实现 → 在 `my_kernel.cu`/`my_kernel.py` 中自己写 → 编译验证。

`07-triton/` 是例外：纯 Python，无 `.cu` 文件，不需要 CMake，直接 `python kernel.py`。

## 编译与运行

### CUDA C++ — 单文件快速验证（nvcc）

```bash
cd /workspace/src/cuda-learning/01-vector-add
nvcc -std=c++17 -O2 -arch=sm_89 my_kernel.cu -o kernel && ./kernel
```

### CUDA C++ — cmake 构建（所有章节）

```bash
cd /workspace/src/cuda-learning
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build .
```

### Triton（容器内直接跑）

```bash
cd /workspace/src/cuda-learning/01-vector-add
python kernel.py
```

Triton 的 `@triton.jit` 装饰器首次运行 JIT 编译，后续走缓存。

## cmake 骨架

每章子目录的 CMakeLists.txt（只构建 `my_kernel.cu`）：

```cmake
add_executable(elementwise my_kernel.cu)
set_target_properties(elementwise PROPERTIES CUDA_ARCHITECTURES 89)
```

顶层 CMakeLists.txt：

```cmake
cmake_minimum_required(VERSION 3.20)
project(cuda_learning LANGUAGES CXX CUDA)
add_subdirectory(01-vector-add)
add_subdirectory(02-elementwise)
add_subdirectory(03-reduce)
add_subdirectory(04-softmax)
add_subdirectory(05-mat-transpose)
add_subdirectory(06-sgemm)
```

## GPU 参数速查

| GPU | Compute Capability | 编译参数 |
|-----|-------------------|----------|
| RTX 4060/4090 | 8.9 | `-arch=sm_89` |
| RTX 3080 | 8.6 | `-arch=sm_86` |
| A100 | 8.0 | `-arch=sm_80` |
| H100 | 9.0a | `-arch=sm_90a` |

## 前置概念速记

| 概念 | 一句话 |
|------|--------|
| Host vs Device | CPU 是 host，GPU 是 device。Kernel 在 device 上跑，host 负责调度 |
| Grid → Block → Thread | 三级并行：Grid 分 Block，Block 分 Thread |
| Warp | 32 个 Thread 一组，同时执行同一条指令（SIMT） |
| Global Memory | 显存，容量大但慢（RTX 4060 约 256 GB/s） |
| Shared Memory | Block 内共享的片上 SRAM，快约 10 倍（~10 TB/s） |
| Register | 每个线程私有，最快，但数量有限 |
