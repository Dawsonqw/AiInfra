# AI Infra — CLAUDE.md

## 环境

GPU: RTX 4060 Laptop (Ada Lovelace, sm_89, compute capability 8.9)
容器: nvcr.io/nvidia/pytorch:26.04-py3 (NGC)
工作流: 物理机写代码 → 容器内编译运行

## 容器

```bash
# 容器已就绪，挂载项目根目录到 /workspace
# 物理机 /home/dawsonqw/owner/AiInfra == 容器 /workspace

# 进入容器
docker exec -it aiinfra bash

# 如果容器没起（重启后）
docker start aiinfra
```

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
src/cuda-learning/        # CUDA 学习代码（每章 .cu + .py）
  CMakeLists.txt           # 顶层 cmake
  01-vector-add/
  02-relu/
  03-softmax/
  04-sgemm/
  05-triton-fused/
docs/cuda-learning/        # 学习计划（markdown）
```

## 编译与运行

### CUDA C++（nvcc 直接编译单文件，快速验证）

```bash
cd /workspace/src/cuda-learning/01-vector-add
nvcc -std=c++17 -O2 -arch=sm_89 kernel.cu -o kernel && ./kernel
```

### CUDA C++（cmake 构建）

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

Triton 脚本自带 `@triton.jit` 装饰器，首次运行 JIT 编译，后续走缓存。

## cmake 骨架

每个子目录的 CMakeLists.txt 模板：

```cmake
enable_language(CUDA)
add_executable(relu kernel.cu)
set_target_properties(relu PROPERTIES CUDA_ARCHITECTURES 89)
```

顶层 CMakeLists.txt：

```cmake
cmake_minimum_required(VERSION 3.20)
project(cuda-learning LANGUAGES CXX CUDA)
add_subdirectory(01-vector-add)
add_subdirectory(02-relu)
# ...
```
