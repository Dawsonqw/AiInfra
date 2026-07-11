# `<vec_add>`

## 1. 问题定义

### 目标

<!-- 这个 Sample 或算子解决什么问题？ -->
普通向量加法
### 输入与输出

```text
输入：
  T *left,T *right
输出：
  T *result
计算：
  T *result=T *left+T *right; 
```

### 涉及的 CUDA 概念

* grid /block /threadss
* global memory
* coalesced acess

### 初始性能判断

* 预计属于：`Memory Bound`
* 可能的主要瓶颈：数据搬运慢于计算，大部分耗时停留在等待数据搬运
* 性能衡量指标：`带宽 / GFLOP/s / 延迟 / 吞吐`

---

## 2. 官方版本运行

### 源码位置

```text
https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/vectorAdd/vectorAdd.cu
```

### 编译命令

```bash
```

### 运行命令

```bash
```

### 运行参数

* 输入规模：5^10e4
* 数据类型：float
* Block Size：256
* Grid Size：(5^10e4+255)/255

### 运行结果

* [ ] 编译成功
* [ ] 运行成功
* [ ] 正确性验证通过
* [ ] Compute Sanitizer 通过

```text
程序输出：

```

---

## 3. 执行模型

### 数据流

```text
Host Input
    ↓ H2D
Device Input
    ↓
Kernel
    ↓
Device Output
    ↓ D2H
Host Output
```

### Grid 和 Block

```cpp
dim3 block(...);
dim3 grid(...);
```

### Thread 到 Data 的映射

```cpp
idx=blockIdx.x*blockDim.x+threadidx.x
```

### 每个线程负责的工作

```text
读取：
    left[idx] right[idx]
计算：
    left[idx]+right[idx]
写入：
    result[idx]=left[idx]+right[idx]
```

### 内存访问

* Global Memory 读取是否连续：yes
* Global Memory 写入是否连续：yes
* 是否使用 Shared Memory：no
* 是否存在 Bank Conflict：no
* 是否使用 Register 保存中间结果：no

### 同步与边界

* 同步位置：
* 同步原因：
* 边界判断：

```cpp
```

---

## 4. 自己的实现

### CPU Reference

```cpp
```

### 版本规划

| Version | 核心思路         | 状态 |
| ------- | ------------ | -- |
| v0      | 最简单的 CUDA 实现 |    |
| v1      |              |    |
| v2      |              |    |

### v0

核心思路：

```cpp
```

正确性结果：

* 最大误差：
* 是否通过：
* 存在的问题：

### v1

相比 v0 的变化：

*
*

```cpp
```

正确性结果：

* 最大误差：
* 是否通过：

### v2

相比 v1 的变化：

*
*

```cpp
```

正确性结果：

* 最大误差：
* 是否通过：

### 与官方实现的差异

*
*

---

## 5. 性能分析

### Benchmark 配置

* Warmup 次数：
* Benchmark 次数：
* 输入规模：
* 数据类型：
* 是否只统计 Kernel 时间：
* 编译模式：`Release / Debug`

### 性能结果

| Version | Block Size | Time | Bandwidth / GFLOP/s | Correct |
| ------- | ---------: | ---: | ------------------: | ------: |
| 官方版本    |            |      |                     |         |
| v0      |            |      |                     |         |
| v1      |            |      |                     |         |
| v2      |            |      |                     |         |

### Nsight Compute

```bash
ncu --set basic ./program
```

重点结果：

* Compute Throughput：
* Memory Throughput：
* Occupancy：
* DRAM Throughput：
* Bank Conflict：
* 主要 Warp Stall：
* 判断：`Memory Bound / Compute Bound / 其他`

### 优化结论

* 最快版本：
* 相对 v0 加速：
* 最有效的优化：
* 优化有效的原因：
* 无效或负优化：
* 当前主要瓶颈：
* 下一步可以尝试：

---

# 完成检查

* [ ] 能解释问题和计算过程。
* [ ] 能解释 Grid、Block 和线程映射。
* [ ] 能解释主要内存访问方式。
* [ ] 能独立实现正确版本。
* [ ] 能实现至少一个变体。
* [ ] 能覆盖非规整输入。
* [ ] 已通过 Compute Sanitizer。
* [ ] 已使用 CUDA Event 测量。
* [ ] 已计算带宽或计算吞吐。
* [ ] 已使用 Nsight Compute 分析。
* [ ] 能解释优化为什么有效或无效。

---

# 总结

>

