# CUDA Sample 五步学习闭环

对于每一个 CUDA Sample，都按照下面五个步骤完成学习。目标不是仅仅把官方代码运行起来，而是能够理解其执行模型、独立重写、构造变体，并通过性能工具验证自己的判断。

---

## 第一步：先阅读 README，不看实现

在打开 `.cu` 源码之前，先阅读 Sample 目录中的：

```text
README.md
CMakeLists.txt
```

先回答以下问题：

* 这个 Sample 要解决什么问题？
* 它主要展示了哪个 CUDA 概念？
* 输入和输出的数据结构是什么？
* Kernel 的 Grid 和 Block 可能如何划分？
* 可能会使用哪些 CUDA Runtime API？
* 这个 Kernel 更可能是 Memory Bound，还是 Compute Bound？
* 正确性应该如何验证？
* 性能应该使用什么指标衡量？

例如，对于 Vector Add：

```text
输入：A[N]、B[N]
输出：C[N]

计算：
C[i] = A[i] + B[i]

可能的线程映射：
一个线程处理一个元素
```

在阅读源码前，先尝试自己写出伪代码：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

### 本步骤输出

在学习笔记中记录：

```markdown
## 问题定义

## 输入输出

## CUDA 概念

## 线程映射猜想

## 可能的性能瓶颈
```

---

## 第二步：编译并运行官方版本

先不要修改官方代码，保持原始 Sample 不变。

单独编译目标 Sample：

```bash
cmake \
    -S <sample_directory> \
    -B build/<sample_name> \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build/<sample_name> -j"$(nproc)"
```

运行程序：

```bash
./build/<sample_name>/<executable>
```

如果不确定可执行文件位置：

```bash
find build/<sample_name> -type f -executable
```

运行时确认：

* 程序能正常启动。
* CUDA Device 能正确识别。
* 输出结果通过正确性验证。
* 没有 CUDA Runtime Error。
* 输入规模和运行参数是什么。
* Sample 是否支持命令行参数。
* Debug 和 Release 模式是否存在明显性能差异。

可以使用 Compute Sanitizer 检查：

```bash
compute-sanitizer --tool memcheck ./program
```

涉及 Shared Memory 或同步时，还可以使用：

```bash
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool synccheck ./program
```

### 本步骤输出

记录：

```markdown
## 官方版本运行结果

- CUDA Toolkit：
- GPU：
- Compute Capability：
- 输入规模：
- Block Size：
- Grid Size：
- 执行结果：
- 是否通过正确性验证：
- 是否通过 Compute Sanitizer：
```

---

## 第三步：画出执行模型和数据流

在深入源码之前，先把 Host、Device、Kernel 和数据传输关系画出来。

例如，Vector Add 的数据流：

```text
Host A ──H2D──> Device A ─┐
                           ├── Vector Add Kernel ──> Device C ──D2H──> Host C
Host B ──H2D──> Device B ─┘
```

线程映射：

```text
Block 0:
thread 0   → element 0
thread 1   → element 1
...
thread 255 → element 255

Block 1:
thread 0   → element 256
thread 1   → element 257
...
```

索引计算：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

手动计算一个具体例子：

```text
元素数量 N = 1000
Block Size = 256

Grid Size：
ceil(1000 / 256) = 4

总线程数量：
4 × 256 = 1024

多余线程数量：
1024 - 1000 = 24
```

这些线程必须通过边界判断过滤：

```cpp
if (idx < n) {
    // 处理数据
}
```

对于 Shared Memory、Reduction、Transpose 等 Sample，还需要画出：

* 每个 Block 负责哪部分数据。
* 每个线程读取哪个 Global Memory 地址。
* 每个线程写入哪个 Shared Memory 地址。
* Warp 内线程的地址是否连续。
* Shared Memory 地址映射到哪个 Bank。
* 哪些位置需要 `__syncthreads()`。
* 中间结果存放在 Register、Shared Memory 还是 Global Memory。

### 本步骤输出

记录：

```markdown
## 数据流

## Grid 和 Block 配置

## Thread 到 Data 的映射

## Global Memory 访问模式

## Shared Memory 访问模式

## 同步位置

## 边界处理
```

---

## 第四步：关闭官方代码，独立重写

理解执行模型后，关闭官方源码，在自己的学习仓库中重新实现。

不要直接复制官方代码，也不要在官方仓库中修改。

推荐目录：

```text
cuda-kernel-lab/
├── include/
│   ├── cuda_check.cuh
│   ├── cuda_timer.cuh
│   └── benchmark.cuh
├── week01/
│   └── vector_add/
│       ├── vector_add_v0.cu
│       ├── vector_add_v1.cu
│       ├── test.cpp
│       └── README.md
└── references/
    └── cuda-samples/
```

每个算子至少包含：

```text
CPU Reference
CUDA Naive Version
CUDA Optimized Version
Correctness Test
Benchmark
实验记录
```

推荐实现顺序：

```text
v0：最简单、最容易验证的版本
v1：使用标准 CUDA 优化方法
v2：尝试自己的优化
```

例如 Vector Add：

```text
v0：一个线程处理一个元素
v1：Grid-Stride Loop
v2：一个线程处理多个元素
v3：float4 向量化 Load/Store
```

独立实现后，再打开官方源码进行对比：

* 索引方式是否相同？
* Grid 和 Block 的配置是否相同？
* 官方代码如何处理边界？
* 官方代码如何检查错误？
* 官方代码是否使用模板？
* 官方实现是否考虑了特殊硬件能力？
* 自己的实现是否遗漏了输入规模或对齐问题？

测试时不要只使用规整尺寸。

例如：

```text
1
31
32
33
255
256
257
1000
1024
1025
1024 × 1024
```

矩阵算子还应测试：

```text
M = 1000
N = 1003
K = 997
```

### 本步骤输出

记录：

```markdown
## 自己的实现

### v0

### v1

### v2

## 与官方实现的差异

## 遇到的问题

## 错误原因

## 修复方法
```

---

## 第五步：构造变体并进行性能分析

代码正确只是起点。最后一步需要建立 Benchmark，并使用性能分析工具验证自己的判断。

每个版本至少记录：

* 输入规模。
* 数据类型。
* Block Size。
* Grid Size。
* Shared Memory 使用量。
* Register 使用量。
* 平均执行时间。
* 有效带宽或计算吞吐。
* 相对 CPU、官方版本或 CUDA Library 的性能。
* 正确性误差。

### Benchmark 基本要求

先进行预热：

```text
Warmup：10～20 次
```

再正式测试：

```text
Benchmark：100～1000 次
```

使用 CUDA Event 测量 Kernel 时间：

```cpp
cudaEventRecord(start, stream);

kernel<<<grid, block, shared_memory, stream>>>(...);

cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);
```

不要把以下操作混入 Kernel 时间：

```text
cudaMalloc
cudaFree
Host 数据初始化
H2D 拷贝
D2H 拷贝
文件读写
结果打印
```

### 有效带宽

例如 Vector Add：

```text
读取 A：N × sizeof(float)
读取 B：N × sizeof(float)
写入 C：N × sizeof(float)

总数据量：
3 × N × sizeof(float)
```

有效带宽：

```cpp
double bandwidth_gbps =
    static_cast<double>(3) * n * sizeof(float)
    / (milliseconds * 1e-3)
    / 1e9;
```

### 矩阵乘计算吞吐

对于：

```text
C[M, N] = A[M, K] × B[K, N]
```

浮点运算量近似为：

```text
2 × M × N × K FLOPs
```

吞吐量：

```cpp
double tflops =
    2.0 * M * N * K
    / (milliseconds * 1e-3)
    / 1e12;
```

### Nsight Compute 分析顺序

基础分析：

```bash
ncu --set basic ./program
```

完整分析：

```bash
ncu --set full ./program
```

只分析指定 Kernel：

```bash
ncu \
    --kernel-name regex:kernel_name \
    --set full \
    ./program
```

导出报告：

```bash
ncu --export result.ncu-rep ./program
```

优先查看：

```text
1. SpeedOfLight
2. LaunchStats
3. MemoryWorkloadAnalysis
4. ComputeWorkloadAnalysis
5. Occupancy
6. SchedulerStats
7. WarpStateStats
8. SourceCounters
```

需要回答：

* Kernel 是 Memory Bound 还是 Compute Bound？
* DRAM 带宽利用率是多少？
* SM 计算利用率是多少？
* Global Memory 是否合并访问？
* 是否存在 Shared Memory Bank Conflict？
* Register 使用是否限制 Occupancy？
* Shared Memory 使用是否限制 Block 数量？
* Warp 是否因为内存依赖而等待？
* Warp 是否因为同步而等待？
* 优化后减少了哪些内存访问？
* 优化后是否增加了 Register Pressure？
* 性能提升是否来自真实优化，而不是测试误差？

### 结果记录

```markdown
## 性能结果

| Version | Input | Block | Time | Bandwidth/GFLOP/s | Correct |
|---|---:|---:|---:|---:|---:|
| v0 | | | | | |
| v1 | | | | | |
| v2 | | | | | |

## Nsight Compute 结果

### v0

### v1

### v2

## 优化结论

## 性能提升原因

## 尚未解决的问题
```

---

# 五步闭环完成标准

一个 Sample 只有满足下面条件，才算真正完成：

* 能解释它解决的问题。
* 能画出 Host、Device 和 Kernel 的数据流。
* 能解释 Grid、Block 和线程映射。
* 能独立实现一个正确版本。
* 能实现至少一个不同版本。
* 能覆盖非规整输入尺寸。
* 能使用 Compute Sanitizer 检查错误。
* 能使用 CUDA Event 完成 Benchmark。
* 能计算有效带宽或计算吞吐。
* 能使用 Nsight Compute 找到主要瓶颈。
* 能解释优化为什么有效或为什么无效。
* 能将结论整理到 README 中。

完整闭环：

```text
阅读问题
    ↓
运行官方版本
    ↓
画执行模型
    ↓
独立重新实现
    ↓
构造变体和性能分析
    ↓
形成自己的实验结论
```

核心原则：

> 官方 Sample 用来提供正确参考，自己的代码、实验和性能分析才是最终的学习成果。