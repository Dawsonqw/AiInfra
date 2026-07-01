# Chapter 3: Block All Reduce — Warp Shuffle 与 Shared Memory

## 目标

实现 `y = sum(A)`（对整个向量求和），掌握 GPU 上线程协作的两大原语：warp shuffle 和 shared memory。这两个原语是 Softmax、LayerNorm、Attention 等几乎所有归约型算子的基础。

参考：`LeetCUDA/kernels/reduce/block_all_reduce.cu`

## 要解决的问题

- Elementwise 每个线程独立工作，但 reduce 需要线程间通信——怎么做？
- Warp 内 32 个线程共享寄存器，能绕过 shared memory 直接通信吗？
- 超过一个 warp（>32 线程）时，如何跨 warp 汇总结果？

## 核心概念

### Warp Shuffle（`__shfl_xor_sync`）

同一 warp 内的 32 个线程可以直接交换寄存器值，**零延迟，无需 shared memory**。

```
Lane:  0   1   2   3  ... 15  16  17  18  ...31
       ↕   ↕   ↕   ↕       ↕   ↕   ↕   ↕
mask=16: 0↔16, 1↔17, 2↔18, ...（butterfly）
mask=8:  0↔8,  1↔9,  ...
mask=4:  0↔4,  ...
mask=2:  0↔2,  ...
mask=1:  0↔1,  ...
```

经过 `log2(32) = 5` 轮 XOR，lane 0 持有所有 32 个元素的 sum。

```cpp
template <const int kWarpSize = 32>
__device__ float warp_reduce_sum(float val) {
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;  // 每个 lane 都持有最终 sum
}
```

### Block Reduce（warp reduce → shared memory → 再 warp reduce）

Block 最多 1024 个线程 = 最多 32 个 warp。步骤：

1. 每个 warp 内部做 `warp_reduce_sum`，warp leader（lane 0）得到 warp 小计
2. 32 个 warp leader 把小计写入 `__shared__ float smem[32]`
3. `__syncthreads()` 保证所有写入完成
4. 第一个 warp 从 smem 读出 32 个值再做一次 `warp_reduce_sum`
5. thread 0 得到最终结果，用 `atomicAdd` 累加到全局输出

### `__syncthreads()` 的语义

Block 级屏障：所有线程都到达该点之后才继续执行。用于：
- 确保 shared memory 写入对所有线程可见
- 防止线程领先覆写还在被其他线程读的 smem

## CUDA C++ 实现路径

| 版本 | 技术点 |
|------|--------|
| `f32` 标量 | warp reduce + block reduce 基础框架 |
| `f32x4` | float4 向量化加载，每线程先本地累加 4 个元素 |
| `f16` / `f16x2` | FP16 acc 或 FP32 acc，体会精度差异 |
| `f16x8_pack` | 128-bit 加载 + FP32 acc |

## Triton 要点

- `tl.sum(vals, axis=0)` 自动做 block 级 reduce
- 只需关注 tile 的划分，warp shuffle / smem 由编译器处理

## 验收标准

- f32 版与 `torch.sum` 误差 < 1e-4（大 N 时浮点误差会累积，适当放宽）
- f32x4 版比 f32 标量版快 > 1.5x（N=32M）
- f16 版（fp32 acc）与 f32 版结果相近，f16 acc 版允许稍大误差
- 测试 N 不是 blockDim 整数倍的边界情况
