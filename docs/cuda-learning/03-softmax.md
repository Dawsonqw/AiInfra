# Chapter 3: Online Safe Softmax — Warp Reduce

## 目标

实现按行 Softmax：`y_i = exp(x_i) / sum(exp(x_j))`，理解 GPU 上的归约操作。

## 要解决的问题

- 为什么 Naive Softmax 在大值输入下会 NaN？
- Safe Softmax（减 max）解决了溢出，但需要遍历两次 x——能不能一次？
- 32 个线程的 warp 内如何做 reduce？跨 warp 怎么办？

## 三种算法对比

| 算法 | 遍历次数 | 大值稳定性 |
|------|---------|-----------|
| Naive | 2 (sum + write) | 溢出 → NaN |
| Safe | 3 (max + sum + write) | 稳定 |
| Online Safe | 1 | 稳定 |

**Online Safe Softmax 的数学**：维护 running `(m, d)`：

- `m` = 当前遇到的最大值
- `d` = 以当前 m 为基准的 exp 和
- 遇到新值 v 时：`m_new = max(m, v)`，`d_new = d * exp(m - m_new) + exp(v - m_new)`

FlashAttention 里用的就是这个技巧。

## 核心概念

- **Warp Shuffle**：`__shfl_xor_sync(mask, val, offset)` — warp 内线程间零延迟交换寄存器值
- **Warp Reduce**：O(log N) 蝴蝶归约，所有结果汇聚到 lane 0
- **Block Reduce**：warp reduce → shared memory 跨 warp → 再 warp reduce
- **`__syncthreads()`**：block 级屏障，确保所有线程都写完了 shared memory

## CUDA C++ 要点

1. 先实现 `warp_reduce_sum` 和 `warp_reduce_max`（纯寄存器，O(log WARP_SIZE)）
2. 再用它们实现 `block_reduce_sum` 和 `block_reduce_max`（warp + smem）
3. Online Softmax 的难点是 **跨线程 merge `(m, d)`**—需要自定义 merge 操作
4. 参考 LeetCUDA 的 `warp_reduce_md_op`（`src/Reference/LeetCUDA/kernels/softmax/`）

## Triton 要点

- `tl.max(vals, axis=0)` 和 `tl.sum(vals, axis=0)` 自动做 reduce
- 维护 running `(m, d)` 的代码和 CUDA 逻辑一样，但不用手写 warp shuffle
- 可以把它做成 Fused Softmax（一次遍历完成，PyTorch 版本需要 4 个 kernel）

## 验收标准

- 正常输入：与 `torch.softmax` 误差 < 1e-5
- 大值输入（×80）：Naive 版 NaN，Safe/Online 版正确
- D <= 32（一个 warp 内）、D > 1024（跨 warp）都要测
