# Chapter 5: Mat Transpose — Shared Memory 与 Bank Conflict

## 目标

实现矩阵转置 `B[j][i] = A[i][j]`，深入理解 shared memory 的 bank 结构，学会用 padding 消除 bank conflict。

参考：`LeetCUDA/kernels/mat-transpose/mat_transpose.cu`

## 要解决的问题

- 矩阵转置写全局内存时是 **strided access**（步长 = 行数），为什么这是性能杀手？
- Shared memory 内部分成 32 个 bank，什么情况会触发 bank conflict？
- 加一个 `PAD=1` 的 padding 为什么就能消除 bank conflict？

## 核心概念

### 全局内存的读写不对称

矩阵转置的本质：读是行优先（coalesced），写是列优先（strided）。或者反过来：写是行优先（coalesced），读是列优先（strided）。不管怎么做，**总有一侧访存是非连续的**，直接用全局内存无法同时 coalesced。

解决方案：用 shared memory 做缓冲，先 coalesced 写入 smem，再 coalesced 从 smem 读出写全局内存。

```
全局内存 A（行优先读）→ smem tile → 全局内存 B（列优先读转换成行优先写）
```

### Shared Memory Bank 结构

RTX 40 系的 shared memory 分成 **32 个 bank**，每个 bank 宽度 4 字节，每隔 128 字节循环一次。

地址映射：`bank_id = (address / 4) % 32`

**Bank conflict**：同一 warp 内多个线程访问 **同一 bank 不同地址** → 串行化。

典型场景：`float smem[32][32]`，列方向访问时 `smem[0][0]`、`smem[1][0]`、...、`smem[31][0]` 全部落在 bank 0 → 32 路 bank conflict。

### Padding 消除 Bank Conflict

```cpp
__shared__ float smem[TILE][TILE + 1];  // +1 padding
```

加 1 列 padding 后，`smem[row][0]` 的 bank_id = `(row * (TILE+1)) % 32`，每行错开 1，不再全撞 bank 0。

## CUDA C++ 实现路径

| 版本 | 技术点 |
|------|--------|
| `f32_col2row` | 最简版，直接转置，一侧 strided |
| `f32_row2col` | 另一方向，coalesced 读 strided 写 |
| `f32x4_col2row` | float4 向量化，但仍有 bank conflict |
| `f32x4_shared_col2row` | 引入 shared memory tile，两侧 coalesced |
| `f32x4_shared_bcf_col2row` | +1 padding，消除 bank conflict |

关键代码骨架（shared 版）：

```cpp
__shared__ float smem[TILE][TILE];  // 无 padding 版
// 每个线程：全局→smem (coalesced 读)
smem[threadIdx.y][threadIdx.x] = A[row * N + col];
__syncthreads();
// 交换 x/y，实现转置：smem→全局 (coalesced 写)
B[col_out * M + row_out] = smem[threadIdx.x][threadIdx.y];
```

## Triton 要点

- `tl.load` 加载一个 tile，`tl.store` 写出时交换行列偏移即可实现转置
- Triton 编译器自动处理 shared memory 和 bank conflict

## 验收标准

- 转置结果与 `A.T` 完全一致（整数精度，无误差）
- shared_bcf 版比无 shared 版快 > 3x（M=N=4096）
- shared_bcf 版比 shared 无 padding 版快 > 1.5x（验证 padding 效果）
- 测试非方阵（M≠N）的边界处理
