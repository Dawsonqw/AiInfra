"""
矩阵转置练习 —— Triton stub
=============================
参考 kernel.py 理解原理，然后在此独立实现。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mat_transpose_kernel(
    X_ptr, Y_ptr,
    M, N,
    TILE: tl.constexpr,
):
    """
    TODO: 实现矩阵转置 kernel。

    提示：
    1. 用 tl.program_id(0/1) 获取 tile 的行/列索引
    2. 构造行/列偏移 rm, rn（各长度 TILE 的向量）
    3. 用 rm[:, None] 和 rn[None, :] 构造 2D 索引，加载 X 的 TILE×TILE 块
    4. 注意边界 mask：(rm[:, None] < M) & (rn[None, :] < N)
    5. 用 tl.trans(x) 转置加载的块
    6. 写入 Y：行列互换（Y[rn, rm]），注意 Y 的 stride = M
    7. 写入 mask：(rn[:, None] < N) & (rm[None, :] < M)
    """
    # TODO: 实现 kernel
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # rm = ...
    # rn = ...
    # mask = ...
    # x = tl.load(...)
    # mask_t = ...
    # tl.store(...)
    pass


def mat_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    TODO: 实现 wrapper，调用 kernel 完成转置。

    提示：
    - 输出 shape = (N, M)，dtype 与输入相同
    - TILE = 32
    - grid = (ceil(M/TILE), ceil(N/TILE))
    """
    assert x.ndim == 2
    M, N = x.shape
    # TODO: 创建输出张量 y，配置 grid，调用 kernel
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    # ...
    return y


def main():
    M, N = 1024, 1024
    x = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)

    y = mat_transpose(x)
    expected = x.T.contiguous()

    if torch.equal(y, expected):
        print(f"mat_transpose ({M}x{N} -> {N}x{M}): PASS")
    else:
        diff = (y - expected).abs().max().item()
        print(f"mat_transpose ({M}x{N} -> {N}x{M}): FAIL  max_diff={diff}")


if __name__ == "__main__":
    main()
