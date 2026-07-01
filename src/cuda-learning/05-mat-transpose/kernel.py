"""
矩阵转置 —— Triton 实现
========================
演示 Triton 如何通过 tl.trans() 在 tile 级别完成转置，
框架自动处理 shared memory 和 bank conflict。
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
    将 M×N 矩阵 X 转置为 N×M 矩阵 Y。

    grid = (ceil(M/TILE), ceil(N/TILE))
    pid_m：tile 的行索引（对应 X 的行方向）
    pid_n：tile 的列索引（对应 X 的列方向）
    """
    pid_m = tl.program_id(0)  # tile 行
    pid_n = tl.program_id(1)  # tile 列

    # 当前 tile 在 X 中的行/列起始偏移
    rm = pid_m * TILE + tl.arange(0, TILE)  # [TILE]，X 的行索引
    rn = pid_n * TILE + tl.arange(0, TILE)  # [TILE]，X 的列索引

    # 加载 X[rm, rn]，shape = [TILE, TILE]，边界用 0 填充
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(X_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0)

    # tl.trans(x)：将 [TILE, TILE] 张量转置为 [TILE, TILE]（行列互换）
    # 写入 Y[rn, rm]，即 Y 的行 = X 的列，Y 的列 = X 的行
    # Y 的 shape = N×M，stride = M
    mask_t = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(
        Y_ptr + rn[:, None] * M + rm[None, :],
        tl.trans(x),
        mask=mask_t,
    )


def mat_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    对 2D 张量 x (M×N) 做转置，返回 N×M 张量。
    """
    assert x.ndim == 2, "仅支持 2D 矩阵"
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)

    TILE = 32
    grid = (triton.cdiv(M, TILE), triton.cdiv(N, TILE))

    mat_transpose_kernel[grid](
        x, y,
        M, N,
        TILE=TILE,
    )
    return y


def main():
    torch.manual_seed(0)
    M, N = 1024, 1024

    # 用整数值初始化，便于精确验证
    x = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)

    y = mat_transpose(x)

    # 验证：y 应等于 x.T（精确匹配）
    expected = x.T.contiguous()
    if torch.equal(y, expected):
        print(f"mat_transpose ({M}x{N} -> {N}x{M}): PASS")
    else:
        diff = (y - expected).abs().max().item()
        print(f"mat_transpose ({M}x{N} -> {N}x{M}): FAIL  max_diff={diff}")

    # 非方阵测试
    M2, N2 = 512, 768
    x2 = torch.randn(M2, N2, device="cuda")
    y2 = mat_transpose(x2)
    expected2 = x2.T.contiguous()
    if torch.allclose(y2, expected2, atol=1e-5):
        print(f"mat_transpose ({M2}x{N2} -> {N2}x{M2}): PASS")
    else:
        print(f"mat_transpose ({M2}x{N2} -> {N2}x{M2}): FAIL")


if __name__ == "__main__":
    main()
