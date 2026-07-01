"""
练习：用 Triton 实现 GEMM（FP32）

目标：理解 Triton program_id、tile 索引、tl.dot 的使用方式。

运行参考实现看输出：
    python kernel.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # TODO: 计算当前 program 负责的行/列起始索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # TODO: 构造行/列/K 方向的偏移向量（tl.arange）
    # rm = ...   shape [BM]
    # rn = ...   shape [BN]
    # rk = ...   shape [BK]

    # TODO: 初始化累加器 acc = tl.zeros((BM, BN), dtype=tl.float32)

    # TODO: K 方向分块循环
    # for k in range(0, K, BK):
    #     a = tl.load(...)   # [BM, BK]，加边界 mask
    #     b = tl.load(...)   # [BK, BN]，加边界 mask
    #     acc += tl.dot(a, b)

    # TODO: 写回 C（加边界 mask）
    pass


def matmul(A: torch.Tensor, B: torch.Tensor,
           BM: int = 32, BN: int = 32, BK: int = 32) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # TODO: 调用 matmul_kernel
    return C


def main():
    M = K = N = 512
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    C_triton = matmul(A, B)
    C_ref    = torch.matmul(A, B)
    max_err  = (C_triton - C_ref).abs().max().item()
    print(f"max_err = {max_err:.2e}  {'PASS' if max_err < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()
