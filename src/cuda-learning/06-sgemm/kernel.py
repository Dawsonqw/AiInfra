"""
SGEMM (FP32) with Triton
C = A @ B, M=K=N=512
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 当前 program 负责的行/列区间
    rm = pid_m * BM + tl.arange(0, BM)   # [BM]
    rn = pid_n * BN + tl.arange(0, BN)   # [BN]
    rk = tl.arange(0, BK)                # [BK]

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K, BK):
        # 加载 A 的 [BM, BK] tile，行 mask + 列 mask
        a = tl.load(
            A_ptr + rm[:, None] * K + (k + rk)[None, :],
            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K),
            other=0.0,
        )
        # 加载 B 的 [BK, BN] tile
        b = tl.load(
            B_ptr + (k + rk)[:, None] * N + rn[None, :],
            mask=((k + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        # tl.dot：FP32 矩阵乘，在 sm_89 上利用 FP32 Tensor Core（TF32 路径）
        acc += tl.dot(a, b)

    # 写回 C
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc, mask=mask)


def matmul(A: torch.Tensor, B: torch.Tensor,
           BM: int = 32, BN: int = 32, BK: int = 32) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Shape mismatch: A({M},{K}) @ B({K2},{N})"
    assert A.is_cuda and B.is_cuda

    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](A, B, C, M, K, N, BM=BM, BN=BN, BK=BK)
    return C


def benchmark(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3  # ms


def main():
    M = K = N = 512
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # 正确性验证
    C_triton = matmul(A, B)
    C_ref    = torch.matmul(A, B)
    max_err  = (C_triton - C_ref).abs().max().item()

    # 计时
    ms_triton = benchmark(lambda: matmul(A, B))
    ms_torch  = benchmark(lambda: torch.matmul(A, B))

    status = "PASS" if max_err < 1e-2 else "FAIL"
    print(f"Triton matmul : {ms_triton:.3f}ms  torch: {ms_torch:.3f}ms  "
          f"{status}  (max_err={max_err:.2e})")
    print(f"speedup vs torch: {ms_torch / ms_triton:.2f}x")


if __name__ == "__main__":
    main()
