"""
Ch7: Triton 综合演示 —— Fused Softmax + Matmul（fp16 Tensor Core）

本文件展示 Triton 相对于 CUDA C++ 的核心优势：
  1. 高层抽象（tile 级编程）替代线程级编程
  2. 自动选择 Tensor Core 路径（fp16 输入时 tl.dot → HMMA 指令）
  3. Kernel fusion：多个 GPU 操作合并为一个 kernel，消除中间 buffer

=== Section 1: Fused Softmax ===
PyTorch 的 torch.softmax 需要 3 个 kernel：
  max reduction → exp → sum reduction → div
Triton 用一个 kernel 完成所有步骤（在 SRAM 中完成中间结果，不写 HBM）。
这对带宽密集型操作（softmax 是 memory-bound）效果显著。

=== Section 2: Matmul with Tensor Core ===
fp16 输入时 tl.dot 自动生成 HMMA 指令（Tensor Core）。
RTX 4060 Laptop (sm_89) fp16 Tensor Core 峰值 ~100+ TFLOPS，
远超 fp32 CUDA Core (~15 TFLOPS)。
"""

import torch
import triton
import triton.language as tl
import time


# =========================================================================
# Section 1: Fused Softmax
# 直接复用 Ch4 的 online safe softmax 实现（逐行展开，一个 program 处理一行）
# =========================================================================

@triton.jit
def online_safe_softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    """
    每个 program 处理输入矩阵的一行。
    Online（单 pass）safe softmax 算法：
      pass1: 同时求 max 和 sum*exp，一次扫描完成。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载一行数据
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=-float('inf'))

    # 数值稳定：减去最大值
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max

    # exp 和归一化
    exp_x = tl.exp(x_shifted)
    row_sum = tl.sum(exp_x, axis=0)
    y = exp_x / row_sum

    # 写回
    tl.store(Y_ptr + row * N + cols, y, mask=mask)


def fused_softmax(X: torch.Tensor) -> torch.Tensor:
    """对矩阵每行做 softmax（fused，单 kernel）。"""
    assert X.ndim == 2
    M, N = X.shape
    Y = torch.empty_like(X)
    # BLOCK_N 需为 2 的幂且 >= N；此处简单取最近的 2 的幂
    BLOCK_N = triton.next_power_of_2(N)
    online_safe_softmax_kernel[(M,)](X, Y, M, N, BLOCK_N=BLOCK_N)
    return Y


# =========================================================================
# Section 2: Matmul (fp16) with Tensor Core
# 使用 Ch6 的 matmul_kernel，输入改为 fp16。
# fp16 路径：tl.dot 自动调用 HMMA 指令（Tensor Core）。
# =========================================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    # fp16 累加器使用 fp32 避免精度损失
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K, BK):
        a = tl.load(
            A_ptr + rm[:, None] * K + (k + rk)[None, :],
            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + (k + rk)[:, None] * N + rn[None, :],
            mask=((k + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        # fp16 输入时，tl.dot 自动使用 HMMA 指令（Tensor Core）
        # 累加到 fp32 acc 避免溢出
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # 写回为 fp16
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        C_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.float16),
        mask=mask,
    )


def matmul_fp16(A: torch.Tensor, B: torch.Tensor,
                BM: int = 64, BN: int = 64, BK: int = 32) -> torch.Tensor:
    """fp16 矩阵乘法，利用 Tensor Core。"""
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](A, B, C, M, K, N, BM=BM, BN=BN, BK=BK)
    return C


# =========================================================================
# Benchmark 工具
# =========================================================================

def benchmark(fn, warmup: int = 5, rep: int = 20) -> float:
    """返回平均执行时间（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3


# =========================================================================
# Main
# =========================================================================

def main():
    # ------------------------------------------------------------------
    # Section 1: Fused Softmax
    # ------------------------------------------------------------------
    print("=== Fused Softmax ===")
    M, N = 1024, 1024
    X = torch.randn(M, N, device="cuda", dtype=torch.float32)

    Y_triton = fused_softmax(X)
    Y_torch  = torch.softmax(X, dim=-1)
    max_err  = (Y_triton - Y_torch).abs().max().item()

    ms_triton_sf = benchmark(lambda: fused_softmax(X))
    ms_torch_sf  = benchmark(lambda: torch.softmax(X, dim=-1))
    status = "PASS" if max_err < 1e-4 else "FAIL"
    print(f"  Triton: {ms_triton_sf:.3f}ms  torch: {ms_torch_sf:.3f}ms  {status}"
          f"  (max_err={max_err:.2e})")
    print(f"  说明：Triton 单 kernel 完成 max/exp/sum，"
          f"PyTorch 需 3 个 kernel，中间结果落 HBM")

    # ------------------------------------------------------------------
    # Section 2: Matmul (fp16, Tensor Core)
    # ------------------------------------------------------------------
    print("\n=== Matmul (fp16, Tensor Core) ===")
    Mf, Kf, Nf = 1024, 1024, 1024
    torch.manual_seed(42)
    A = torch.randn(Mf, Kf, device="cuda", dtype=torch.float16)
    B = torch.randn(Kf, Nf, device="cuda", dtype=torch.float16)

    C_triton = matmul_fp16(A, B)
    C_torch  = torch.matmul(A, B)
    max_err  = (C_triton.float() - C_torch.float()).abs().max().item()
    # fp16 精度误差阈值宽松些
    status = "PASS" if max_err < 1.0 else "FAIL"

    ms_triton_mm = benchmark(lambda: matmul_fp16(A, B))
    ms_torch_mm  = benchmark(lambda: torch.matmul(A, B))
    print(f"  Triton: {ms_triton_mm:.3f}ms  torch: {ms_torch_mm:.3f}ms  {status}"
          f"  (max_err={max_err:.2e})")
    print(f"  说明：fp16 输入 tl.dot → HMMA 指令（Tensor Core）"
          f"，理论 ~8x vs fp32 CUDA Core")


if __name__ == "__main__":
    main()
