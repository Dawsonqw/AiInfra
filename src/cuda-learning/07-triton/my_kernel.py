"""
练习：用 Triton 实现 Fused Softmax + Matmul

目标：体验 Triton 高层抽象 vs CUDA C++ 底层控制的差异
  - Triton：tile 级编程，编译器负责 shared memory / warp 调度
  - CUDA C++：线程级编程，程序员手动管理 __shared__ / __syncthreads__

运行参考实现看输出：
    python kernel.py
"""

import torch
import triton
import triton.language as tl


# =========================================================================
# TODO Section 1: Fused Softmax
#
# 每个 program 处理矩阵的一行，在 SRAM 内完成 max/exp/sum，
# 不像 PyTorch 那样写 3 次 HBM。
#
# 提示：
#   1. tl.program_id(0) 获取行索引
#   2. tl.arange(0, BLOCK_N) 构造列偏移
#   3. tl.load(..., mask=mask, other=-inf) 加载一行
#   4. tl.max(x, axis=0) 求行最大值（数值稳定）
#   5. tl.exp, tl.sum 完成 softmax
#   6. tl.store 写回结果
# =========================================================================

@triton.jit
def online_safe_softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    # TODO
    pass


def fused_softmax(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty_like(X)
    BLOCK_N = triton.next_power_of_2(N)
    # TODO: 调用 online_safe_softmax_kernel
    return Y


# =========================================================================
# TODO Section 2: Matmul (fp16 Tensor Core)
#
# 复用 Ch6 的 matmul_kernel 结构，但：
#   - 输入为 fp16（A, B 的 dtype = torch.float16）
#   - acc 用 fp32（tl.zeros((BM, BN), dtype=tl.float32)）避免溢出
#   - tl.dot(a, b, out_dtype=tl.float32) 触发 HMMA 指令
#   - 写回前转回 fp16：acc.to(tl.float16)
#
# sm_89 Tensor Core 峰值（fp16）约是 fp32 CUDA Core 的 8 倍。
# =========================================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # TODO
    pass


def matmul_fp16(A: torch.Tensor, B: torch.Tensor,
                BM: int = 64, BN: int = 64, BK: int = 32) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), device=A.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # TODO: 调用 matmul_kernel
    return C


def main():
    print("=== Fused Softmax ===")
    M, N = 1024, 1024
    X = torch.randn(M, N, device="cuda", dtype=torch.float32)
    Y_triton = fused_softmax(X)
    Y_ref    = torch.softmax(X, dim=-1)
    err = (Y_triton - Y_ref).abs().max().item()
    print(f"  max_err = {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    print("\n=== Matmul (fp16, Tensor Core) ===")
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C_triton = matmul_fp16(A, B)
    C_ref    = torch.matmul(A, B)
    err = (C_triton.float() - C_ref.float()).abs().max().item()
    print(f"  max_err = {err:.2e}  {'PASS' if err < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
