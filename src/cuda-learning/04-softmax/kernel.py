"""
Online Safe Softmax — Triton 参考实现
======================================
每个 program 处理矩阵的一行。
算法：tl.max + tl.exp + tl.sum，等价于 online safe softmax。

数值稳定性测试：构造一行 [0,...,0,80]，naive 版 NaN，本实现正确。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def online_safe_softmax_kernel(
    X_ptr,          # 输入矩阵，shape (M, D)，行主序
    Y_ptr,          # 输出矩阵，shape (M, D)
    M,              # 行数
    D,              # 列数（每行元素数）
    BLOCK: tl.constexpr,  # 编译时常量，必须 >= D 且为 2 的幂
):
    row = tl.program_id(0)   # 当前处理第几行
    offsets = tl.arange(0, BLOCK)
    mask = offsets < D

    # 加载一行；越界位置填 -inf，不影响 max/sum
    x = tl.load(X_ptr + row * D + offsets, mask=mask, other=-float("inf"))

    # Online safe softmax：一次内核完成
    x_max = tl.max(x, axis=0)          # 行最大值（数值稳定基准）
    exp_x = tl.exp(x - x_max)          # shift 后不会溢出
    # 越界位置 exp(-inf) = 0，不影响求和
    exp_sum = tl.sum(exp_x, axis=0)    # 归一化分母

    y = exp_x / exp_sum

    # 越界位置不写出
    tl.store(Y_ptr + row * D + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    对 2D tensor 按行做 softmax。
    x: (M, D)，cuda float32
    返回同形状 tensor。
    """
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.float32
    M, D = x.shape

    # BLOCK 必须是 2 的幂且 >= D
    BLOCK = triton.next_power_of_2(D)

    y = torch.empty_like(x)
    grid = (M,)
    online_safe_softmax_kernel[grid](x, y, M, D, BLOCK=BLOCK)
    return y


# ============================================================
# main：验证 + 数值稳定性测试 + 计时
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # ----------------------------------------------------------
    # 测试 1：小矩阵正确性（M=4, D=8）
    # ----------------------------------------------------------
    M, D = 4, 8
    x = torch.randn(M, D, device="cuda", dtype=torch.float32)
    y_triton = softmax(x)
    y_ref = torch.softmax(x, dim=-1)
    max_err = (y_triton - y_ref).abs().max().item()
    status = "PASS" if max_err < 1e-5 else "FAIL"
    print(f"[Small M={M} D={D}] max_err={max_err:.2e}  {status}")

    # ----------------------------------------------------------
    # 测试 2：数值稳定性（行末尾有极大值 80）
    # ----------------------------------------------------------
    x_stab = torch.zeros(1, 8, device="cuda", dtype=torch.float32)
    x_stab[0, -1] = 80.0
    y_stab = softmax(x_stab)
    y_stab_ref = torch.softmax(x_stab, dim=-1)
    max_err_stab = (y_stab - y_stab_ref).abs().max().item()
    status_stab = "PASS" if max_err_stab < 1e-5 else "FAIL"
    print(f"\n--- 数值稳定性测试 ---")
    print(f"  input     = {x_stab[0].tolist()}")
    print(f"  Triton[7] = {y_stab[0, -1].item():.6f}  (expected ~{y_stab_ref[0, -1].item():.6f})")
    print(f"  max_err   = {max_err_stab:.2e}  {status_stab}")

    # ----------------------------------------------------------
    # 测试 3：大矩阵正确性（M=1024, D=256）
    # ----------------------------------------------------------
    M, D = 1024, 256
    x_large = torch.randn(M, D, device="cuda", dtype=torch.float32)
    y_triton_large = softmax(x_large)
    y_ref_large = torch.softmax(x_large, dim=-1)
    max_err_large = (y_triton_large - y_ref_large).abs().max().item()
    status_large = "PASS" if max_err_large < 1e-5 else "FAIL"
    print(f"\n[Large M={M} D={D}] max_err={max_err_large:.2e}  {status_large}")

    # ----------------------------------------------------------
    # 计时（warmup + timed runs）
    # ----------------------------------------------------------
    print(f"\n--- 计时 M={M} D={D} ---")
    # warmup
    for _ in range(10):
        softmax(x_large)
    torch.cuda.synchronize()

    import time
    ITERS = 200
    t0 = time.perf_counter()
    for _ in range(ITERS):
        softmax(x_large)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_us = (t1 - t0) / ITERS * 1e6
    print(f"Triton online_safe_softmax  avg {avg_us:.1f} us / call")

    # PyTorch baseline
    t0 = time.perf_counter()
    for _ in range(ITERS):
        torch.softmax(x_large, dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pt_us = (t1 - t0) / ITERS * 1e6
    print(f"PyTorch torch.softmax       avg {pt_us:.1f} us / call")
