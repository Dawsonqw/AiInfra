"""
Online Safe Softmax — Triton 练习 stub
========================================
参考 kernel.py 完成下面的 TODO：
  1. 补全 online_safe_softmax_kernel（Triton JIT kernel）
  2. 补全 softmax wrapper 函数
  3. 在 main 中验证正确性和数值稳定性

算法提示
--------
Online safe softmax 每个 program 处理一行：
  1. 加载一行：tl.load（越界填 -inf）
  2. 求行最大值：tl.max(x, axis=0)
  3. 计算 exp(x - max)
  4. 求 exp 之和：tl.sum(exp_x, axis=0)
  5. 归一化并写回：tl.store
"""

import torch
import triton
import triton.language as tl


@triton.jit
def online_safe_softmax_kernel(
    X_ptr,
    Y_ptr,
    M,
    D,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < D

    # TODO: 加载一行（越界用 -inf 填充）
    x = tl.load(X_ptr + row * D + offsets, mask=mask, other=0.0)

    # TODO: 实现 online safe softmax
    # Hint:
    #   x_max  = tl.max(x, axis=0)
    #   exp_x  = tl.exp(x - x_max)
    #   exp_sum = tl.sum(exp_x, axis=0)
    #   y = exp_x / exp_sum
    y = x  # 占位，替换为正确实现

    # TODO: 写回结果（越界不写）
    tl.store(Y_ptr + row * D + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    对 2D cuda float32 tensor 按行做 softmax。
    TODO: 补全 BLOCK 计算、grid 设置、kernel 调用。
    """
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.float32
    M, D = x.shape
    y = torch.empty_like(x)

    # TODO: 计算 BLOCK（提示：triton.next_power_of_2(D)）
    BLOCK = D  # 暂时不对，需要改为 2 的幂

    # TODO: 设置 grid 并调用 kernel
    # online_safe_softmax_kernel[grid](x, y, M, D, BLOCK=BLOCK)

    return y  # 暂时返回原始 x（未实现），实现后删除此行并返回 y


if __name__ == "__main__":
    torch.manual_seed(0)

    # TODO: 验证小矩阵（M=4, D=8）
    M, D = 4, 8
    x = torch.randn(M, D, device="cuda", dtype=torch.float32)
    y_mine = softmax(x)
    y_ref = torch.softmax(x, dim=-1)
    max_err = (y_mine - y_ref).abs().max().item()
    print(f"[Small M={M} D={D}] max_err={max_err:.2e}  {'PASS' if max_err < 1e-5 else 'FAIL (未实现?)'}")

    # TODO: 数值稳定性测试（构造 [0,...,0,80]）
    x_stab = torch.zeros(1, 8, device="cuda", dtype=torch.float32)
    x_stab[0, -1] = 80.0
    y_stab = softmax(x_stab)
    y_stab_ref = torch.softmax(x_stab, dim=-1)
    print(f"\n--- 数值稳定性测试 ---")
    print(f"  output[7] = {y_stab[0, -1].item():.6f}  (expected ~{y_stab_ref[0, -1].item():.6f})")

    # TODO: 大矩阵验证（M=1024, D=256）
    M, D = 1024, 256
    x_large = torch.randn(M, D, device="cuda", dtype=torch.float32)
    y_large = softmax(x_large)
    y_large_ref = torch.softmax(x_large, dim=-1)
    max_err_large = (y_large - y_large_ref).abs().max().item()
    print(f"\n[Large M={M} D={D}] max_err={max_err_large:.2e}  {'PASS' if max_err_large < 1e-5 else 'FAIL (未实现?)'}")
