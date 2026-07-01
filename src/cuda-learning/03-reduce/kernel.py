"""
03-reduce/kernel.py
Triton 实现：Block Reduce Sum

每个 Triton program 负责 BLOCK 个元素：
  1. tl.load 加载数据（带 mask 处理边界）
  2. tl.sum(chunk, axis=0) 在 BLOCK 内做 reduce
  3. tl.atomic_add 将小计累加到全局输出
"""

import time
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────
# Triton Kernel
# ─────────────────────────────────────────────────────────────
@triton.jit
def block_reduce_sum_kernel(
    A_ptr,              # 输入数组指针
    y_ptr,              # 输出标量指针（单个 float）
    N,                  # 元素个数
    BLOCK: tl.constexpr # 每个 program 处理的元素数（编译期常量）
):
    # 当前 program 在 grid 中的 id
    pid = tl.program_id(0)

    # 本 program 负责的起始偏移
    offset = pid * BLOCK + tl.arange(0, BLOCK)

    # mask：防止越界读取
    mask = offset < N

    # 从全局内存加载数据；越界位置用 0.0 填充（不影响 sum）
    chunk = tl.load(A_ptr + offset, mask=mask, other=0.0)

    # 在 BLOCK 维度上求和，得到标量
    local_sum = tl.sum(chunk, axis=0)

    # 原子加到全局输出（多个 program 并发执行）
    tl.atomic_add(y_ptr, local_sum)


# ─────────────────────────────────────────────────────────────
# Python Wrapper
# ─────────────────────────────────────────────────────────────
def block_reduce_sum(a: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    """
    对 1D CUDA tensor a 做全局 sum，返回包含结果的 scalar tensor。

    参数：
        a     : 1D float32 CUDA tensor
        BLOCK : 每个 Triton program 处理的元素数（建议为 2 的幂）
    """
    assert a.is_cuda and a.dtype == torch.float32, \
        "输入必须是 float32 CUDA tensor"
    assert a.ndim == 1, "输入必须是 1D tensor"

    N = a.numel()

    # 输出：初始化为 0，dtype 与输入一致
    y = torch.zeros(1, device=a.device, dtype=a.dtype)

    # grid：每个 program 处理 BLOCK 个元素
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    block_reduce_sum_kernel[grid](a, y, N, BLOCK=BLOCK)

    return y


# ─────────────────────────────────────────────────────────────
# 验证与计时
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 1 << 24  # 16M elements

    # 全 1 数组，期望 sum = N
    a = torch.ones(N, device="cuda", dtype=torch.float32)

    # ── Warmup ──────────────────────────────────────────────
    for _ in range(5):
        _ = block_reduce_sum(a)
    torch.cuda.synchronize()

    # ── 计时：Triton kernel ──────────────────────────────────
    ITERS = 20
    t0 = time.perf_counter()
    for _ in range(ITERS):
        result = block_reduce_sum(a)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) / ITERS * 1000.0
    result_val = result.item()

    # ── 参考值 ───────────────────────────────────────────────
    ref = a.sum().item()

    # ── 验证：允许累计浮点误差 < 1.0 ────────────────────────
    error = abs(result_val - ref)
    pass_flag = error < 1.0

    print(f"block_reduce_sum (Triton) : time={elapsed_ms:.3f}ms  "
          f"sum={result_val:.0f}  ref={ref:.0f}  err={error:.2f}  "
          f"{'PASS' if pass_flag else 'FAIL'}")
