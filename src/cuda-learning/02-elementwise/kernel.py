"""
第 02 章：Elementwise 操作 — Triton 参考实现

两个 kernel：
  elementwise_add_kernel      — f32 向量加法
  elementwise_add_relu_kernel — 向量加法 + ReLU

Triton 会自动向量化 tl.load/tl.store（生成 LDG.128），
无需像 CUDA C++ 那样手动用 float4。
"""

import time

import torch
import triton
import triton.language as tl


# ============================================================
# Kernel 1: elementwise add（f32）
#   每个 program 处理 BLOCK 个连续元素
# ============================================================
@triton.jit
def elementwise_add_kernel(
    A_ptr,            # 输入 A 的显存地址
    B_ptr,            # 输入 B 的显存地址
    C_ptr,            # 输出 C 的显存地址
    N,                # 元素总数
    BLOCK: tl.constexpr,  # 每个 program 处理的元素数（编译时常量）
):
    pid = tl.program_id(0)                        # 当前 program 的编号
    offsets = pid * BLOCK + tl.arange(0, BLOCK)   # 本 program 负责的索引
    mask = offsets < N                            # 越界位置不读写

    a = tl.load(A_ptr + offsets, mask=mask)       # 向量化加载，自动发 LDG.128
    b = tl.load(B_ptr + offsets, mask=mask)

    c = a + b

    tl.store(C_ptr + offsets, c, mask=mask)       # 向量化写回


# ============================================================
# Kernel 2: elementwise add + ReLU
#   在加法结果上做 tl.maximum(0.0, c)，即 ReLU
#   tl.maximum 会生成 FMAX 指令，与 CUDA fmaxf 等价
# ============================================================
@triton.jit
def elementwise_add_relu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    c = tl.maximum(0.0, a + b)   # ReLU: max(0, a+b)

    tl.store(C_ptr + offsets, c, mask=mask)


# ============================================================
# Python 包装函数
# ============================================================

def elementwise_add(a: torch.Tensor, b: torch.Tensor,
                     BLOCK: int = 1024) -> torch.Tensor:
    """elementwise_add_kernel 的 Python wrapper"""
    assert a.shape == b.shape and a.is_cuda and b.is_cuda
    N = a.numel()
    c = torch.empty_like(a)
    grid = (triton.cdiv(N, BLOCK),)
    elementwise_add_kernel[grid](a, b, c, N, BLOCK=BLOCK)
    return c


def elementwise_add_relu(a: torch.Tensor, b: torch.Tensor,
                          BLOCK: int = 1024) -> torch.Tensor:
    """elementwise_add_relu_kernel 的 Python wrapper"""
    assert a.shape == b.shape and a.is_cuda and b.is_cuda
    N = a.numel()
    c = torch.empty_like(a)
    grid = (triton.cdiv(N, BLOCK),)
    elementwise_add_relu_kernel[grid](a, b, c, N, BLOCK=BLOCK)
    return c


# ============================================================
# 简单计时辅助（兼容没有 triton.testing 的环境）
# ============================================================
def bench(fn, warmup=5, rep=20):
    """对 fn() 做 warmup + rep 次计时，返回平均毫秒数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / rep * 1000.0   # ms


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    N = 1 << 24  # 16M 个 float

    # 随机初始化，部分值为负（方便观察 ReLU 效果）
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    # ---- 测试 kernel 1: elementwise_add ----
    c_triton = elementwise_add(a, b)
    c_ref = a + b

    max_err = (c_triton - c_ref).abs().max().item()
    t_ms = bench(lambda: elementwise_add(a, b))
    status = "PASS" if max_err < 1e-6 else "FAIL"
    print(f"elementwise_add      : time={t_ms:.3f}ms  {status}  (max_err={max_err:.2e})")

    # ---- 测试 kernel 2: elementwise_add_relu ----
    c_relu_triton = elementwise_add_relu(a, b)
    c_relu_ref = torch.clamp(a + b, min=0.0)   # PyTorch 参考

    max_err_relu = (c_relu_triton - c_relu_ref).abs().max().item()
    t_relu_ms = bench(lambda: elementwise_add_relu(a, b))
    status_relu = "PASS" if max_err_relu < 1e-6 else "FAIL"
    print(f"elementwise_add_relu : time={t_relu_ms:.3f}ms  {status_relu}  (max_err={max_err_relu:.2e})")
