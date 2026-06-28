"""
Vector Add — Triton 参考实现
    每个 program 处理 BLOCK 个元素
    C[i] = A[i] + B[i]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    A_ptr,          # A 的显存地址（由 torch Tensor 自动传入）
    B_ptr,          # B 的显存地址
    C_ptr,          # C 的显存地址（输出）
    N,              # 元素总数
    BLOCK: tl.constexpr,  # 编译时常量：每个 program 处理的元素数
):
    # ---- 计算当前 program 负责的索引范围 ----
    pid = tl.program_id(0)                        # 等价于 CUDA 的 blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)   # 长度为 BLOCK 的数组，如 [256, 257, ..., 511]
    mask = offsets < N                            # 越界位置标记为 False

    # ---- 从显存加载 ----
    a = tl.load(A_ptr + offsets, mask=mask)       # tl.load 自动做 128-bit 向量化
    b = tl.load(B_ptr + offsets, mask=mask)

    # ---- 计算 ----
    c = a + b

    # ---- 写回显存 ----
    tl.store(C_ptr + offsets, c, mask=mask)       # mask 内的位置才真正写入


def vector_add(a: torch.Tensor, b: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    """包装函数：算 grid → 调 kernel → 返回结果"""
    assert a.shape == b.shape and a.is_cuda and b.is_cuda
    N = a.numel()
    c = torch.empty_like(a)

    # grid 控制有多少个 program 并行。向上取整。
    grid = (triton.cdiv(N, BLOCK),)   # triton.cdiv = (N + BLOCK - 1) // BLOCK

    # 启动 kernel。BLOCK 是 constexpr 参数，必须用关键字传。
    vector_add_kernel[grid](a, b, c, N, BLOCK=BLOCK)

    return c


if __name__ == "__main__":
    N = 256
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    c_triton = vector_add(a, b)
    c_ref = a + b

    max_err = (c_triton - c_ref).abs().max().item()
    print(f"Vector Add Triton — N={N}")
    print(f"  Max error: {max_err:.2e}")
    print("  PASS" if max_err < 1e-6 else "  FAIL")
