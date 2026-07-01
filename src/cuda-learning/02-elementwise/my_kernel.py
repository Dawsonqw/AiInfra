import torch
import triton
import triton.language as tl


# TODO: 实现 elementwise_add_kernel
# 要求：
#   - 每个 program 处理 BLOCK 个连续元素
#   - 用 tl.program_id(0) 计算起始偏移
#   - 用 mask 防止越界读写
@triton.jit
def elementwise_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK: tl.constexpr):
    # 你的代码
    pass


def elementwise_add(a: torch.Tensor, b: torch.Tensor,
                     BLOCK: int = 1024) -> torch.Tensor:
    # TODO: 实现 wrapper
    #   1. 断言 a, b 形状相同且都在 GPU 上
    #   2. 计算 grid = (N + BLOCK - 1) // BLOCK（或用 triton.cdiv）
    #   3. 启动 elementwise_add_kernel
    #   4. 返回结果 tensor
    pass


if __name__ == "__main__":
    N = 1 << 24
    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    # TODO: 调用你的实现并验证
    #   1. c = elementwise_add(a, b)
    #   2. 与 PyTorch 参考 a + b 比较
    #   3. 打印 PASS/FAIL 和 max error
    print("TODO")
