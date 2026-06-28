import triton
import torch
import triton.language as tl

@triton.jit
def vector_add_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCKS:tl.constexpr
):
    pid=tl.program_id(0)
    offsets=pid*BLOCKS+tl.arange(0,BLOCKS)
    mask=offsets<N

    a=tl.load(A_ptr+offsets,mask=mask)
    b=tl.load(B_ptr+offsets,mask=mask)

    c=a+b

    tl.store(C_ptr+offsets,c,mask=mask)


def vector_add(a:torch.Tensor,b:torch.Tensor,BLOCK:int=1024)->torch.Tensor:
    assert a.shape==b.shape  and a.is_cuda and b.is_cuda
    N = a.numel()
    c=torch.empty_like(a)

    grid=(triton.cdiv(N,BLOCK),)
    vector_add_kernel[grid](a,b,c,N,BLOCK)

    return c


if __name__ == "__main__":
    N=256
    a=torch.randn(N,device="cuda",dtype=torch.float32)
    b=torch.randn(N,device="cuda",dtype=torch.float32)

    c_triton=vector_add(a,b)
    c_ref=a+b

    max_err=(c_triton-c_ref).abs().max().item()

    print(f"max_err: {max_err:.2e}")