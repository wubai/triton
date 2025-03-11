import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  x_stride_0, y_stride_0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    a_offsets = a_ptr + pid * x_stride_0 + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_offsets)
    for i in tl.range(N):
        b_offsets = b_ptr + tl.arange(0, BLOCK_SIZE) * y_stride_0 + i
        b = tl.load(b_offsets)
        out = tl.sum(a * b)
        c_offset = c_ptr + pid * N + i
        tl.store(c_offset, out)


def matmul(A, B):
    rows_x, cols_x = A.shape
    rows_y, cols_y = B.shape
    assert cols_x == rows_y

    C = torch.empty((rows_x, cols_y), device='cuda')

    x_stride_0 = A.stride(0)
    y_stride_0 = B.stride(0)

    M = rows_x
    N = cols_y
    K = cols_x
    BLOCK_SIZE = triton.next_power_of_2(cols_x)
    grid = lambda meta:(M,)
    matmul_kernel[grid](A, B, C, M, N, K, x_stride_0, y_stride_0, BLOCK_SIZE=BLOCK_SIZE)
    return C



torch.manual_seed(42)
a = torch.randn([128, 128], device='cuda')
b = torch.randn([128, 128], device='cuda')
ref = torch.matmul(a, b)
triton_output = matmul(a, b)
if torch.allclose(triton_output, ref, atol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


