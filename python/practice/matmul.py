import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, ouput_ptr,
                  M, N, K,
                  a_stride_0, b_stride_0, c_stride_0,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in tl.range(tl.cdiv(K, BLOCK_SIZE_K)):
        offsets_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offsets_am = a_ptr + (offset_am * a_stride_0)[:, None] + offsets_k[None, :]
        offsets_bn = b_ptr + (offsets_k * b_stride_0)[:, None] + offset_bn[None, :]
        a_mk = tl.load(offsets_am, mask=offsets_k[None, :] < K, other=0.0)
        b_kn = tl.load(offsets_bn, mask=offsets_k[:, None] < K, other=0.0)
        accumulator += tl.dot(a_mk, b_kn)
    c = accumulator.to(tl.float16)

    offsets_c = ouput_ptr + (offset_am * c_stride_0)[:, None] + offset_bn[None, :]
    tl.store(offsets_c, accumulator, mask=(offset_am[:,None] < M) & (offset_bn[None, :] < N))




def matmul(A: torch.Tensor, B: torch.Tensor):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    M, K = A.shape
    K, N = B.shape
    assert A.shape[1] == B.shape[0]
    output = torch.empty([M, N], device='cuda', dtype=torch.float16)

    A_stride_0 = A.stride(0)
    B_stride_0 = B.stride(0)
    C_stride_0 = output.stride(0)

    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](A, B, output,
                        M, N, K,
                        A_stride_0, B_stride_0, C_stride_0,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    return output

M = 64
N = 130
K = 68

torch.manual_seed(42)
A = torch.randn([M, K], device='cuda', dtype=torch.float16)
B = torch.randn([K, N], device='cuda', dtype=torch.float16)
ref = torch.matmul(A, B)

triton_output = matmul(A, B)
print(triton_output)
print(ref)
if torch.allclose(triton_output, ref, atol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")





