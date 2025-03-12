import torch
import triton
import triton.language as tl



@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  x_stride_0,
                  y_stride_0,
                  z_stride_0,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_a = a_ptr + pid_m * BLOCK_SIZE_M * x_stride_0 + tl.arange(0, BLOCK_SIZE_M)[:, None] * x_stride_0
    offsets_b = b_ptr + pid_n * BLOCK_SIZE_N  + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K , BLOCK_SIZE_K)):
        offsets_ak = offsets_a + (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :]
        offsets_bk = offsets_b + (k * BLOCK_SIZE_K * y_stride_0 + tl.arange(0, BLOCK_SIZE_K) * y_stride_0)[:, None]
        ak = tl.load(offsets_ak, mask = tl.arange(0, BLOCK_SIZE_K)[None, :] < K - k*BLOCK_SIZE_K, other=0.0)
        bk = tl.load(offsets_bk, mask = tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k*BLOCK_SIZE_K, other=0.0)
        c += tl.dot(ak, bk)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    offsets_c = c_ptr + (pid_m * BLOCK_SIZE_M * z_stride_0 + tl.arange(0, BLOCK_SIZE_M) * z_stride_0)[:, None] + \
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
    tl.store(offsets_c, c, mask=c_mask)




def matmul(A, B):
    rows_x, cols_x = A.shape
    rows_y, cols_y = B.shape
    assert cols_x == rows_y

    C = torch.empty((rows_x, cols_y), dtype=torch.float16, device='cuda')

    x_stride_0 = A.stride(0)
    y_stride_0 = B.stride(0)
    z_stride_0 = C.stride(0)

    M = rows_x
    N = cols_y
    K = cols_x

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = lambda meta:(triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](A, B, C, M, N, K,
                        x_stride_0,
                        y_stride_0,
                        z_stride_0,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K)
    return C



torch.manual_seed(42)
a = torch.randn([512, 512], dtype=torch.float16, device='cuda')
b = torch.randn([512, 512], dtype=torch.float16, device='cuda')
ref = torch.matmul(a, b)
triton_output = matmul(a, b)
print(triton_output)
print(ref)
if torch.allclose(triton_output, ref, atol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


