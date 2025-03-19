import torch
import triton
import triton.language as tl

@triton.jit
def grouped_gemm_kernel(tensor_a_ptrs, tensor_b_ptrs, tensor_c_ptrs,
                        tensor_shapes, tensor_strides, lengths,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
                        ):
    tile_index = tl.program_id(0)
    last_problem_end = 0
    for i in tl.range(lengths):

        M = tl.load(tensor_shapes + 3 * i)
        N = tl.load(tensor_shapes + 3 * i + 1)
        K = tl.load(tensor_shapes + 3 * i + 2)
        a_stride_0 = tl.load(tensor_strides + 3 * i)
        b_stride_0 = tl.load(tensor_strides + 3 * i + 1)
        c_stride_0 = tl.load(tensor_strides + 3 * i + 2)

        tile_m = tl.cdiv(M, BLOCK_SIZE_M)
        tile_n = tl.cdiv(N, BLOCK_SIZE_N)
        tile_size = tile_m * tile_n
        # tile_index = pid

        while (tile_index >= last_problem_end and tile_index < last_problem_end + tile_size):
            a_ptr = tl.load(tensor_a_ptrs + i).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(tensor_b_ptrs + i).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(tensor_c_ptrs + i).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_index - last_problem_end
            tile_index_m = tile_idx_in_gemm // tile_n
            tile_index_n = tile_idx_in_gemm % tile_n

            k = tl.cdiv(K, BLOCK_SIZE_K)
            offset_am = tile_index_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offset_bn = tile_index_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
            for kk in tl.range(k):
                offset_k = kk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                offsets_am = a_ptr + offset_am[:, None] * a_stride_0 + offset_k[None, :]
                offsets_bn = b_ptr + offset_k[:, None] * b_stride_0 + offset_bn[None, :]
                # tl.multiple_of(offsets_am, [16, 16])
                # tl.multiple_of(offsets_bn, [16, 16])
                a_tile = tl.load(offsets_am)
                b_tile = tl.load(offsets_bn)
                accumulator += tl.dot(a_tile, b_tile)
            accumulator = accumulator.to(tl.float16)
            offsets_c = c_ptr + offset_am[:, None] * c_stride_0 + offset_bn[None, :]
            tl.store(offsets_c, accumulator)
            tile_index += 46
        last_problem_end = last_problem_end + tile_size





def grouped_gemm(group_A, group_B):
    lengths = len(group_A)
    a_ptrs = []
    b_ptrs = []
    c_ptrs = []
    shapes = []
    strides = []
    output = []
    for i in range(lengths):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty([M, N], dtype=torch.float16, device='cuda')
        output.append(C)
        a_ptrs.append(A.data_ptr())
        b_ptrs.append(B.data_ptr())
        c_ptrs.append(C.data_ptr())
        shapes += [M, N, K]
        strides += [A.stride(0), B.stride(0), C.stride(0)]
    # print(f"shapes: {shapes}")
    # print(f"a_ptrs: {a_ptrs}")
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    tensor_a_ptrs = torch.tensor(a_ptrs, device='cuda')
    tensor_b_ptrs = torch.tensor(b_ptrs, device='cuda')
    tensor_c_ptrs = torch.tensor(c_ptrs, device='cuda')
    tensor_shapes = torch.tensor(shapes, dtype=torch.int32, device='cuda')
    tensor_strides = torch.tensor(strides, dtype=torch.int32, device='cuda')
    grid = (46,)
    grouped_gemm_kernel[grid](tensor_a_ptrs, tensor_b_ptrs, tensor_c_ptrs,
                              tensor_shapes, tensor_strides, lengths,
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    return output


group_m = [1024, 512, 256]
group_n = [1024, 512, 256]
group_k = [512, 512, 512]

# group_m = [1024]
# group_n = [1024]
# group_k = [512]

group_size = len(group_m)

group_A = []
group_B = []
# group_C = []
group_strides = []
torch.manual_seed(42)
for i in range(group_size):
    A = torch.randn([group_m[i], group_k[i]], dtype=torch.float16, device='cuda')
    B = torch.randn([group_k[i], group_n[i]], dtype=torch.float16, device='cuda')
    # C = torch.randn([group_m[i], group_n[i]], dtype=torch.float16, device='cuda')
    group_A.append(A)
    group_B.append(B)

torch_ref = []
for i in range(group_size):
    torch_ref.append(torch.matmul(group_A[i], group_B[i]))

print(torch_ref)
triton_output = grouped_gemm(group_A, group_B)
print(triton_output)

for i in range(group_size):
    assert torch.allclose(torch_ref[i], triton_output[i], atol=1e-1, rtol=1e-1)
