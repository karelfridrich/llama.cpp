#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

// one warp per output, for long windows such as the MoE amax row max
static __global__ void pool1d_nchw_warp_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx  = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res = op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f;
    for (int i = b + lane; i < e; i += WARP_SIZE) {
        const float cur = __ldg(i_ptr + i);
        res = op == GGML_OP_POOL_MAX ? max(res, cur) : res + cur;
    }

    if (op == GGML_OP_POOL_MAX) {
        res = warp_reduce_max(res);
    } else {
        res = warp_reduce_sum(res);
        res = e > b ? res / (e - b) : 0.0f;
    }

    if (lane == 0) {
        o_ptr[cur_ow] = res;
    }
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    if (kw >= WARP_SIZE) {
        const int outputs_per_block = CUDA_POOL1D_BLOCK_SIZE / WARP_SIZE;
        const int num_blocks = (parallel_elements + outputs_per_block - 1) / outputs_per_block;
        pool1d_nchw_warp_kernel<<<num_blocks, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
        return;
    }
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

struct pool1d_amax_params {
    float s0, b0, s1, b1, cmin, cmax, s2, b2;
};

// one warp per row, same float ops in the same order as the unfused graph
static __global__ void pool1d_amax_fused_kernel(
        const float * src, float * dst, const int ncols, const int nrows, const pool1d_amax_params p) {
    const int row  = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= nrows) {
        return;
    }

    const float * x = src + (int64_t) row * ncols;
    float amax = -FLT_MAX;
    for (int i = lane; i < ncols; i += WARP_SIZE) {
        amax = max(amax, fabsf(__ldg(x + i)));
    }
    amax = warp_reduce_max(amax);

    if (lane == 0) {
        float e = p.s0 * amax + p.b0;
        e = logf(e);
        e = p.s1 * e + p.b1;
        e = ceilf(e);
        e = fminf(fmaxf(e, p.cmin), p.cmax);
        e = p.s2 * e + p.b2;
        dst[row] = expf(e);
    }
}

bool ggml_cuda_should_fuse_pool1d_amax(ggml_tensor * const * nodes) {
    const ggml_tensor * abs  = nodes[0];
    const ggml_tensor * pool = nodes[1];
    const ggml_tensor * src  = abs->src[0];

    if (ggml_get_unary_op(abs) != GGML_UNARY_OP_ABS || ggml_get_unary_op(nodes[5]) != GGML_UNARY_OP_CEIL ||
        ggml_get_unary_op(nodes[8]) != GGML_UNARY_OP_EXP) {
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        if (nodes[i]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src)) {
        return false;
    }

    // a single window covering the whole row
    const int32_t * opts = (const int32_t *) pool->op_params;
    return opts[0] == GGML_OP_POOL_MAX && opts[1] == src->ne[0] && opts[2] == src->ne[0] && opts[3] == 0 &&
           pool->ne[0] == 1;
}

void ggml_cuda_op_pool1d_amax_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    const ggml_tensor * src = nodes[0]->src[0];
    ggml_tensor * dst = nodes[8];

    pool1d_amax_params p;
    memcpy(&p.s0,   (const float *) nodes[2]->op_params + 0, sizeof(float));
    memcpy(&p.b0,   (const float *) nodes[2]->op_params + 1, sizeof(float));
    memcpy(&p.s1,   (const float *) nodes[4]->op_params + 0, sizeof(float));
    memcpy(&p.b1,   (const float *) nodes[4]->op_params + 1, sizeof(float));
    memcpy(&p.cmin, (const float *) nodes[6]->op_params + 0, sizeof(float));
    memcpy(&p.cmax, (const float *) nodes[6]->op_params + 1, sizeof(float));
    memcpy(&p.s2,   (const float *) nodes[7]->op_params + 0, sizeof(float));
    memcpy(&p.b2,   (const float *) nodes[7]->op_params + 1, sizeof(float));

    const int ncols = src->ne[0];
    const int nrows = ggml_nrows(src);
    const int rows_per_block = CUDA_POOL1D_BLOCK_SIZE / WARP_SIZE;
    const int num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
    pool1d_amax_fused_kernel<<<num_blocks, CUDA_POOL1D_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data, ncols, nrows, p);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int parallel_elements = (int)(nr * OW);

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}
