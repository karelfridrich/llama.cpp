#include "mmid-tc.cuh"
#include "mmid.cuh"
#include "convert.cuh"

#include <mma.h>

// Each CUDA block computes 64 weight rows x up to 64 tokens of one expert, over K in steps of 64 values.
// A step dequantizes the 64x64 weight tile to FP16 in shared memory, gathers the 64 FP16 activation rows of
// the tile's tokens, and multiplies them with WMMA (16x16x16, FP32 accumulation). The raw data of the next step
// is loaded into registers while the tensor cores work on the current one.

static constexpr int MMID_TC_BM = 64, MMID_TC_BN = 64, MMID_TC_BK = 64, MMID_TC_PAD = 8, MMID_TC_NWARPS = 4;

// Weights loaders: raw data of 8 consecutive values of one row (load) and their dequantization (dequant).
// A group g (0..7) covers the values k0 + 8*g .. k0 + 8*g + 7, k0 a multiple of 64 in a block of 256 values.
template <ggml_type type> struct mmid_tc_q;

// 0x6400 | q is the half 1024 + q for q < 1024: two values per 32-bit word, from bytes 2p and 2p+1 of w
static __device__ __forceinline__ uint32_t mmid_tc_lanes(const uint32_t w, const int p) {
    return __byte_perm(w, 0, p ? 0x4342 : 0x4140);
}

template <> struct mmid_tc_q<GGML_TYPE_Q5_K> {
    struct raw { uint32_t dm, sc[3]; uint2 ql, qh; };

    static __device__ __forceinline__ raw load(const char * row, const int k0, const int g) {
        const block_q5_K * x = (const block_q5_K *) row + k0/QK_K;
        const int c = (k0%QK_K)/64, j = (8*g)%32;
        raw r;
        r.dm    = *(const uint32_t *) &x->dm;
        r.sc[0] = ((const uint32_t *) x->scales)[0];
        r.sc[1] = ((const uint32_t *) x->scales)[1];
        r.sc[2] = ((const uint32_t *) x->scales)[2];
        r.ql    = *(const uint2 *) (x->qs + 32*c + j);
        r.qh    = *(const uint2 *) (x->qh + j);
        return r;
    }

    static __device__ __forceinline__ void dequant(const raw & r, const int k0, const int g, half2 * v) {
        const int c = (k0%QK_K)/64, hi = 8*g >= 32, js = 2*c + hi;
        // scale bytes by shifts: indexing the register array would put it in local memory
        auto sb = [&](const int n) -> uint32_t {
            const uint32_t w = n < 4 ? r.sc[0] : (n < 8 ? r.sc[1] : r.sc[2]);
            return (w >> 8*(n%4)) & 0xFF;
        };
        uint32_t sc, m;
        if (js < 4) {
            sc = sb(js) & 63; m = sb(js + 4) & 63;
        } else {
            sc = (sb(js + 4) & 0xF) | ((sb(js - 4) >> 6) << 4);
            m  = (sb(js + 4) >>  4) | ((sb(js)     >> 6) << 4);
        }
        const half2 dm  = *(const half2 *) &r.dm;
        const half2 d   = __float2half2_rn(__low2float(dm)*sc);
        const half2 mn  = __float2half2_rn(-__high2float(dm)*m);
        const half2 off = __float2half2_rn(1024.0f);
        const uint32_t wl[2] = { r.ql.x, r.ql.y }, wh[2] = { r.qh.x, r.qh.y };
#pragma unroll
        for (int w = 0; w < 2; ++w) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint32_t q = ((mmid_tc_lanes(wl[w], p) >> 4*hi) & 0x000F000F)
                                 | (((mmid_tc_lanes(wh[w], p) >> js) & 0x00010001) << 4) | 0x64006400;
                v[2*w + p] = __hfma2(__hsub2(*(const half2 *) &q, off), d, mn);
            }
        }
    }
};

template <> struct mmid_tc_q<GGML_TYPE_Q6_K> {
    struct raw { uint32_t ql[2], qh[2]; uint32_t d_sc; }; // d in the low 16 bits, the int8 scale above

    // block_q6_K is 210 bytes, only 2-byte aligned: 16-bit loads
    static __device__ __forceinline__ uint32_t load_u32(const uint8_t * p) {
        return *(const uint16_t *) p | (uint32_t(*(const uint16_t *) (p + 2)) << 16);
    }

    static __device__ __forceinline__ raw load(const char * row, const int k0, const int g) {
        const block_q6_K * x = (const block_q6_K *) row + k0/QK_K;
        const int t = (k0%QK_K)/8 + g, ip = t/16, l0 = 8*(t%16), qr = l0/32, il = l0%32;
        const uint8_t * ql = x->ql + 64*ip + 32*(qr%2) + il;
        const uint8_t * qh = x->qh + 32*ip + il;
        raw r;
        r.ql[0] = load_u32(ql); r.ql[1] = load_u32(ql + 4);
        r.qh[0] = load_u32(qh); r.qh[1] = load_u32(qh + 4);
        r.d_sc  = *(const uint16_t *) &x->d | (uint32_t(uint8_t(x->scales[8*ip + il/16 + 2*qr])) << 16);
        return r;
    }

    static __device__ __forceinline__ void dequant(const raw & r, const int k0, const int g, half2 * v) {
        const int t = (k0%QK_K)/8 + g, qr = (8*(t%16))/32;
        const uint16_t dh = r.d_sc & 0xFFFF;
        const half2 d   = __float2half2_rn(__half2float(*(const half *) &dh)*int8_t(r.d_sc >> 16));
        const half2 off = __float2half2_rn(1024.0f + 32.0f);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint32_t q = ((mmid_tc_lanes(r.ql[w], p) >> 4*(qr/2)) & 0x000F000F)
                                 | (((mmid_tc_lanes(r.qh[w], p) >> 2*qr) & 0x00030003) << 4) | 0x64006400;
                v[2*w + p] = __hmul2(__hsub2(*(const half2 *) &q, off), d);
            }
        }
    }
};

template <> struct mmid_tc_q<GGML_TYPE_IQ4_XS> {
    struct raw { uint32_t d_sh; uint32_t sl; uint2 qs; }; // d and scales_h, the scales_l byte, 8 quant bytes

    static __device__ __forceinline__ raw load(const char * row, const int k0, const int g) {
        const block_iq4_xs * x = (const block_iq4_xs *) row + k0/QK_K;
        const int v0 = k0%QK_K + 8*g, ib = v0/32, jj = v0%32;
        raw r;
        r.d_sh = *(const uint32_t *) x; // d, scales_h
        r.sl   = x->scales_l[ib/2];
        r.qs   = *(const uint2 *) (x->qs + 16*ib + jj%16);
        return r;
    }

    // kvalues_iq4nl as 4 words of 4 int8 each
    static __device__ __forceinline__ uint32_t table(const int w) {
        return (uint32_t(uint8_t(kvalues_iq4nl[4*w + 0])) <<  0) | (uint32_t(uint8_t(kvalues_iq4nl[4*w + 1])) <<  8) |
               (uint32_t(uint8_t(kvalues_iq4nl[4*w + 2])) << 16) | (uint32_t(uint8_t(kvalues_iq4nl[4*w + 3])) << 24);
    }

    static __device__ __forceinline__ void dequant(const raw & r, const int k0, const int g, half2 * v) {
        const int v0 = k0%QK_K + 8*g, ib = v0/32, hi = v0%32 >= 16;
        const uint16_t dh = r.d_sh & 0xFFFF;
        const int ls = ((r.sl >> 4*(ib%2)) & 0xF) | (((r.d_sh >> (16 + 2*ib)) & 3) << 4);
        const half2 d   = __float2half2_rn(__half2float(*(const half *) &dh)*(ls - 32));
        const half2 off = __float2half2_rn(1024.0f + 128.0f);
        const uint32_t t0 = table(0), t1 = table(1), t2 = table(2), t3 = table(3);
        const uint32_t wq[2] = { r.qs.x, r.qs.y };
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            // the 4 nibbles of word w as 4 table indices, one per selector nibble
            uint32_t x = (wq[w] >> 4*hi) & 0x0F0F0F0F;
            x = (x | (x >> 4)) & 0x00FF00FF;
            x = (x | (x >> 8)) & 0x0000FFFF;
            const uint32_t lo = __byte_perm(t0, t1, x & 0x7777);              // entries 0..7
            const uint32_t up = __byte_perm(t2, t3, x & 0x7777);              // entries 8..15
            const uint32_t q8 = __byte_perm(lo, up, 0x3210 | ((x & 0x8888) >> 1)); // pick by bit 3
            const uint32_t u8 = q8 ^ 0x80808080;                              // int8 + 128
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint32_t q = mmid_tc_lanes(u8, p) | 0x64006400;
                v[2*w + p] = __hmul2(__hsub2(*(const half2 *) &q, off), d);
            }
        }
    }
};

template <ggml_type type>
static __global__ void __launch_bounds__(MMID_TC_NWARPS*WARP_SIZE) mul_mat_id_tc(
        const char * __restrict__ x, const int64_t nb01, const int64_t nb02,
        const half * __restrict__ y, const int64_t stride_y,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        float * __restrict__ dst, const int64_t stride_dst, const int K) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    using namespace nvcuda;
    constexpr int BM = MMID_TC_BM, BN = MMID_TC_BN, BK = MMID_TC_BK, PAD = MMID_TC_PAD, NT = MMID_TC_NWARPS*WARP_SIZE;
    constexpr int LDC = BM + 4;
    constexpr int SMEM_AB = (BM + BN)*(BK + PAD)*sizeof(half), SMEM_C = BN*LDC*sizeof(float);
    constexpr int AI = BM*8/NT, BI = BN*8/NT; // groups of 8 values per thread
    using q = mmid_tc_q<type>;

    __shared__ __align__(16) char smem[SMEM_AB > SMEM_C ? SMEM_AB : SMEM_C];
    __shared__ int rows_dst[BN];
    half (*As)[BK + PAD] = (half (*)[BK + PAD]) smem;
    half (*Bs)[BK + PAD] = (half (*)[BK + PAD]) (smem + BM*(BK + PAD)*sizeof(half));

    // the 64-row tiles of one token tile run next to each other, so its activation rows stay in L2
    const ggml_cuda_mm_id_tile_spec ts = tile_spec[blockIdx.y];
    if (ts.col_low == ts.col_high) {
        return;
    }
    const int ncols  = ts.col_high - ts.col_low;
    const int m0     = blockIdx.x*BM;
    const int tid    = threadIdx.x, warp = tid/WARP_SIZE;
    const int wm     = warp % 2, wn = warp / 2;
    const bool narrow = ncols <= BN/2; // each warp then takes 16 rows x 32 columns

    if (tid < BN) {
        rows_dst[tid] = tid < ncols ? ids_dst[ts.col_low + tid] : -1;
    }

    const char * x_e = x + ts.expert*nb02 + m0*nb01;
    typename q::raw ar[AI];
    uint4 br[BI];
    int64_t yrow[BI];
#pragma unroll
    for (int i = 0; i < BI; ++i) {
        const int n = (tid + i*NT)/8;
        yrow[i] = n < ncols ? (int64_t) ids_src1[ts.col_low + n]*stride_y : -1;
    }

    auto load = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int it = tid + i*NT;
            ar[i] = q::load(x_e + (it/8)*nb01, k0, it%8);
        }
#pragma unroll
        for (int i = 0; i < BI; ++i) {
            const int g = (tid + i*NT)%8;
            br[i] = yrow[i] >= 0 ? *(const uint4 *) (y + yrow[i] + k0 + 8*g) : make_uint4(0, 0, 0, 0);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    load(0);
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int it = tid + i*NT;
            half2 v[4];
            q::dequant(ar[i], k0, it%8, v);
            *(uint4 *) &As[it/8][8*(it%8)] = *(const uint4 *) v;
        }
#pragma unroll
        for (int i = 0; i < BI; ++i) {
            const int it = tid + i*NT;
            *(uint4 *) &Bs[it/8][8*(it%8)] = br[i];
        }
        __syncthreads();

        if (k0 + BK < K) {
            load(k0 + BK); // in flight while the tensor cores work on this step
        }

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
            if (narrow) {
                wmma::load_matrix_sync(a[0], &As[16*warp][kk], BK + PAD);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::load_matrix_sync(b[j], &Bs[16*j][kk], BK + PAD);
                    wmma::mma_sync(acc[0][j], a[0], b[j], acc[0][j]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(a[i], &As[32*wm + 16*i][kk], BK + PAD);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::load_matrix_sync(b[j], &Bs[32*wn + 16*j][kk], BK + PAD);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // stage the tile as [token][row] in shared memory, then write each token row contiguously
    float * Cs = (float *) smem;
    if (narrow) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(Cs + 16*j*LDC + 16*warp, acc[0][j], LDC, wmma::mem_col_major);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::store_matrix_sync(Cs + (32*wn + 16*j)*LDC + 32*wm + 16*i, acc[i][j], LDC, wmma::mem_col_major);
            }
        }
    }
    __syncthreads();
    for (int it = tid; it < BN*BM/4; it += NT) {
        const int n = it/(BM/4), m4 = it%(BM/4);
        if (rows_dst[n] >= 0) {
            *(float4 *) (dst + rows_dst[n]*stride_dst + m0 + 4*m4) = *(const float4 *) (Cs + n*LDC + 4*m4);
        }
    }
#else
    GGML_UNUSED_VARS(x, nb01, nb02, y, stride_y, ids_src1, ids_dst, tile_spec, dst, stride_dst, K);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
}


bool ggml_cuda_should_use_mmid_tc(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const int cc) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_MMID_TC") != nullptr;
    if (disabled || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_VOLTA || ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_IQ4_XS) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % MMID_TC_BM != 0 || !ggml_is_contiguous(src1) || src1->type != GGML_TYPE_F32) {
        return false;
    }
    // prompt batches only: at least 16 rows per expert on average, else MMQ wastes less on empty tile columns
    const int64_t n_rows = ids->ne[0]*ids->ne[1];
    return n_rows >= 16*src0->ne[2];
}

void ggml_cuda_mul_mat_id_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == ggml_element_size(ids));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && nb0 == sizeof(float) && nb2 == ne1*nb1);
    GGML_ASSERT(ne13 == 1 && ne3 == 1);

    cudaStream_t stream = ctx.stream();
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_rows        = ne12*n_expert_used;

    // FP16 activations, rows as in src1 (contiguous)
    ggml_cuda_pool_alloc<half> y_f16(ctx.pool(), ggml_nelements(src1));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, y_f16.get(), ggml_nelements(src1), stream);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    const int ntj = (n_rows + MMID_TC_BN - 1)/MMID_TC_BN + ne02;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), tile_spec.get(),
        ne02, ne12, n_expert_used, ne11, MMID_TC_BN, ids->nb[1]/ggml_element_size(ids), nb12/nb11, /*write_inverse =*/ false, stream);

    const dim3 grid(ne01/MMID_TC_BM, ntj);
    const dim3 block(MMID_TC_NWARPS*WARP_SIZE);
    const char * x = (const char *) src0->data;
    float * d = (float *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_Q5_K:
            mul_mat_id_tc<GGML_TYPE_Q5_K><<<grid, block, 0, stream>>>(x, nb01, nb02, y_f16.get(), ne10, ids_src1.get(), ids_dst.get(), tile_spec.get(), d, nb1/sizeof(float), ne00);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_id_tc<GGML_TYPE_Q6_K><<<grid, block, 0, stream>>>(x, nb01, nb02, y_f16.get(), ne10, ids_src1.get(), ids_dst.get(), tile_spec.get(), d, nb1/sizeof(float), ne00);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_id_tc<GGML_TYPE_IQ4_XS><<<grid, block, 0, stream>>>(x, nb01, nb02, y_f16.get(), ne10, ids_src1.get(), ids_dst.get(), tile_spec.get(), d, nb1/sizeof(float), ne00);
            break;
        default:
            GGML_ABORT("unsupported type for mmid_tc");
    }
    CUDA_CHECK(cudaGetLastError());

}
