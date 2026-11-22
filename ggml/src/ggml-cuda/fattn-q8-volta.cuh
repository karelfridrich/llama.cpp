#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>

// sm_70 FlashAttention for MTP target verification on Qwen3.5/3.8 27B: 4 query tokens per sequence (3 drafts + 1),
// D=256, 24 Q heads over 4 KV heads, q8_0 K/V. The KV cache stays q8_0; each 16-key tile is widened to FP16 in
// shared memory and fed to Volta mma.sync.m8n8k4. The keys are split across many blocks, so the long-context
// cost spreads over the whole GPU, and a second kernel merges the partial softmax results.
// Ported from jackjusko/jusko-llama-volta-qwen3flash (4cb009882), extended to several sequences (ne[3]) and FP32
// partial results.
namespace ggml_q8v {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
// Fragment addressing of mma.m8n8k4 on Volta.

// Q / P: tile<32,4,half2,I_MAJOR>
__device__ __forceinline__ int qp_get_i() { return threadIdx.x & 31; }

// K: tile<8,4,half2,I_MAJOR_MIRRORED>
__device__ __forceinline__ int k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}

// QK^T accumulator: tile<32,8,float,I_MAJOR>
__device__ __forceinline__ int d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}

// V: tile<8,4,half2,J_MAJOR_MIRRORED>
__device__ __forceinline__ int v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l / 2) * 4) + (lane % 4);
}
__device__ __forceinline__ int v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 2) + (l % 2);
}

// Volta has no ldmatrix: plain loads, stride in half2.
__device__ __forceinline__ void load_qp(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
    const int row = qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[row * stride + l];
    }
}

__device__ __forceinline__ void load_k(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
    const int row = k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[row * stride + l];
    }
}

__device__ __forceinline__ void load_v(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[v_get_i(l) * stride + v_get_j(l)];
    }
}

// D[32x8 f32] += Q[32x8 f16] @ K[8x8 f16]^T
__device__ __forceinline__ void mma_qk(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
                 : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}

// PV[32x8 f16] += P[32x8 f16] @ V[8x8 f16]
__device__ __forceinline__ void mma_pv(half2 (&pv)[4], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * A = reinterpret_cast<const int *>(p);
    const int * B = reinterpret_cast<const int *>(v);
    int * D = reinterpret_cast<int *>(pv);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
                 : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}

// The QK^T accumulator straight into the PV A-operand layout: registers and one shuffle, no shared memory.
__device__ __forceinline__ void softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 4) {
        p[l0 / 2 + 0] = __floats2half2_rn(d[l0 + 0], d[l0 + 1]);
        p[l0 / 2 + 1] = __floats2half2_rn(d[l0 + 2], d[l0 + 3]);
        const int lane = threadIdx.x & 31;
        const int swap = l0 / 2 + (((lane % 4) / 2) ^ 1);
        p[swap] = __shfl_xor_sync(0xFFFFFFFFu, p[swap], 2, 32);
    }
}
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

constexpr int D           = 256;
constexpr int QH          = 24;
constexpr int KVH         = 4;
constexpr int G           = QH / KVH;
constexpr int T           = 4;
constexpr int BC          = 16;  // keys per tile
constexpr int BR          = 32;  // query rows per tile: T*G = 24 used
constexpr int THREADS     = 128;
constexpr int D_SLICE     = 64;  // output columns per warp
constexpr int SMEM_STRIDE = 264;
constexpr int MAX_SPLITS  = 160;
constexpr int MIN_KV      = 8192; // below this the generic kernels are as fast or faster
constexpr float LOG2E     = 1.4426950408889634074f;

// Takes the shape it was written for: 4 tokens per sequence, any number of sequences, plain masked attention, long KV.
static inline bool supports(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return Q->type == GGML_TYPE_F32 && Q->ne[0] == D && Q->ne[1] == T && Q->ne[2] == QH && Q->nb[0] == sizeof(float) &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
        K->ne[0] == D && V->ne[0] == D && K->ne[2] == KVH && V->ne[2] == KVH &&
        K->ne[1] >= MIN_KV && K->ne[1] == V->ne[1] && K->ne[3] == Q->ne[3] && V->ne[3] == Q->ne[3] &&
        K->nb[1] % sizeof(block_q8_0) == 0 && K->nb[2] % sizeof(block_q8_0) == 0 && K->nb[3] % sizeof(block_q8_0) == 0 &&
        V->nb[1] % sizeof(block_q8_0) == 0 && V->nb[2] % sizeof(block_q8_0) == 0 && V->nb[3] % sizeof(block_q8_0) == 0 &&
        mask != nullptr && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && (mask->ne[3] == 1 || mask->ne[3] == Q->ne[3]) &&
        dst->src[4] == nullptr && max_bias == 0.0f && logit_softcap == 0.0f;
}

static inline int split_count(int n_kv) {
    const int tiles           = (n_kv + BC - 1) / BC;
    const int requested       = std::min(MAX_SPLITS, tiles);
    const int tiles_per_split = (tiles + requested - 1) / requested;
    return (tiles + tiles_per_split - 1) / tiles_per_split; // only non-empty splits
}

// Per sequence and split: the unnormalized output and its running max and sum.
struct extra_data {
    uintptr_t pacc = 0, pm = 0, pl = 0, end = 0;
    int splits = 0;
};

static inline extra_data get_extra(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const size_t partials = (size_t) split_count((int) K->ne[1]) * dst->src[0]->ne[3] * T * QH;
    extra_data e;
    e.splits = split_count((int) K->ne[1]);
    e.end    = GGML_PAD((uintptr_t) dst->data + ggml_nbytes(dst), 128);
    e.pacc   = e.end;
    e.end    = GGML_PAD(e.end + partials * D * sizeof(float), 128);
    e.pm     = e.end;
    e.end    = GGML_PAD(e.end + partials * sizeof(float), 128);
    e.pl     = e.end;
    e.end   += partials * sizeof(float);
    return e;
}

static inline size_t get_alloc_size(const ggml_tensor * dst) {
    return get_extra(dst).end - (uintptr_t) dst->data;
}

__device__ __forceinline__ int row_qh(int row, int kvh) { return kvh * G + row % G; }
__device__ __forceinline__ int row_tok(int row) { return row / G; }
__device__ __forceinline__ int64_t pacc_idx(int qh, int d, int tok, int split, int seq, int splits) {
    return d + (int64_t) D * (qh + (int64_t) QH * (tok + (int64_t) T * (split + (int64_t) splits * seq)));
}
__device__ __forceinline__ int64_t pstat_idx(int qh, int tok, int split, int seq, int splits) {
    return qh + (int64_t) QH * (tok + (int64_t) T * (split + (int64_t) splits * seq));
}

__device__ __forceinline__ int4 deq8(const block_q8_0 & b, int off) {
    const float s = __half2float(b.d);
    half2 h[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        h[i] = __floats2half2_rn((float) b.qs[off + 2*i] * s, (float) b.qs[off + 2*i + 1] * s);
    }
    return *reinterpret_cast<int4 *>(h);
}

// Grid: KV head x split x sequence. Each block attends its 6 query heads x 4 tokens over one range of keys.
__launch_bounds__(THREADS, 2)
static __global__ void partial_kernel(
        const float * q, const block_q8_0 * K, const block_q8_0 * V, const half * mask,
        int n_kv, int splits, float scale,
        int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s1, int64_t k_s2, int64_t k_s3,
        int64_t v_s1, int64_t v_s2, int64_t v_s3, int64_t mask_s1, int64_t mask_s3,
        float * pacc, float * pm, float * pl) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    __shared__ __align__(16) half q_s[BR*SMEM_STRIDE];
    __shared__ __align__(16) half k_s[BC*SMEM_STRIDE];
    __shared__ __align__(16) half v_s[BC*SMEM_STRIDE];
    __shared__ half m_s[T*BC];

    const int kvh = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int row_count = T*G;

    q    += seq*q_s3;
    K    += seq*k_s3;
    V    += seq*v_s3;
    mask += seq*mask_s3;

    const int tiles           = (n_kv + BC - 1) / BC;
    const int tiles_per_split = (tiles + splits - 1) / splits;
    const int first_key       = split*tiles_per_split*BC;
    const int split_end       = min(n_kv, first_key + tiles_per_split*BC);
    if (first_key >= split_end) {
        return;
    }
    const int key_blocks = (split_end - first_key + BC - 1) / BC;

    for (int row = warp; row < BR; row += 4) {
        float vals[8] = {};
        if (row < row_count) {
            const int qh = row_qh(row, kvh), tok = row_tok(row);
#pragma unroll
            for (int p = 0; p < 8; p++) {
                vals[p] = q[lane + 32*p + q_s1*tok + q_s2*qh];
            }
        }
#pragma unroll
        for (int p = 0; p < 8; p++) {
            q_s[row*SMEM_STRIDE + lane + 32*p] = __float2half(vals[p]);
        }
    }
    __syncthreads();

    float acc[D_SLICE/8][8];
#pragma unroll
    for (int c = 0; c < D_SLICE/8; c++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[c][i] = 0.0f;
        }
    }
    float mlo = -CUDART_INF_F, mhi = -CUDART_INF_F, llo = 0.0f, lhi = 0.0f;

    for (int kb = 0; kb < key_blocks; kb++) {
        const int k0 = first_key + kb*BC;
        for (int chunk = tid; chunk < BC*(D/8); chunk += THREADS) {
            const int key_l = chunk / (D/8), d = (chunk - key_l*(D/8))*8, key = k0 + key_l;
            half * kd = &k_s[key_l*SMEM_STRIDE + d];
            half * vd = &v_s[key_l*SMEM_STRIDE + d];
            if (key < split_end) {
                const int bo = d/32, off = d & 31;
                *reinterpret_cast<int4 *>(kd) = deq8(K[key*k_s1 + kvh*k_s2 + bo], off);
                *reinterpret_cast<int4 *>(vd) = deq8(V[key*v_s1 + kvh*v_s2 + bo], off);
            } else {
                const int4 z = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(kd) = z;
                *reinterpret_cast<int4 *>(vd) = z;
            }
        }
        for (int i = tid; i < T*BC; i += THREADS) {
            const int tok = i / BC, col = i % BC, key = k0 + col;
            m_s[i] = key < n_kv ? mask[key + mask_s1*tok] : __float2half(-CUDART_INF_F);
        }
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int sub_k0 = k0 + sub*8;
            float ds[8] = {};
#pragma unroll
            for (int c = 0; c < D/8; c++) {
                half2 qf[4], kf[4];
                load_qp(qf, reinterpret_cast<const half2 *>(&q_s[c*8]), SMEM_STRIDE/2);
                load_k(kf, reinterpret_cast<const half2 *>(&k_s[sub*8*SMEM_STRIDE + c*8]), SMEM_STRIDE/2);
                mma_qk(ds, qf, kf);
            }
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const int row = d_get_i(l), col = d_get_j(l), key = sub_k0 + col;
                if (row < row_count && key < split_end) {
                    ds[l] = ds[l]*scale + __half2float(m_s[row_tok(row)*BC + sub*8 + col]);
                } else {
                    ds[l] = -CUDART_INF_F;
                }
            }
            float bmlo = fmaxf(fmaxf(ds[0], ds[1]), fmaxf(ds[4], ds[5]));
            float bmhi = fmaxf(fmaxf(ds[2], ds[3]), fmaxf(ds[6], ds[7]));
            bmlo = fmaxf(bmlo, __shfl_xor_sync(0xFFFFFFFFu, bmlo, 2));
            bmhi = fmaxf(bmhi, __shfl_xor_sync(0xFFFFFFFFu, bmhi, 2));
            const float nmlo = fmaxf(mlo, bmlo), nmhi = fmaxf(mhi, bmhi);
            const float alo = mlo == -CUDART_INF_F ? 0.0f : exp2f((mlo - nmlo)*LOG2E);
            const float ahi = mhi == -CUDART_INF_F ? 0.0f : exp2f((mhi - nmhi)*LOG2E);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const float nm = (l & 2) ? nmhi : nmlo;
                ds[l] = nm > -CUDART_INF_F && ds[l] > -CUDART_INF_F ? exp2f((ds[l] - nm)*LOG2E) : 0.0f;
            }
            float bllo = ds[0] + ds[1] + ds[4] + ds[5], blhi = ds[2] + ds[3] + ds[6] + ds[7];
            bllo += __shfl_xor_sync(0xFFFFFFFFu, bllo, 2);
            blhi += __shfl_xor_sync(0xFFFFFFFFu, blhi, 2);
            llo = llo*alo + bllo;
            lhi = lhi*ahi + blhi;
            mlo = nmlo;
            mhi = nmhi;

            const float alpha = (lane & 2) ? ahi : alo;
            half2 pp[4];
            softmax_to_half2(pp, ds);
#pragma unroll
            for (int c = 0; c < D_SLICE/8; c++) {
                half2 vf[4];
                load_v(vf, reinterpret_cast<const half2 *>(&v_s[sub*8*SMEM_STRIDE + warp*D_SLICE + c*8]), SMEM_STRIDE/2);
                half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                mma_pv(pv, pp, vf);
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    const float2 f = __half22float2(pv[n]);
                    acc[c][2*n]     = acc[c][2*n]*alpha     + f.x;
                    acc[c][2*n + 1] = acc[c][2*n + 1]*alpha + f.y;
                }
            }
        }
        __syncthreads();
    }

    const int row = lane;
    if (row >= row_count) {
        return;
    }
    const int qh = row_qh(row, kvh), tok = row_tok(row);
    if (warp == 0) {
        pm[pstat_idx(qh, tok, split, seq, splits)] = (lane & 2) ? mhi : mlo;
        pl[pstat_idx(qh, tok, split, seq, splits)] = (lane & 2) ? lhi : llo;
    }
#pragma unroll
    for (int c = 0; c < D_SLICE/8; c++) {
        float4 * out = reinterpret_cast<float4 *>(&pacc[pacc_idx(qh, warp*D_SLICE + c*8, tok, split, seq, splits)]);
        out[0] = make_float4(acc[c][0], acc[c][1], acc[c][2], acc[c][3]);
        out[1] = make_float4(acc[c][4], acc[c][5], acc[c][6], acc[c][7]);
    }
#else
    GGML_UNUSED_VARS(q, K, V, mask, n_kv, splits, scale, q_s1, q_s2, q_s3, k_s1, k_s2, k_s3,
        v_s1, v_s2, v_s3, mask_s1, mask_s3, pacc, pm, pl);
    NO_DEVICE_CODE;
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

// Grid: Q head x token x sequence, one thread per output column: merges the splits' partial softmax.
static __global__ void reduce_kernel(const float * pacc, const float * pm, const float * pl, int splits,
        float * out, int64_t o_s1, int64_t o_s2, int64_t o_s3) {
    const int qh = blockIdx.x, tok = blockIdx.y, seq = blockIdx.z, d = threadIdx.x;
    __shared__ float red[D];

    float m = -CUDART_INF_F;
    for (int s = d; s < splits; s += D) {
        m = fmaxf(m, pm[pstat_idx(qh, tok, s, seq, splits)]);
    }
    red[d] = m;
    __syncthreads();
    for (int st = D/2; st; st >>= 1) {
        if (d < st) {
            red[d] = fmaxf(red[d], red[d + st]);
        }
        __syncthreads();
    }
    const float max = red[0];
    __syncthreads();

    float * o = out + d + o_s1*qh + o_s2*tok + o_s3*seq;
    if (max == -CUDART_INF_F) {
        *o = 0.0f;
        return;
    }

    float l = 0.0f;
    for (int s = d; s < splits; s += D) {
        const float ls = pl[pstat_idx(qh, tok, s, seq, splits)];
        if (ls > 0.0f) {
            l += ls*expf(pm[pstat_idx(qh, tok, s, seq, splits)] - max);
        }
    }
    red[d] = l;
    __syncthreads();
    for (int st = D/2; st; st >>= 1) {
        if (d < st) {
            red[d] += red[d + st];
        }
        __syncthreads();
    }
    const float sum = red[0];

    float num = 0.0f;
    for (int s = 0; s < splits; s++) {
        if (pl[pstat_idx(qh, tok, s, seq, splits)] > 0.0f) {
            num += pacc[pacc_idx(qh, d, tok, s, seq, splits)]*expf(pm[pstat_idx(qh, tok, s, seq, splits)] - max);
        }
    }
    *o = sum > 0.0f ? num/sum : 0.0f;
}

static inline void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    const extra_data e = get_extra(dst);
    const int n_seq = (int) Q->ne[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();
    partial_kernel<<<dim3(KVH, e.splits, n_seq), THREADS, 0, stream>>>(
        (const float *) Q->data, (const block_q8_0 *) K->data, (const block_q8_0 *) V->data, (const half *) M->data,
        (int) K->ne[1], e.splits, scale,
        Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float), Q->nb[3]/sizeof(float),
        K->nb[1]/sizeof(block_q8_0), K->nb[2]/sizeof(block_q8_0), K->nb[3]/sizeof(block_q8_0),
        V->nb[1]/sizeof(block_q8_0), V->nb[2]/sizeof(block_q8_0), V->nb[3]/sizeof(block_q8_0),
        M->nb[1]/sizeof(half), M->ne[3] == 1 ? 0 : M->nb[3]/sizeof(half),
        (float *) e.pacc, (float *) e.pm, (float *) e.pl);
    CUDA_CHECK(cudaGetLastError());
    reduce_kernel<<<dim3(QH, T, n_seq), D, 0, stream>>>(
        (const float *) e.pacc, (const float *) e.pm, (const float *) e.pl, e.splits,
        (float *) dst->data, dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), dst->nb[3]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ggml_q8v
