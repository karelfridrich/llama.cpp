#include "common.cuh"

#define CUDA_POOL1D_BLOCK_SIZE 256

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// MoE amax rescaling: abs, pool_1d max over the row, scale, log, scale, ceil, clamp, scale, exp
bool ggml_cuda_should_fuse_pool1d_amax(ggml_tensor * const * nodes);
void ggml_cuda_op_pool1d_amax_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes);
