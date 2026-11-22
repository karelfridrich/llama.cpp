#pragma once

#include "common.cuh"

// MUL_MAT_ID on Volta tensor cores for prompt batches: the expert weights are dequantized to FP16 in shared memory.
bool ggml_cuda_should_use_mmid_tc(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

void ggml_cuda_mul_mat_id_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst);
