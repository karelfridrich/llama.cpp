# Branch `vulkan-iq-int8`: int8 kernels for IQ2/IQ3 quants on Vulkan

On Vulkan, the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S quants had no int8 (integer dot product) kernels. Whenever
more than one token was computed at once, they fell back to a path that reads the activations as f32, so every extra
token cost almost as much as a whole token. That hurts most in MTP (speculative) decoding, where each step verifies a
small batch of draft tokens, and in prompt processing.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Changes

| Commit | What |
|---|---|
| int8 MMVQ for IQ2/IQ3 | Mat-vec kernels using int8 dot products for the five types, modeled on the existing IQ4_XS kernel. Signs are applied with a per-byte mask |
| int8 MMQ, row counts, Q2_K on RDNA2 | Int8 matmul (prompt processing) for IQ2_XXS/XS/S and IQ3_XXS; mat-vec workgroups take 2 rows for one token and 4 for batches, so each copy of the grid table covers more work; on RDNA2, Q2_K matmul uses the fp16 path because its int8 version is slower there |

## Results

Swift-1.5-Qwen3.8-27B IQ3_S on an RX 6800 (tokens/s):

| Batch | Before | After |
|---|---|---|
| 1 token (plain decode) | 31.6 | 32.8 |
| 3 tokens (MTP verify) | 62.5 | 86.3 |
| 4 tokens | 65.4 | 106.8 |
| 8 tokens | 116 | 148 |
| 512 tokens (prompt) | 339 | 377 |

End to end with MTP (2 draft tokens), code generation: 40.8 -> 53.3 t/s.

## Testing

`test-backend-ops` MUL_MAT and MUL_MAT_ID for all five types against the CPU backend. Each new kernel was also broken
on purpose (signs, parity bit, half-block scales, high index bits) to confirm the tests catch it.
