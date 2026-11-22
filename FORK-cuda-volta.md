# Branch `cuda-volta`: CUDA speedups for Volta (V100)

Upstream's CUDA backend is tuned for Turing and newer. On Volta (sm70) several fast paths are skipped because they
need Turing's MMA instructions, and some kernels fall back to slower generic code. This branch closes those gaps,
mostly for single-request decode of MoE models. Every change is either limited to Volta or measured not to slow
other cases.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Changes

| Commit | What | Effect on V100 |
|---|---|---|
| Volta q8_0 FA kernel | A flash-attention kernel for a q8_0 KV cache on Volta: K/V stay q8_0 in memory and are widened to f16 tile by tile for the tensor cores, with the keys split across many blocks. Meant for long-context MTP verification. Ported from jackjusko's Volta fork (see Credits) and extended to several sequences | faster decode with q8_0 KV at long context |
| faster POOL_1D, fused MoE amax | One warp per output for long strided pooling windows (whole-row windows use upstream's row kernel), and the MoE down-projection rescaling chain fused into one kernel (same float ops, same results) | Flint tg 88.7 -> 107.5 t/s |
| keep gate/up inputs alive | The gate/up/GLU fusion now works without Turing MMA | Flint 106.9 -> 111.6, Gemma-4 26B 92.2 -> 94.4, Qwen3-30B 132.8 -> 135.3 |
| small-K MMVQ for IQ types | The small-K mat-vec path for IQ quants, only when most of a block would idle (MoE down projections with K = 512) | Flint 112.1 -> 113.5 |
| Q/K/V as one MMVQ | Separate Q/K/V weights stored back to back run as one mat-vec: one input quantization, one launch | Flint 114.1 -> 123.9, Qwen3-30B 135.4 -> 140.8 |
| quantize the MoE input once | Routed and shared experts share one input quantization even when they can't share a kernel | Flint 123.9 -> 125.1 |
| plain K/V loads in Volta flash attention | Since upstream #29510 the K/V tile loader checks for `kv_rows` per element. `kv_rows` needs Turing MMA and never reaches Volta, so Volta builds only the plain path | Gemma 4 31B pp512 at 16k 441 -> 492, Qwen3.8 27B 687 -> 722 |
| MMVF for unquantized MoE experts | BF16/F16 expert weights at small batch had no kernel on NVIDIA without a matching MMF (BF16 before Ampere) and took the synchronizing fallback, which turns CUDA graphs off for the whole model. Use MMVF there, as AMD already does | Inkling 2 Mini UD-Q8_K_XL (BF16 experts in 4 layers) tg 26.8 -> 31.3 |
| vectorized dequantization to FP16 | Batched matmuls of dense models dequantize the weights to FP16 for cuBLAS once per ubatch. The Q4_0, Q4_K, Q5_K and Q6_K kernels wrote scattered 2-byte values from 32- or 64-thread blocks; now each thread writes 8 consecutive values with one 16-byte store, in 256-thread blocks. Not Volta-specific | pp2048 (ubatch 512): Gemma 4 31B Q4_0 689 -> 825, Qwen3.8 27B Q4_K/Q5_K 860 -> 1007; MoE models unchanged |
| MoE experts on the tensor cores | For prompt batches (at least 16 rows per expert), MUL_MAT_ID with Q5_K, Q6_K or IQ4_XS experts dequantizes each 64x64 weight tile to FP16 in shared memory and multiplies it with WMMA, the next tile's raw data loaded during the math. MMQ runs on dp4a on Volta. Off with `GGML_CUDA_DISABLE_MMID_TC` | pp2048 (ubatch 2048): Tiel Coder 35B-A3B 3637 -> 4362, Flint 4037 -> 4846, Gemma 4 26B-A4B Q6_K 3497 -> 3876 |

Now upstream, dropped in the November 2026 rebase: MMQ for MoE prefill (upstream uses MMQ for every MoE model on
NVIDIA) and the sm70 flash-attention config for 256-wide heads (PR #27997).

Numbers are `llama-bench` tg128 (single-stream decode, tokens/s) unless marked pp512 (prompt processing) on a V100 PCIe 32 GB. Flint is `flint-UD-Q4_K_XL`.

## Credits

- The q8_0 flash-attention kernel (`ggml/src/ggml-cuda/fattn-q8-volta.cuh`) is ported from
  [jackjusko/jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash), commit
  `4cb009882`, and extended to several sequences and FP32 partial results.

## Testing

- `test-backend-ops` against the CPU backend, including new perf and whole-graph cases so the fused paths are exercised
- Perplexity: fused and unfused paths give bit-identical results for our fusions (float order unchanged)
