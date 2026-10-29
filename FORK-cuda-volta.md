# Branch `cuda-volta`: CUDA speedups for Volta (V100)

Upstream's CUDA backend is tuned for Turing and newer. On Volta (sm70) several fast paths are skipped because they
need Turing's MMA instructions, and some kernels fall back to slower generic code. This branch closes those gaps,
mostly for single-request decode of MoE models. Every change is either limited to Volta or measured not to slow
other cases.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Changes

| Commit | What | Effect on V100 |
|---|---|---|
| use MMQ for many-expert MoE | Prompt processing for MoE models with > 64 experts uses MMQ instead of cuBLAS, which ran each small expert separately | prefill up to ~2x |
| Volta q8_0 FA kernel | A flash-attention kernel for a q8_0 KV cache on Volta: K/V stay q8_0 in memory and are widened to f16 tile by tile for the tensor cores, with the keys split across many blocks. Meant for long-context MTP verification. Ported from jackjusko's Volta fork (see Credits) and extended to several sequences | faster decode with q8_0 KV at long context |
| sm70 FA config for 256x256 heads | Adds the Volta flash-attention config for 256-wide heads (Gemma, Qwen3.8), taken from upstream PR #27997 | |
| faster POOL_1D, fused MoE amax | One warp per output for long pooling windows, and the MoE down-projection rescaling chain fused into one kernel (same float ops, same results) | Flint tg 88.7 -> 107.5 t/s |
| keep gate/up inputs alive | The gate/up/GLU fusion now works without Turing MMA | Flint 106.9 -> 111.6, Gemma-4 26B 92.2 -> 94.4, Qwen3-30B 132.8 -> 135.3 |
| small-K MMVQ for IQ types | The small-K mat-vec path for IQ quants, only when most of a block would idle (MoE down projections with K = 512) | Flint 112.1 -> 113.5 |
| Q/K/V as one MMVQ | Separate Q/K/V weights stored back to back run as one mat-vec: one input quantization, one launch | Flint 114.1 -> 123.9, Qwen3-30B 135.4 -> 140.8 |
| quantize the MoE input once | Routed and shared experts share one input quantization even when they can't share a kernel | Flint 123.9 -> 125.1 |

Numbers are `llama-bench` tg128 (single-stream decode, tokens/s) on a V100 PCIe 32 GB. Flint is `flint-UD-Q4_K_XL`.

## Credits

- The q8_0 flash-attention kernel (`ggml/src/ggml-cuda/fattn-q8-volta.cuh`) is ported from
  [jackjusko/jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash), commit
  `4cb009882`, and extended to several sequences and FP32 partial results.
- The sm70 flash-attention config for 256-wide heads is from upstream
  [PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997).

## Testing

- `test-backend-ops` against the CPU backend, including new perf and whole-graph cases so the fused paths are exercised
- Perplexity: fused and unfused paths give bit-identical results for our fusions (float order unchanged)
