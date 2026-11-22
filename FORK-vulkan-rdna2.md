# Branch `vulkan-rdna2`: Vulkan heuristics tuned for RDNA2 (RX 6800)

Upstream picks between kernel variants with per-vendor and per-architecture heuristics. On RDNA2 a few of those
choices are slower than the alternative; this branch collects the measured exceptions.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Changes

| Commit | What |
|---|---|
| fp mat-vec for one column of narrow Q4_0/Q4_K/Q5_K | Single-token decode uses the fp mat-vec instead of the int8 (MMVQ) one for these types when k <= 4096. Wider matrices, batches (MTP verification) and the IQ types keep int8 |
| int8 matmul tiles of 256 invocations | Without coopmat, AMD never enabled the large MMQ tile, so RDNA2 ran every prompt matmul on the medium tile with 4 subgroups. Now the medium tile has 8 subgroups for Q8_0, MXFP4, Q5_K, Q6_K and the IQ types, and Q5_0/Q5_1/Q4_K use a large tile with 8 subgroups when there are at least two large tiles per CU. Q4_0, Q4_1 and Q3_K keep the default tiles |

Also RDNA2-specific, but on branch [`vulkan-iq-int8`](https://github.com/karelfridrich/llama.cpp/tree/vulkan-iq-int8):
Q2_K matmul uses the fp path, since its int8 version is slower on RDNA2.

## Results

RX 6800, decode (tokens/s):

| Model | Before | After |
|---|---|---|
| Gemma 4 E4B QAT Q4_0 | 110.3 | 115.6 |
| Qwen3.5-4B UD-Q5_K_XL | 105.5 | 106.2 |
| Qwen3.8 27B, Swift-Bonsai | unchanged (all their matrices are wider) | |

RX 6800, prompt processing at 512 tokens (tokens/s), int8 matmul tiles:

| Model | Before | After |
|---|---|---|
| Qwen3.8 27B, Swift-1.5 IQ3_S | 396 | 411 |
| Qwen3.8 27B, Swift-Bonsai PQ2_0 | 386 | 429 |
| Gemma 4 12B UD-Q4_K_XL | 851 | 976 |
| Gemma 4 E4B/E2B QAT (Q4_0) | unchanged (Q4_0 keeps the default tiles) | |

Per type on Qwen3.8 27B FFN shapes (17408 x 5120 and 5120 x 17408, 512 tokens): Q8_0 +14%, PQ2_0 +20%, Q5_K +23%, Q6_K +12%, IQ2 +13-16%, IQ3_XXS +10%, IQ4_XS +6%, Q5_0 +12%, Q4_K +9%, IQ3_S +1%.

Three-token batches (MTP verification) are unchanged. Per shape, the fp path wins on narrow matrices but loses on
wide ones (Gemma's 2560 x 10240: 463 -> 385 GB/s), hence the k <= 4096 limit.
