# Branch `vulkan-rdna2`: Vulkan heuristics tuned for RDNA2 (RX 6800)

Upstream picks between kernel variants with per-vendor and per-architecture heuristics. On RDNA2 a few of those
choices are slower than the alternative; this branch collects the measured exceptions.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Changes

| Commit | What |
|---|---|
| fp mat-vec for one column of narrow Q4_0/Q4_K/Q5_K | Single-token decode uses the fp mat-vec instead of the int8 (MMVQ) one for these types when k <= 4096. Wider matrices, batches (MTP verification) and the IQ types keep int8 |

Also RDNA2-specific, but on branch [`vulkan-iq-int8`](https://github.com/karelfridrich/llama.cpp/tree/vulkan-iq-int8):
Q2_K matmul uses the fp path, since its int8 version is slower on RDNA2.

## Results

RX 6800, decode (tokens/s):

| Model | Before | After |
|---|---|---|
| Gemma 4 E4B QAT Q4_0 | 110.3 | 115.6 |
| Qwen3.5-4B UD-Q5_K_XL | 105.5 | 106.2 |
| Qwen3.8 27B, Swift-Bonsai | unchanged (all their matrices are wider) | |

Three-token batches (MTP verification) are unchanged. Per shape, the fp path wins on narrow matrices but loses on
wide ones (Gemma's 2560 x 10240: 463 -> 385 GB/s), hence the k <= 4096 limit.
