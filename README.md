# llama.cpp: karelfridrich fork

A personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for two older GPUs in one
home server:

- **NVIDIA Tesla V100 32 GB** (Volta, sm70, CUDA): a card that upstream no longer optimizes for
- **AMD Radeon RX 6800 16 GB** (RDNA2, Vulkan)

Each change lives on its own topic branch, based on upstream so that GitHub shows exactly what it changes.
`main` merges them all and is what actually runs here.

## Branches

| Branch | What it does | Details |
|---|---|---|
| [`cuda-volta`](https://github.com/karelfridrich/llama.cpp/tree/cuda-volta) | CUDA decode speedups for Volta: fused MoE and Q/K/V mat-vecs, a q8_0 flash-attention kernel for MTP verification, small-K IQ mat-vecs. Flint (MoE) decode on V100: 88.7 -> 125.1 t/s | [FORK-cuda-volta.md](FORK-cuda-volta.md) |
| [`prism-pq2`](https://github.com/karelfridrich/llama.cpp/tree/prism-pq2) | `PQ2_0`, the 2-bit format of PrismML's Bonsai models, for CPU, CUDA and Vulkan, plus a fused sign + Hadamard rotation on Vulkan | [FORK-prism-pq2.md](FORK-prism-pq2.md) |
| [`vulkan-iq-int8`](https://github.com/karelfridrich/llama.cpp/tree/vulkan-iq-int8) | Int8 kernels for the IQ2/IQ3 quants on Vulkan. On RX 6800: prompt processing +11%, MTP verification batches up to +63% | [FORK-vulkan-iq-int8.md](FORK-vulkan-iq-int8.md) |
| [`vulkan-gdn-state`](https://github.com/karelfridrich/llama.cpp/tree/vulkan-gdn-state) | Vulkan writes the gated delta net state (Qwen3.5/3.6/3.8) straight into the recurrent cache instead of copying it there. RX 6800 decode and MTP +2% | [FORK-vulkan-gdn-state.md](FORK-vulkan-gdn-state.md) |
| [`vulkan-rdna2`](https://github.com/karelfridrich/llama.cpp/tree/vulkan-rdna2) | Vulkan heuristics tuned for RDNA2: single-token decode of narrow Q4_0/Q4_K/Q5_K matrices uses the fp mat-vec. Gemma 4 E4B decode +5% | [FORK-vulkan-rdna2.md](FORK-vulkan-rdna2.md) |
| `main` | All of the above, merged | this page |
| `master` | Plain upstream, untouched | |

The branches are independent of each other; any one of them can be built on its own.

## Building

Nothing fork-specific: build exactly like upstream ([build docs](docs/build.md)). What runs here:

```sh
# V100
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=ON
# RX 6800
cmake -B build-vulkan -DGGML_VULKAN=ON
```

All of upstream's documentation still applies: see the [upstream README](https://github.com/ggml-org/llama.cpp#readme)
and [docs/](docs/).

## Status

- Based on upstream `418f905` (October 2026); rebased now and then.
- Tested with `test-backend-ops` and perplexity/KL-divergence checks against the CPU backend, on the two GPUs above
  only. Other GPUs get the same code paths in principle, but some changes are deliberately limited to Volta or RDNA2.
- Not submitted upstream. Findings may be reported there as issues.

Some of the code comes from other forks, credited on each branch's page: the Volta q8_0 flash-attention kernel from
[jackjusko/jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash), and PQ2_0
support ported from [PrismML's fork](https://github.com/PrismML-Eng/llama.cpp).

Written with AI assistance (Claude); the commits carry `Assisted-by` lines. License: MIT, same as upstream
([LICENSE](LICENSE)).
