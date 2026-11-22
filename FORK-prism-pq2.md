# Branch `prism-pq2`: PrismML PQ2_0 support

PrismML releases 2-bit models in a format called `PQ2_0` (for example
Swift-Bonsai-2, a 27B model in 7.2 GB). Upstream llama.cpp can't load them; they need PrismML's own fork. This
branch brings the format into this fork's normal build, so the same binaries run Bonsai models next to everything
else.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## What PQ2_0 is

- Blocks of 128 weights: one f16 scale plus 2-bit codes that map to {-1, 0, 1, 2}
- The weights are stored rotated by a Hadamard transform, so at run time the activations are rotated too.
  The model's metadata (`prism.hadamard.*`) says which tensors need it; llama.cpp already reads it

## Changes

| Commit | What |
|---|---|
| add PQ2_0 | The type (id 142) on CPU (reference), CUDA (dequant, mat-vec, matmul; a Volta-specific mat-vec config) and Vulkan (dequant, mat-vec, matmul, Hadamard rotation for widths up to 8192), with test cases |
| fuse the sign vector into the FWHT | On Vulkan the rotation's sign multiply and the fast Walsh-Hadamard transform run as one kernel (ported from PrismML), and the graph optimizer keeps them together |

## Credits

PQ2_0 and its Hadamard handling come from [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
which supports it on CPU, CUDA and Vulkan. This branch ports that support onto current upstream:

- taken from PrismML: the format and CPU reference code, the Vulkan dequant shader, the shared-memory FWHT for wide
  rotations, and the signed (fused) FWHT
- done here: the CUDA paths, built on upstream's Q2_0 kernels made generic over the block type; the Volta mat-vec
  config; and the graph-optimizer fix that keeps the fused rotation together (PrismML's own fork loses a third of
  the fusions to the reordering)

## Results

Swift-Bonsai-2 PQ2_0:

| | Prompt (pp512) | Decode (tg128) |
|---|---|---|
| RX 6800, this fork | 380 t/s | 43.3 t/s |
| RX 6800, PrismML fork | 227 t/s | 43.6 t/s |
| V100, this fork | 900-1100 t/s | 56.0 t/s |

Output matches the CPU backend: KL divergence 0.00013 (Vulkan vs CPU), perplexity 8.2357 (CUDA) vs 8.2331 (CPU).
