# Branch `vulkan-gdn-state`: gated delta net state straight into the cache (Vulkan)

Models with gated delta net (linear attention) layers, such as Qwen3.5/3.6/3.8 and their tunes, keep a recurrent
state per layer: 3 MB per layer for Qwen3.8 27B, 48 such layers. Every token, the delta net op wrote its new state into
its own output, and a separate copy then moved it into the recurrent cache. With MTP (speculative decoding) that copy
moves one snapshot per verified token. It is pure memory traffic, and decode on these GPUs is limited by memory
bandwidth.

Back to the [fork overview](https://github.com/karelfridrich/llama.cpp#readme).

## Change

On Vulkan, `GATED_DELTA_NET` followed by the `CPY` that stores its state now runs as one dispatch: the shader writes
the state, or the MTP snapshots, directly into the cache rows and the copy is skipped. Upstream's CUDA backend already
does the same. The graph optimizer keeps the two together, since its reordering would otherwise split them.

## Results

RX 6800, tokens/s:

| | Before | After |
|---|---|---|
| Swift-1.5-Qwen3.8-27B IQ3_S, decode | 32.80 | 33.39 |
| Swift-Bonsai-2 PQ2_0, decode | 43.18 | 44.04 |
| Swift-1.5 + MTP (2 draft tokens), code | 54.2 | 55.3 |
| Swift-1.5 + MTP, planning prompts | 46.3 | 47.5 |

## Testing

- Bit-identical output with the fusion on and off: perplexity in decode and batch mode, and MTP greedy generations
- `test-backend-ops` `GATED_DELTA_NET_CACHE_FUSION`, upstream's cases plus new plain-decode ones; breaking the fused
  store on purpose makes them fail

Still open: the matching read side (the state is gathered from the cache into a temporary before the op), which needs
changes outside the Vulkan backend.
