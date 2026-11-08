# server-spec-max-slots

`llama-server --spec-draft-max-slots N`: start new speculative drafts only on decode steps where at most N slots are
generating. With more, every generating slot decodes one plain token in the shared batch; a slot in the middle of
replaying a partly accepted draft still finishes it. The check runs every step, so a lane left alone gets its drafts
back on its next token.

Default 0: always draft, as upstream. Works for every draft type (MTP, DFlash, EAGLE3, draft models).

## Why

With several slots generating, the batch already keeps the GPU busy, and drafting adds verification work and host
work (draft bookkeeping, sampling every verified position) for each slot. Whether skipping it pays depends on what
limits the model:

| Model, V100, 3 slots generating, Slaver coding runs | Always draft | `--spec-draft-max-slots 1` |
|---|---|---|
| Tiel Coder 35B-A3B (MoE, MTP n-max 3), ~30k context | 120 t/s | 144 t/s (+20%) |
| Tiel Coder 35B-A3B, ~35k | 112 t/s | 142 t/s (+27%) |
| Tiel Coder 35B-A3B, ~40k | 109 t/s | 133 t/s (+22%) |
| Muse Glimmer 30B (dense, DFlash n-max 3), ~20k | 68 t/s | 62 t/s (-8%) |

- MoE models with few active parameters are limited by the host thread (llama-server's main thread at 100%, the GPU
  80-89% busy at 130-170 W): fewer verified positions per step is a clear win, and the slots run in lockstep instead of
  swinging with draft acceptance.
- Dense models are limited by reading the weights: a draft rides on the same read, so drafting stays worthwhile with
  several slots. Leave the flag off for them.
- A slot alone keeps its drafts: Tiel's MTP acceptance over a whole task was 67% with the flag, after long stretches of
  plain decoding among three slots.
