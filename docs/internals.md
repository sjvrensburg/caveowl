# caveowl internals

How the training rows are rendered and masked, which modules get LoRA, and where the memory goes. The README links here; nothing in this file is needed to run the pipeline.

## Template and loss masking

llama-server serves the template **embedded in the GGUF**, which is an Unsloth-fixed variant of
the HF repo's `chat_template.jinja` : it merges a developer message into
the system turn and renders scalar tool arguments with `|string` instead of `|tojson`. The
markup the masks depend on is identical in both, but training renders with the GGUF's copy
(`template/`), extracted byte-exact from `Ornith-1.5-35B-Q4_K_M.gguf`.

**Thinking off.** This template renders every assistant turn, history included, with a think
block; empty reasoning gives `<think>\n\n</think>\n\n`. With `enable_thinking=false` (what pi's
`--thinking off` sends), the generation prompt is `<|im_start|>assistant\n<think>\n\n</think>\n\n`,
byte-identical to that block. So at inference the empty think block is **prompt**, prefilled by
the server, and training treats it the same way:

- loss **off**: system prompt and tool declarations, user turns, tool results (in this template
  they are `<|im_start|>user\n<tool_response>...` turns), and each assistant turn's
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
- loss **on**: each assistant turn's content, its tool calls (`<tool_call>\n<function=...>`
  through `</tool_call>`) and the closing `<|im_end|>`; the newline after it is not trained.

Spans are prefix differences of the render (`owl_data.tokenize_chat`), which is exact here
because the template renders history position-independently. A row is refused rather than
guessed at when: a prefix does not line up, a call or result is missing from the render, a token
straddles a span edge (the server tokenises prompt and completion apart), or the row's own text
contains template markup (`<think>`, `<tool_call>`, `<|im_start|>`...). Scalar tool arguments are
pre-rendered JSON-style, because the GGUF template's `|string` would otherwise teach `True`/`None`
where the model (and llama-server's parser) expect `true`/`null` (23 values in the tool slice).

`--prepare-only` prints one masked text pair, one multi-turn tool row, one call-only row and the
shortest skill trajectory, with the loss-bearing runs in `⟦ ⟧`.

Token stats from `--prepare-only` (2026-10-12, Ornith tokenizer, GGUF template; no row was
refused for markup, prefix mismatch or a straddling token; `data/out/prepare_only.log`):

| Slice | Rows | Median | p90 | Max | Tokens | Loss-bearing |
|---|---:|---:|---:|---:|---:|---:|
| text | 1679 | 368 | 829 | 3007 | 749,440 | 466,884 (62%) |
| tool | 439 | 794 | 2040 | 3847 | 451,538 | 66,179 (15%) |
| anchor | 358 | 170 | 329 | 662 | 68,813 | 21,656 (31%) |
| skill (x2) | 268 | 2835 | 3903 | 4094 | 799,690 | 56,942 (7%) |
| **train** | **2744** | 415 | 2078 | 4094 | **2,069,481** | **611,661 (30%)** |
| eval text / tool | 187 / 42 | 349 / 923 | | | 85,054 / 50,035 | |

2744 rows at batch 1 x accumulation 16 = **172 optimiser steps** for the epoch; checkpoints every
25 steps.

## LoRA targets (verified)

Module names come from `Qwen3_5MoeForConditionalGeneration` built from the pinned `config.json`
on the meta device (transformers 5.17) and the checkpoint's `model.safetensors.index.json`:

| Target | Layers | Note |
|---|---:|---|
| `self_attn.{q,k,v,o}_proj` | 10 | full attention only every 4th layer (3, 7, ..., 39); `q_proj` is 2048 -> 8192 (query + output gate) |
| `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` | 30 | Gated DeltaNet. `in_proj_a`/`in_proj_b` (32-wide) left alone |
| `mlp.shared_expert.{gate,up,down}_proj` | 40 | |
| `mlp.experts.gate_up_proj`, `down_proj` | 40 | fused 3-D Parameters (256 experts), LoRA via Unsloth's MoE path (PEFT target_parameters) |
| not targeted | | `mlp.gate` (router), `mlp.shared_expert_gate`, embeddings, `lm_head`, the vision tower |

`checks/attach_tiny.py` (a 4-layer random model of the same class, run 2026-10-12) confirms
Unsloth's `FastModel.get_peft_model` with this list and `finetune_vision_layers=False`: it loads
the ConditionalGeneration class, prints "Enabling LoRA on MoE parameters:
['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']", wraps every listed leaf, puts **zero**
trainable parameters in the vision tower, and every `lora_B` (experts included) gets a gradient
on one backward. By arithmetic for the real model at r16: ~19M on attention, DeltaNet and the
shared expert, ~923M on the routed experts (per expert r x (in + out)), **~0.94B trainable**.
Not yet seen on the real model: the smoke run prints the actual count.

## Memory

| | GiB |
|---|---:|
| bf16 weights (35.1B params, vision tower included) | ~65 |
| LoRA ~0.94B params + grads + fp32 AdamW state | ~10-15 |
| activations (Unsloth checkpointing, 4096 tokens, batch 1) + logits (4096 x 248320) | ~5 |
| **total** | **~80-85** (the probe's estimate; nothing at 35B scale is measured yet) |

The resident servers hold ~46 GiB of GTT (2026-10-12: `:8092` E4B, `:8088` 35B, winnow, embedding),
so **they must be down**: `takedown_servers.sh`. QLoRA (`load_in_4bit = true`, ~35-40 GiB) would
fit beside them, at the accuracy cost Unsloth warns about for Qwen3.5. `causal_conv1d` has no
ROCm build, so DeltaNet's short convolution runs Unsloth's slow PyTorch fallback: expect the step
time to be worse than the probe's small-model numbers suggest.
