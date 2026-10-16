# caveowl

A caveman-style LoRA of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
trained in bf16 with Unsloth on the EVO-X3 (Strix Halo gfx1151, 128 GB unified, GTT limit
112 GiB). It is the 35B port of [../cavegemma-e4b](../cavegemma-e4b), whose v3 (Gemma-4-E4B) was
live on `:8092` until caveowl replaced it. Built 2026-10-12. Run 5 is **live on `:8092`** (pi's default) since 2026-10-16, at temp 0.3: see
"Deployed" below.

## Why

The terse "caveman" style belongs in the weights, not in a brevity line in pi-signal's AGENTS.md:
fewer tokens per turn and the same facts. cavegemma showed what breaks when you try, and in what
order:

| E4B version | Data | Outcome |
|---|---|---|
| v1 | caveman text pairs + tool slice, 3 epochs, lr 2e-4 | tools 9/9, but code-eval 10/12 -> 2/12: the style leaked into code |
| v2 | + 353 self-distilled MBPP code anchors, 1 epoch, lr 1e-4 | passed every harness; rolled back the same day: on Signal it saw the weather skill and never read it (0/3 vs stock 2/3) |
| v3 | + 105 self-distilled pi skill trajectories (x2) | passed every gate incl. the skill gate; **live** |

caveowl starts at v3's recipe. Ornith-1.5-35B-A3B is the model on `:8088` (Open Notebook's
worker, pi's `local-ornith35`); whether a caveowl ever replaces anything is a decision for after
the gate, not part of this repo.

## What is different from cavegemma

| | cavegemma v3 (E4B) | caveowl (35B-A3B) |
|---|---|---|
| Trainer | transformers + peft + TRL | **Unsloth** `FastModel` + transformers `Trainer`. ../unsloth-probe (2026-10-12) showed Unsloth works on gfx1151 (~2.9x peft on Qwen3-0.6B, same loss curve; on a tiny qwen35moe the loss matches transformers and CPU fp32 and expert LoRA trains) |
| Precision | bf16 LoRA (~16 GB model) | bf16 LoRA, **~80-85 GiB**: every resident server must be down. Unsloth advises against 4-bit QLoRA on Qwen3.5 |
| LoRA targets | q/k/v/o + gate/up/down | attention q/k/v/o (10 full-attention layers), Gated DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj` (30 layers), shared expert gate/up/down (40), **routed experts** `gate_up_proj`/`down_proj` via Unsloth's MoE path |
| Template | Gemma 4 (`<\|turn>model`), tool results inside the model turn | Qwen-style ChatML from the **stock GGUF**, think block, `<tool_call>` XML, tool results as user turns (below) |
| Anchors | stock E4B's own answers | regenerated from **stock Ornith** (self-distillation is per model) |
| Quant | Q8_0 | Q4_K_M with an importance matrix, as the stock GGUF |

## Files

| File | What |
|---|---|
| `train_unsloth.py` | data build + masking + Unsloth bf16 LoRA. `--prepare-only` renders everything with the tokenizer alone and prints masked samples and token stats |
| `owl_data.py` | rendering and loss masks (thinking-off form), shared with the skill-anchor converter |
| `config_owl.toml` / `config_owl_smoke.toml` | full run / 6-step smoke (no save, no eval) |
| `template/ornith-gguf-chat-template.jinja` | the chat template embedded in the stock GGUF, byte-exact |
| `checks/attach_tiny.py` | which modules get LoRA, on a tiny random qwen35moe VL model |
| `merge_and_gguf.sh` + `merge_owl.py` | merge (with tensor-level checks) -> BF16 GGUF -> imatrix -> Q4_K_M. Not run |
| `takedown_servers.sh` / `restore_servers.sh` | stop / bring back exactly the resident model servers (operator runs these) |
| `scripts/serve_bench.sh` | any GGUF on the bench port `:8110` with `:8088`'s flags |
| `data/build_anchor.py`, `data/build_skill_anchor.py`, `data/pi_traj.py`, `data/pi_bench_guard.ts` | cavegemma's anchor builders, pointed at Ornith |
| `eval/skill_gate.py`, `eval/skill_gate_wide.py` | cavegemma's skill gates, pointed at Ornith |

Venv: `~/venvs/unsloth-probe` (recipe: `../unsloth-probe/RECIPE.sh`; unsloth 2026.10.3,
transformers 5.17.0, torch 2.11.0+rocm7.13.0 via a `.pth` onto `~/venvs/torch-rocm`). Run with
`HF_HUB_DISABLE_XET=1` (Xet stalls here).

## Data

Mix as cavegemma v3: text pairs + tool slice (target 25%) + code anchors (target 20%) + skill
trajectories (x2). Shares are of the text count; a slice with fewer rows than its target goes in
whole.

| Slice | Source | Rows (train) |
|---|---|---:|
| Caveman text pairs | `../cavegemma-e4b/data/out/train.jsonl` (sha256 `11ac05e3...`), eval `eval.jsonl` (`43340bfe...`) | 1679 (+187 eval) |
| Tool slice | `../cavegemma-e4b/data/out/tool_clean.jsonl` (`a5c7c80c...`): Nemotron multi-turn + xLAM call-only | 439 (+42 eval), 21% |
| Code anchors | `data/out/anchor_all.jsonl` (`data/merge_anchors.py`): equal counts of MBPP Python (`anchor.jsonl`, 358 of 427 kept) and UltraData-Code L3 sh/R/Rust (`anchor_ultra.jsonl`). Both stock Ornith, temperature 0, thinking off, kept only if the task's tests pass; L3 tasks are first kept only if the dataset's OWN reference solution passes its tests (24-41% did) | 592 offered (296 py, 89 sh, 82 R, 125 Rust); the 20% cap takes ~420, ~half each |
| Skill trajectories | `data/out/skill_traj.jsonl`, real `pi -p` runs of stock Ornith, kept only if `pi_traj.judge_session` passes | 134 from 188 prompts / 308 draws, x2 = 268 |

The text pairs and the tool slice are model-independent (Claude-written caveman targets; tool
calls and results byte-exact from the sources), so they are reused as they are. Anchors are not:
an anchor works because it is the model's own behaviour ("stay as you were"), so both anchor sets
were regenerated from stock Ornith, served as the bench unit `cand-owl` on `:8110` (the same GGUF
and binary as `:8088`, `--parallel 2 --ctx-size 262144`; `scripts/serve_bench.sh`). The pi agent
for the skill runs is a scratch `PI_CODING_AGENT_DIR` whose one provider copies production's
`local-ornith35` entry (`ornith-1.5-35b-a3b`, thinkingLevelMap off/medium) with `baseUrl` at the
logging proxy in front of `:8110`; thinking off, as pi-job and pi-signal run Ornith (the proxy log
shows pi sending `chat_template_kwargs: {"enable_thinking": false}`). Scratch cwds live under
`/tmp/cavegemma-skill-bench/owl`, never inside a project (cavegemma's uv-resync lesson), and the
bench guard is the only extension loaded.

Generated 2026-10-12 (`data/out/build_anchor.log`, `build_skill_anchor.log`; sha256
`anchor.jsonl` `d7c3db3d...`, `skill_traj.jsonl` `a309bea9...`; `data/out` is not committed):

| Skill family | Kept prompts | Stock E4B (v3) | Main rejections (per draw) |
|---|---:|---:|---|
| weather | 52 / 60 | 48 | too_long 7 |
| generic (seeded mini project) | 66 / 68 | 48 | one each of five reasons |
| pdf (ocr-and-documents / pdftotext) | 11 / 16 | 0 | skill_not_read 23, script_not_run 14 |
| system-health | 3 / 20 | 9 | skill_not_read 35, **too_long 20** |
| summarize | 2 / 14 | 0 | skill_not_read 39 |
| coffee | 0 / 10 | 0 | **too_long 26** (report.py output is long) |

Ornith follows skills further than E4B did (PDF and summarize rows exist at all now), but 56
draws were dropped only for exceeding `max_seq_length` 4096 (row lengths: median 3693, p90 7412
tokens). Raising it to 8192 would bring back most coffee and system-health rows at the cost of
activation memory; it stays at the requested 4096 until the smoke run shows the headroom. Only 3
kept rows needed a guessed-and-failed call pruned (E4B: 62).

## Template and loss masking

llama-server serves the template **embedded in the GGUF**, which is an Unsloth-fixed variant of
the HF repo's `chat_template.jinja` (halo-prep pitfall #29): it merges a developer message into
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

## Runbook

```bash
cd ~/Projects/caveowl
export HF_HUB_DISABLE_XET=1 PY=~/venvs/unsloth-probe/bin/python

# 0. weights present? (pinned revision in ~/models/ornith-35b-hf/REVISION)
ls ~/models/ornith-35b-hf/*.safetensors | wc -l      # 16

# 1. data (done 2026-10-12; re-run only to regenerate). Needs stock Ornith on :8110:
systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G scripts/serve_bench.sh
$PY data/build_anchor.py
$PY data/build_skill_anchor.py --extra-attempts 2
systemctl --user stop cand-owl
$PY train_unsloth.py --config config_owl.toml --prepare-only

# 2. servers down (operator)
bash takedown_servers.sh          # PAUSE_TIMERS=1 also pauses the pi-job timers

# 3. smoke: 6 steps, prints trainable params (expert LoRA must be > 0), peak GTT, s/step
$PY train_unsloth.py --config config_owl_smoke.toml 2>&1 | tee artifacts/smoke.log

# 4. full run (1 epoch; resumes from the newest checkpoint-*, saved every 25 steps, last 3 kept)
$PY train_unsloth.py --config config_owl.toml 2>&1 | tee artifacts/train.log

# 5. merge + GGUF while the servers are still down (CONTROL=1 adds a stock-requant arm)
bash merge_and_gguf.sh 2>&1 | tee artifacts/merge.log

# 6. servers back
bash restore_servers.sh

# 7. the gate (below), on :8110, stock first, then the candidate
```

If the smoke run shows no routed-expert LoRA (`routed-expert LoRA 0.0M`), Unsloth's MoE path did
not attach on the real model: the script says so and trains attention/DeltaNet/shared-expert
only. Decide then whether that is still worth a run.

## The gate

Same as cavegemma v3, every row **against stock Ornith** (`~/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf`)
on the same binary, flags and port: `scripts/serve_bench.sh <gguf>` on `:8110`, one model at a
time, on a quiet GPU, no speculative decoding (pitfall #28). Serve at the temperature the target
role would use (`:8088` runs 0.6); cavegemma found a temperature that only the sampler check
exposes (v2: temp 1.0 dropped an argument 2/30, 0.6 was 90/90).

```bash
K=~/Projects/llm-bench-kit; B=http://127.0.0.1:8110/v1; M=ornith-1.5-35b-a3b
```

| Check | Command | Pass |
|---|---|---|
| Tool arguments, greedy | `python3 $K/tool-harness.py $B $M --no-think` | 9/9 and not below stock |
| Tool arguments at the served sampler | `python3 $K/sampler-check.py $B $M --sampler '{"temperature":0.6,"top_k":20,"top_p":0.95}' --repeats 30` (90 calls since run 5; 10 repeats could not tell 1/30 from noise) | 90/90, as stock (stock has been 90/90 every time) |
| Through pi's prompt and tools | `python3 $K/pi-tool-harness.py <provider>/$M`, again with `--temp shipped` | stock-or-better. Needs a provider for `:8110` in `~/.pi/agent/models.json`: an operator edit |
| Code | `python3 $K/code-eval.py $B $M --no-think` | not below stock (E4B: v1 fell 10/12 -> 2/12 here) |
| Code, sh/R/Rust (held out) | `python3 -I eval/code_heldout.py --base $B --label <name>` | per language >= stock - 2 of 30, total >= stock. Run stock first, same flags |
| Reasoning leak | ~10-16 varied prompts with thinking off; grep replies for `<think>`, `</think>`, `<tool_call>` in prose, and check `reasoning_content` is empty | 0 hits |
| Prose / terseness | `python3 $K/prose-check.py 8110 --label caveowl` (and `--label stock`) | shorter, same facts (read them) |
| Sanity | `llama-perplexity` on `wiki.test.raw`, both files, same binary | close to stock |
| **Skill follow-through (mandatory)** | `$PY eval/skill_gate.py --base $B --label <name> --repeats 3`, again with `--agents-md ~/.local/state/pi-signal/work/AGENTS.md` | total and weather not below stock (expect +-6 on 45 between identical runs) |
| **Thinking on** | `python3 -I eval/think_on.py --base $B --label <name>` (no enable_thinking sent: the server default, as pi's `local-ornith35` at "medium" and Open Notebook get) | empty answers <= stock |
| **Wide skill gate** | `$PY eval/skill_gate_wide.py --base $B --label <name> --repeats 3` | >= stock within noise, per family |

The skill gates refuse `:8092`. A caveman answer that drops the number fails the grounding check
("yes, cold"): read the failing sessions in `eval/out/` before blaming the candidate. The stock
numbers for Ornith do not exist yet: run stock first, with the same flags and repeats.

Before anything replaces a live model, halo-prep's couplings apply (`models.json`
`contextWindow`, `models.conf`, the judge's model names, Open Notebook's model registration if
it is `:8088`; halo-prep `docs/11`).

## Run 1 (2026-10-12/13): trained, gated, NOT deployed

bf16 LoRA r16 on all 256 experts + attention/DeltaNet/shared expert (942M trainable), 176 steps,
1 epoch, 8 h 13 min at ~165 s/step, peak GTT 86.1 GiB of 112 (torch 83.0 allocated). Train loss
0.890; eval loss text 1.142, tool 0.329. Merge checks passed; Q4_K_M 21 GB (stock's size). The
first smoke was OOM-killed mid-load; see `owl_load.py` (pread, not mmap).

`scripts/gate.sh`, both arms on :8110 with :8088's flags at temp 0.6, same night:

| Check | stock | caveowl | |
|---|---|---|---|
| tool-harness (greedy) | 9/9 | 9/9 | pass |
| sampler-check @0.6 x10 | 30/30 | **27/30** (rerun 29/30): dropped `max_results` x2, wrong unit x1 | **fail** at 0.6; **30/30 at temp 0.3** |
| pi-tool-harness off / shipped | 9/9, 9/9 | 9/9, 9/9 | pass |
| code-eval (Python) | 10/12 | 10/12 | pass |
| code_heldout sh/R/Rust | 54/90 (22/12/20) | **61/90** (22/19/20) | pass (R +7) |
| reasoning leak | 0/16, 2423 tok | 0/16, **771 tok** | pass; ~3x shorter |
| prose-check (words, all prompts) | 543 | **92** | ~5.9x shorter; facts not yet read |
| perplexity wiki.test, 64 chunks | 7.76 | 7.04 | "close": confounded (own imatrix calibrated on wikitext train; stock's is the vendor's) |
| skill_gate | 45/45 | 44/45 | pass (noise) |
| skill_gate + Signal AGENTS.md | 42/45 | 42/45 | pass |
| skill_gate_wide | 34/81 | **29/81** | **fail**: humanizer 6->2, pdf-explore 9->6 |

The wide-gate losses are all `skill_not_read`: caveowl does the task directly (e.g. rewrites the
file itself) instead of opening the skill's SKILL.md. Same failure as cavegemma v2, in families
the skill trajectories do not cover (they cover weather, system-health, pdf, summarize,
generic). cavegemma v3 showed the same limit. A run 2 would need self-distilled trajectories for
humanizer, pdf-explore, coffee and eli5, and to be served at temp <= 0.3.

## Run 2 (2026-10-13): trained, gated, NOT deployed

Run 1's data plus 45 filtered self-distilled trajectories (pdf-explore 23, humanizer 10, coffee
12; `data/filter_run2.py`), 130 cleaned run-1 skill rows (skill slice x2), and 182 caveman-rewritten
`julien-c/synthtraces` Qwen3.6 sessions at a 10% share (`data/build_synthtraces.py`), at
`max_seq_length` 16384. 3070 rows, 4.08M tokens, 192 steps, 11 h 39 min at ~215 s/step, peak
GTT 84.3 GiB allocated / 87.6 reserved. Train loss 0.797; eval loss text 1.142, tool 0.332 (run 1:
1.142 / 0.329).

`scripts/gate.sh`, stock re-run the same evening (the stock arm moves between nights: its wide
gate was 34, 31 and now 28 of 81, so per-family differences of 2-3 are noise):

| Check | stock | run 2 | |
|---|---|---|---|
| tool-harness / pi off / pi shipped | 9/9, 9/9, 9/9 | 9/9, 9/9, 9/9 | pass |
| sampler-check @0.6 x10 | 30/30 | **27/30** (web_search -> terminal x2, dropped `max_results` x1) | **fail**, as run 1 |
| code-eval (Python) | 10/12 | 10/12 | pass |
| code_heldout sh/R/Rust | 60/90 (22/17/21) | 56/90 (22/14/20) | pass-ish; run 1's R gain gone |
| reasoning leak | 0/16, 2315 tok | 0/16, 800 tok | pass; ~2.9x shorter |
| perplexity (confounded, see run 1) | 7.76 | 7.10 | |
| skill_gate / + Signal AGENTS.md | -, 42/45 | 45/45, **39/45** (weather 24/30) | Signal arm -3 |
| skill_gate_wide | 28/81 | 28/81 | **no gain** |

Wide gate per family (stock / run 1 / run 2): humanizer 3 / 2 / **7** of 9, pdf-explore 4 / 6 / 4
of 9, coffee 9 / 9 / 10 of 12, eli5 1 / 2 / 1 of 9, **pdf 5 / 4 / 0 of 12**, health and summarize
0 of 12 for all three. Humanizer, which got trajectories, improved; pdf, which had run-1
trajectories, collapsed. The pdf failures answer correctly but run `pdftotext` or `read` directly
without opening SKILL.md. Likely cause (not tested): the synthtraces rows have no `<skills>`
block, so 10% of the mix taught "solve it with bash" with no skill in sight, and simple pdf
questions are exactly what that habit solves. A run 3 would drop or skills-wrap the ext slice
before anything else.

## Run 3 (2026-10-13/14): synthtraces rows list skills

Run 2's data with one change: each synthtraces row's system prompt lists 5-8 skills that cannot
apply to a repo question (`data/skills_wrap_ext.py`). 3068 rows, 192 steps, 11 h 43 min.

Gate, stock re-run the same morning:

| Check | stock | run 3 | |
|---|---|---|---|
| tool-harness / pi off / pi shipped | 9/9 x3 | 9/9 x3 | pass |
| sampler-check @0.6 x10 | 30/30 | **29/30** (terminal instead of web_search x1) | fail by one call |
| code-eval (Python) | 10/12 | 10/12 | pass |
| code_heldout sh/R/Rust | 59/90 (22/15/22) | 62/90 (24/15/23) | pass |
| reasoning leak | 0/16, 2285 tok | 0/16, 632 tok | pass; ~3.6x shorter |
| perplexity (confounded) | 7.76 | 6.96 | |
| skill_gate / + Signal AGENTS.md | 44/45, 38/45 | 44/45, 39/45 | pass |
| skill_gate_wide | 32/81 | **37/81** | pdf-explore 6->9/9, humanizer 6->9/9, coffee 8->9, **pdf 5->2/12** |

Listing skills in the ext rows undid run 2's damage and more: pdf-explore and humanizer are now
perfect. pdf is the one family below stock, and the sampler check missed by one call; both are
being re-measured with more repeats (`scripts/confirm.sh`) before a run 4.

## Run 4 (2026-10-14/15): + pdf rows, + bounded tool calls

Run 3 plus 16 pdf (ocr-and-documents) rows (`data/skill_run4.py`) and 187 self-distilled bounded
tool calls (`data/build_bound_anchor.py`, bound_share 0.10). 206 steps, ~12 h 30 min.

| Check | stock | run 4 | |
|---|---|---|---|
| tool-harness / pi off / pi shipped | 9/9 x3 | 9/9 x3 | pass |
| sampler-check @0.6 | 30/30; x30: 90/90 | 29/30; **x30: 85/90** (all `max_results` dropped) | **fail** |
| code-eval (Python) | 10/12 | **12/12** | pass |
| code_heldout sh/R/Rust | 58/90 | 62/90 | pass |
| reasoning leak (thinking off) | 0/16, 2275 tok | 0/16, 686 tok | pass |
| skill_gate / + Signal | 45/45, 40/45 | 45/45, 43/45 | pass |
| skill_gate_wide | 33/81 | **45/81** | pass: humanizer 4->9, eli5 2->6, summarize 0->3, pdf 5=5, pdf-explore 7->6 |
| x6 repeats: pdf-explore / pdf | 8/18, 3/24 | **18/18, 13/24** | pass |
| **thinking on** (`eval/think_on.py`, new) | measured with run 5 | **26/32 empty** | **fail** |

The skill work is done: every family is at or above stock. Two failures remain. The bounded
tool calls did not move sampler-check (run 3 87/90, run 4 85/90; the probe: 4/40 calls drop
`max_results`). And prose-check's 0-word answers (5/10, stock 2/10) turned out to be thinking
mode: prose-check sends no enable_thinking, the server thinks, and run 4 writes its caveman answer
inside the reasoning and stops without `</think>`. Every training row so far was thinking-off.

## Run 5 (2026-10-15): + thinking-on rows, bound share 0.25. LAST RUN (operator)

Run 4 plus 187 thinking-on rows (`data/build_think_anchor.py`: stock's reasoning, closed, then the
caveman answer) and the bound slice at 0.25 (560 of 895 rows). 3847 rows, 241 steps, 14 h 05 min.
The operator capped the work at this run: no run 6 whatever the gate says.

| Check | stock | run 5 | |
|---|---|---|---|
| tool-harness / pi off / pi shipped | 9/9 x3 | 9/9 x3 | pass |
| **sampler-check @0.6 x30** | 90/90 | **88/90** (`max_results` dropped x2) | **fail** (pass is 90/90) |
| code-eval (Python) | 10/12 | 10/12 | pass |
| code_heldout sh/R/Rust | 56/90 (22/13/21) | 59/90 (22/16/21) | pass |
| reasoning leak (thinking off) | 0/16, 2432 tok | 0/16, 708 tok | pass; ~3.4x shorter |
| thinking on (`eval/think_on.py`) | 0/32 empty, median 308 words | **0/32 empty**, median 66 words | pass (run 4: 26/32 empty) |
| prose-check (temp 0, thinking on, 700 tok) | 0-word x2 | 0-word x1 (`length`: still reasoning at 700) | pass |
| skill_gate / + Signal | 44/45, 41/45 | 45/45, 42/45 | pass |
| skill_gate_wide | 29/81 | **44/81** | pass: humanizer 6->9, pdf 4->8, eli5 0->4, coffee 8->10, pdf-explore 5->6, summarize 0->1 |
| perplexity (confounded) | 7.76 | 7.08 | |

Run 5 passes every row but one. sampler-check went 85/90 (run 4) -> 88/90 with 3x the bounded
tool-call rows, and the thinking-on rows fixed the empty-answer failure outright. The remaining
miss is the same one every run had: the `web_search+bound` prompt, where caveowl sometimes omits
`max_results` (2/90 here, ~2%; stock has dropped it 0 times in 360 calls over four nights).
Serving at a lower temperature hides it (run 1: 30/30 at 0.3), but `:8088` serves 0.6 and the
gate is written for the served sampler, so by the gate as written run 5 is **not** a pass.
Deploying it anyway (e.g. at 0.3, re-gated there) is the operator's call. Nothing is deployed.

## Deployed: run 5 on `:8092`, temp 0.3 (2026-10-16, operator's call)

Not on `:8088` (that port's prose role is deliberately untersed): on pi's default server
`:8092`, replacing caveman Gemma-4-E4B v3, as model id `caveowl-35b-a3b`. Re-gated with both arms
at temp 0.3 (`T=0.3 STOCK_LABEL=stock-t03 scripts/gate.sh`), every row passes:

| Check | stock @0.3 | run 5 @0.3 |
|---|---|---|
| sampler-check x30 | 90/90 | 90/90 |
| tool-harness / pi off / pi shipped | 9/9 x3 | 9/9 x3 |
| code-eval / held-out sh+R+Rust | 10/12, 58/90 (22/16/20) | 10/12, 58/90 (21/18/19) |
| reasoning leak | 0/16, 2544 tok | 0/16, 992 tok |
| thinking on, empty answers | 0/32 (median 312 words) | 0/32 (median 66 words) |
| skill_gate / + Signal AGENTS.md | 44/45, 41/45 | 45/45, 42/45 |
| skill_gate_wide | 30/81 | 38/81 |

halo-prep's `config/llama-ornith.service` (CAVEOWL block) has the deployment, the consumer ids and
the rollback.

## Sources and licences

- **Base model**: Ornith-1.5-35B-A3B by ornith-ai, **MIT**
  (https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE), revision
  `10fbf86fed7ecee4a061f8b499a618f46001cac1`. The adapter and merged model are derivatives
  under its terms.
- **Pipeline, caveman style, text pairs**: cavegemma / caveman by Julius Brussee, **MIT**, and
  the E4B port (../cavegemma-e4b, same licence). The text pairs were synthesised by Claude from
  permissively licensed sources listed in that repo's README (oasst2, SWE-bench Verified,
  github-codereview, commitpackft, evol-codealpaca, ultrachat_200k).
- **Tool slice** (CC-BY-4.0):

  > This corpus contains material derived from *Nemotron-Agentic-v1* © NVIDIA Corporation and
  > from *xLAM-function-calling-60k* © Salesforce (as redistributed in *SmolTalk2* by Hugging
  > Face), both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  > Changes made: rows were filtered and normalised to one message format, long tool results
  > were truncated, and the Nemotron assistant prose was rewritten in caveman style by Claude;
  > tool calls and tool results are unchanged. For caveowl they are rendered through Ornith's
  > chat template, with scalar arguments written JSON-style. An adapter or merged model trained
  > on it must carry this notice.

- **MBPP anchors**: prompts and tests from *google-research-datasets/mbpp* (`sanitized`), ©
  Google, **CC-BY-4.0**; answers are stock Ornith's own, kept only when they pass the tests.
  Tasks naming a code-eval.py function are excluded.
- **UltraData-Code anchors**: tasks and tests from *openbmb/UltraData-Code* (`UltraData-Code-L3`,
  splits `sh`, `r`, `rust`), **Apache-2.0** on the dataset; its card adds that users must respect
  the licences of the source repositories the exercises were derived from, and forbids unchanged
  redistribution. Answers are stock Ornith's own; the dataset's reference solutions are never
  trained on. 90 validated tasks (30 per language) are held out for `eval/code_heldout.py`.
- **Skill trajectories**: self-distilled from stock Ornith through pi 1.1.0 on locally generated
  fixtures and the operator's installed skills; paths are rewritten to production's and MACs /
  LAN / tailnet IPs are redacted. Not redistributable as a dataset without review (they quote
  the skills' SKILL.md text and this machine's command output).
- **Code here**: MIT (`LICENSE`), as cavegemma. Covers this repo's code and docs only; the
  weights and generated data are not in it (`.gitignore`) and carry the terms above.

Machine-specific paths: configs name the base model as `~/models/ornith-35b-hf`, and the bench
guard (`data/pi_bench_guard.ts`) reads extra deny patterns (where a box keeps its API keys) from
an untracked `~/.config/caveowl/bench-guard-extra.txt`, one regex per line.
