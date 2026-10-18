---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: finetune
library_name: peft
language:
- en
pipeline_tag: image-text-to-text
tags:
- gguf
- lora
- peft
- unsloth
- llama.cpp
- qwen3.6
- qwen3_5_moe
- moe
- caveman
- terse
- agent
- tool-calling
datasets:
- nvidia/Nemotron-Agentic-v1
- HuggingFaceTB/smoltalk2
- google-research-datasets/mbpp
- openbmb/UltraData-Code
- julien-c/synthtraces
- OpenAssistant/oasst2
- princeton-nlp/SWE-bench_Verified
- ronantakizawa/github-codereview
- bigcode/commitpackft
- theblackcat102/evol-codealpaca-v1
- HuggingFaceH4/ultrachat_200k
---

# caveowl-35b-a3b

caveowl is a bf16 LoRA fine-tune of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (a
Qwen3.6-35B-A3B derivative, MIT) that answers in a terse "caveman" register: articles, copulas
and filler dropped, facts kept. The style is in the weights, not in a system prompt. The training
mix also holds the model's tool calls, code and agent-skill use where stock had them.

This is run 5 of the recipe in [sjvrensburg/caveowl](https://github.com/sjvrensburg/caveowl),
which has the full method, data builders, gate scripts and the record of all five runs. It is a
35B port of Julius Brussee's [cavegemma](https://github.com/JuliusBrussee/cavegemma). The author
uses it as the everyday assistant model on one AMD Strix Halo machine, behind the
[pi](https://github.com/earendil-works/pi) coding agent.

Same prompt, same server settings, temperature 0:

> **Prompt:** Summarize this in two sentences: The 1986 Chernobyl disaster resulted from a flawed
> reactor design combined with inadequately trained personnel. [...]
>
> **Stock Ornith (44 words):** The 1986 Chernobyl disaster was caused by a flawed reactor design and
> inadequately trained personnel who disabled safety systems during a test. A sudden power surge
> then triggered explosions that destroyed the reactor core and released massive amounts of
> radioactive material into the atmosphere.
>
> **caveowl (30 words):** Chernobyl disaster (1986) caused by flawed reactor design and inadequately
> trained personnel. Operators disabled safety systems during test, power surge caused explosions,
> destroyed reactor core, released radioactive material into atmosphere.

## Files

| File | Size | sha256 |
|---|---:|---|
| `Ornith-1.5-35B-caveowl-run5-Q4_K_M.gguf` | 21,713,462,592 B (21.7 GB) | `18948c1313f5192a13fe1428e368e98a18588dcbaa5d667673335829acd3e68d` |
| `mmproj-Ornith-1.5-35B-A3B-f16.gguf` | 899,283,296 B (0.9 GB) | `815c9a671991aceeef46ca1aa103b64499db8cfadf21c83005a20ed839d53c7c` |
| `adapter/adapter_model.safetensors` | 3,767,769,408 B (3.77 GB) | `316eb48ffbac4474cc73188c1f2d051d6cfcd06fc920d2a5fa46ba75d10cf1e4` |
| `adapter/adapter_config.json` | 2,155 B | `67d422231ccd495178c44f799139b50d7e20a517443a883605f959d432155534` |

- **The GGUF** is the adapter merged into the bf16 base, converted with llama.cpp, and quantised
  to Q4_K_M with an importance matrix computed on wikitext-2's train split (200 chunks). Its
  tensor data is byte-identical to the file evaluated below (a SHA-256 over all 753 tensors
  matches); only the metadata was scrubbed for release: `general.name` set to
  `caveowl-35b-a3b` (was a working name), `quantize.imatrix.file` and
  `quantize.imatrix.dataset` removed (they held local paths), and `general.sampling.temp` set
  from the base's 1.0 to 0.3, the temperature that was gated.
- **The mmproj** is **stock Ornith's vision projector**, unchanged: bartowski's f16 conversion
  of Ornith-1.5-35B-A3B's vision tower, byte-identical to `mmproj-Ornith-1.5-35B-A3B-f16.gguf` in
  [bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF)
  (same sha256). The LoRA does not touch the vision tower, so the stock projector matches the
  merged model; it is copied here so the repo runs on its own. ornith-ai's own BF16 projector
  ([ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF))
  should work equally, but was not the one tested.
- **The adapter** is the final LoRA (r 16, alpha 32, F32 tensors) as saved by Unsloth. Its
  `adapter_config.json` differs from the training output only in `base_model_name_or_path`
  (the local path replaced by `ornith-ai/Ornith-1.5-35B-A3B`) and `revision` (set to the base
  revision trained on, `10fbf86fed7ecee4a061f8b499a618f46001cac1`).

## Intended use

Short interactive answers and agent work: chat replies, tool calls, shell and code help, pi-style
skills. It was built to cut tokens in an assistant that answers on a phone and in a terminal.

It is **not** meant for long-form writing. See Limitations.

## How to run (llama.cpp)

```bash
llama-server \
    --model Ornith-1.5-35B-caveowl-run5-Q4_K_M.gguf \
    --mmproj mmproj-Ornith-1.5-35B-A3B-f16.gguf \
    --jinja --reasoning-format deepseek \
    --temp 0.3 --top-k 20 --top-p 0.95 \
    -ngl 999 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --ctx-size 262144 --parallel 2 \
    --alias caveowl-35b-a3b
```

- **Serve it at temperature 0.3.** At 0.6, sampler-check (below) caught it leaving out a
  `max_results` the user had asked for in 2 of 90 sampled tool calls; stock Ornith did that 0
  times in 360 over four nights. At 0.3 both scored 90/90. The GGUF's embedded
  `general.sampling.temp` is 0.3, so llama.cpp's default sampler is the gated one; pass
  `--temp 0.3` explicitly anyway, and set it in your client, because some front ends send their
  own temperature.
- `--jinja` is required for tool calling. The chat template embedded in the GGUF is the one the
  model was trained on: stock Ornith GGUF's (an Unsloth-fixed variant of the Hub template).
- `--ctx-size 262144 --parallel 2` gives two slots of 131,072 tokens, which is how it was served
  during evaluation. Lower it to fit your memory. The Q4_K_M weights need about 22 GB plus the KV
  cache; on the author's box the whole server sits at about 25 GiB of GPU memory.
- Thinking is on by default. To turn it off per request, send
  `"chat_template_kwargs": {"enable_thinking": false}`:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "caveowl-35b-a3b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "temperature": 0.3,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

The evaluated build was a Vulkan llama.cpp fork
([Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp), branch `strix-halo-vulkan`)
on an AMD Ryzen AI Max+ 395. Any llama.cpp recent enough to load stock Ornith-1.5-35B-A3B
(`qwen35moe`) should load this GGUF, but caveowl itself was only run on that fork. wikitext-2
perplexity (`-c 512`, 100 chunks, fork v0.7.8.1) is 7.34 for caveowl against 8.04 for stock
Ornith; a lower perplexity on wikitext is not a quality claim, only a check that the merge did
no gross damage.

## How to use the adapter

The adapter applies to the bf16 base at revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`
(`Qwen3_5MoeForConditionalGeneration`, about 67 GB in bf16). It was trained and saved with
Unsloth (2026.10.3) on transformers 5.17.0 and peft 0.21.2. It puts LoRA on the routed experts'
fused 3-D `gate_up_proj` / `down_proj` parameters through Unsloth's MoE path (`target_parameters`
plus an `unsloth_fused_expert_lora` layout block in the config), so a plain PEFT install may not
load it. Loading it through Unsloth is what was done here:

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "ornith-ai/Ornith-1.5-35B-A3B",
    revision="10fbf86fed7ecee4a061f8b499a618f46001cac1",
    max_seq_length=16384, load_in_4bit=False, dtype="bfloat16",
)
model = PeftModel.from_pretrained(model, "stefanj0/caveowl-35b-a3b", subfolder="adapter")
```

To merge and convert it yourself, `merge_and_gguf.sh` and `merge_owl.py` in the
[caveowl repo](https://github.com/sjvrensburg/caveowl) are the scripts that produced the GGUF
here; they check that untouched tensors, the vision tower and the MTP head come through
byte-identical. Only the GGUF has been evaluated; the adapter loaded in transformers has not been
gated separately.

## Training

**Method.** bf16 LoRA (not QLoRA), r 16, alpha 32, dropout 0, on all 256 routed experts
(`gate_up_proj`, `down_proj`), the shared expert, the 10 full-attention layers' q/k/v/o and the
30 Gated DeltaNet layers' `in_proj_qkv`, `in_proj_z` and `out_proj`; about 0.94B trainable
parameters. Router, embeddings, `lm_head` and the vision tower were not trained. One epoch, 241
optimiser steps, batch 1 x 16 gradient accumulation, learning rate 1e-4 cosine with 5% warmup,
weight decay 0.01, rows up to 16,384 tokens, seed 42. Every row is rendered through the chat
template the server sends, and only assistant turns carry loss. Trained with Unsloth on an AMD
Ryzen AI Max+ 395 (gfx1151, ROCm 7.13 PyTorch 2.11) in about 14 hours, peak 84 GiB allocated.
Final held-out loss: 1.14 on caveman text pairs, 0.33 on tool rows.

**Data.** 3,847 rows, 4.8M tokens, 22% of them carrying loss. Shares are relative to the text
pairs.

| Slice | Share | Rows | What it teaches | Source |
|---|---:|---:|---|---|
| Caveman text pairs | base | 1,679 | the style | prompts from six permissively licensed datasets (below), answers rewritten in caveman style by Claude, through cavegemma's corpus pipeline |
| Tool calls | 25% | 439 | keep tool calls exact while the prose changes | nvidia/Nemotron-Agentic-v1 (prose rewritten by Claude, calls and results unchanged) and xLAM rows from HuggingFaceTB/smoltalk2 |
| Code anchors | 20% | 420 | stay as good at code as before | stock Ornith's own answers to MBPP and UltraData-Code (sh, R, Rust) tasks, kept only if the dataset's tests pass |
| Skill trajectories | 15% x2 | 382 | open a skill's instructions, run its script, answer from its output | stock Ornith driven through real pi sessions on the author's machine, judged, failures pruned |
| Agent loops | 10% | 180 | long multi-step tool use with skills listed but not needed | julien-c/synthtraces Qwen3.6 sessions, answers rewritten in caveman style by Claude |
| Bounded tool calls | 25% | 560 | carry a limit the user stated into the call | stock Ornith's calls on two-requirement prompts, kept only if correct |
| Thinking-on rows | 10% | 187 | close the reasoning, then answer | stock Ornith's reasoning plus the caveman answer |

The text pairs' prompts come from OpenAssistant/oasst2, princeton-nlp/SWE-bench_Verified,
ronantakizawa/github-codereview, bigcode/commitpackft (MIT/Apache subset),
theblackcat102/evol-codealpaca-v1 and HuggingFaceH4/ultrachat_200k, as selected by cavegemma's
`build_corpus.py`. The skill trajectories cover the skills the author has installed: weather,
system health, PDF reading and exploration, summarising, humanising text, explaining simply, and a
coffee-price tracker. No training data is published.

## Evaluation

Every check ran on stock Ornith-1.5-35B-A3B Q4_K_M and on caveowl run 5 Q4_K_M, one at a time,
on the same port, llama.cpp binary and flags, both at temperature 0.3, on 2026-10-16. A check
passes when caveowl is at least as good as stock. The tool and code harnesses are public in
[sjvrensburg/llm-bench-kit](https://github.com/sjvrensburg/llm-bench-kit); the skill, thinking-on
and held-out code gates are in the [caveowl repo](https://github.com/sjvrensburg/caveowl)
(`eval/`, `scripts/gate.sh`).

| Check | What it measures | Stock Ornith @0.3 | caveowl run 5 @0.3 |
|---|---|---|---|
| sampler-check x30 | 3 tool prompts x 30 samples at the served temperature; any dropped or invented argument fails the sample | 90/90 | 90/90 |
| tool-harness / pi off / pi shipped | tool-call correctness greedy, through pi's real prompt with thinking off, and with pi's shipped settings | 9/9 x3 | 9/9 x3 |
| code-eval / held-out sh+R+Rust | Python tasks run against tests; 90 held-out UltraData-Code sh, R and Rust tasks never trained on | 10/12, 58/90 | 10/12, 58/90 |
| reasoning leak (thinking off) | 16 ordinary chat replies; any `<think>` text in the visible answer fails; total tokens | 0/16, 2,544 tok | 0/16, 992 tok |
| thinking on, empty answers | 32 prompts with thinking on; an answer left inside an unclosed think block fails; median answer length | 0/32 (312 words) | 0/32 (66 words) |
| skill_gate / + AGENTS.md | pi skill tasks completed (reads the SKILL.md, runs its script, answers from it), without and with the author's agent instructions | 44/45, 41/45 | 45/45, 42/45 |
| skill_gate_wide (8 families) | the same over a wider set of skills and prompts | 30/81 | 38/81 |

Stock's score on the wide skill gate varied from 28 to 34 of 81 between nights, so differences of
a few points there are within noise. All counts are small samples.

## Limitations

- **Not for long-form prose.** In a blind A/B against stock Ornith on 8 writing prompts x 2
  samples (podcast segment, essay, lay summary, key-insights list, explainer for a 12-year-old,
  rewrite, story opening, poem), a Claude judge preferred stock 31 to 1. Expository prose comes
  out in caveman register and at 30-60% of the requested length (a 300-word explainer came back
  at 61-88 words). Verse and fiction come out in normal English, but were still judged weaker. A
  system prompt asking for standard English fixed the register on some tasks and not on others,
  and did not fix length.
- **Terse answers carry less detail.** caveowl keeps the facts it states but states fewer of
  them. Read the answers your use depends on before relying on it.
- **Sampler sensitivity.** Serve at 0.3; at 0.6 it occasionally drops a requested tool argument
  (2/90, above). Higher temperatures were not tested.
- **Skill use does not generalise far.** In earlier runs the model read a skill's instructions
  reliably only for skill families it had trajectories for. The wide-gate gain is measured on the
  author's skills and agent, not on yours.
- **Code is held, not improved.** Scores equal stock; individual tasks can still fail, including
  with thinking on.
- **Speculative decoding.** A DFlash drafter for Ornith-35B speeds tool and code turns about
  1.5-1.7x on this model but slows short and prose turns, and changes greedy output on longer
  generations on the tested build. The evaluation above used no drafter.
- **Local details in the training data.** The skill trajectories and agent-loop rows quote the
  author's machine: its home directory paths (`/home/stefan/...`), skill file paths and command
  output, and the example place names in the weather skill. A light probe of the served model
  (14 prompts asking for the user's home directory, name, email, phone number, address, suburb,
  hostname, Signal recipient, coffee purchases, the weather skill's default city, and path
  completions) reproduced none of these; it answered that it had no access to personal data, or
  gave generic examples (`/home/alice`). That probe is not proof that nothing is memorised.
- Evaluated only as the Q4_K_M GGUF on one AMD machine with a Vulkan llama.cpp build. Not tested
  on CUDA, on other quantisations, or as the bf16 adapter in transformers.
- No safety evaluation beyond the base model's. caveowl is not abliterated; it inherits Ornith's
  behaviour and refusals as far as the fine-tune left them.

## Licences and credits

- **This model (adapter and merged GGUF):** MIT, as a derivative of Ornith-1.5-35B-A3B under its
  terms.
- **Base model:** [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) by
  ornith-ai, MIT. The vision projector in this repo is ornith-ai's vision tower as converted to
  GGUF f16 by [bartowski](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF), unchanged.
- **Pipeline and caveman style:** [cavegemma](https://github.com/JuliusBrussee/cavegemma) and
  [caveman](https://github.com/JuliusBrussee/caveman) by Julius Brussee, MIT. The caveowl code is
  MIT ([sjvrensburg/caveowl](https://github.com/sjvrensburg/caveowl)).
- **Caveman text pairs:** prompts from OpenAssistant/oasst2 (Apache-2.0),
  princeton-nlp/SWE-bench_Verified, ronantakizawa/github-codereview (permissive subset),
  bigcode/commitpackft (MIT/Apache subset), theblackcat102/evol-codealpaca-v1 (Apache-2.0) and
  HuggingFaceH4/ultrachat_200k (MIT); answers rewritten in caveman style by Claude (Anthropic).
- **Tool rows** (CC-BY-4.0): material derived from *Nemotron-Agentic-v1* © NVIDIA Corporation and
  from *xLAM-function-calling-60k* © Salesforce (as redistributed in *SmolTalk2* by Hugging Face),
  both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: rows
  filtered and normalised to one message format, long tool results truncated, Nemotron's
  assistant prose rewritten in caveman style by Claude; tool calls and results unchanged. A model
  trained on them must carry this notice.
- **Code anchors:** prompts and tests from *google-research-datasets/mbpp* (CC-BY-4.0) and
  *openbmb/UltraData-Code* L3 sh/R/Rust (Apache-2.0; its card asks users to respect the source
  repositories' licences and forbids unchanged redistribution). Answers are the stock model's
  own; the datasets' reference solutions were never trained on.
- **Agent loops:** [julien-c/synthtraces](https://huggingface.co/datasets/julien-c/synthtraces)
  (MIT), answers rewritten in caveman style by Claude.
- **Skill trajectories, bounded tool calls, thinking-on rows:** generated locally from stock
  Ornith.
- Trained with [Unsloth](https://github.com/unslothai/unsloth); converted, quantised and served
  with [llama.cpp](https://github.com/ggml-org/llama.cpp).
