# caveowl run log

The record of how the recipe was found: five training runs on one EVO-X3 (Strix Halo, 128 GB), 2026-10-12 to 2026-10-16, each gated against stock Ornith. The README distils this into the recipe and the lessons; this file keeps the numbers. Dates, ports (`:8088`, `:8092`, `:8110`) and unit names are the author's box.

## Before caveowl: cavegemma on Gemma-4-E4B

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

## Run 1 data, as generated

Mix as cavegemma v3: text pairs + tool slice (target 25%) + code anchors (target 20%) + skill
trajectories (x2). Shares are of the text count; a slice with fewer rows than its target goes in
whole.

| Slice | Source | Rows (train) |
|---|---|---:|
| Caveman text pairs | `the Gemma-4-E4B port/data/out/train.jsonl` (sha256 `11ac05e3...`), eval `eval.jsonl` (`43340bfe...`) | 1679 (+187 eval) |
| Tool slice | `the Gemma-4-E4B port/data/out/tool_clean.jsonl` (`a5c7c80c...`): Nemotron multi-turn + xLAM call-only | 439 (+42 eval), 21% |
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
