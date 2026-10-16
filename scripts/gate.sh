#!/usr/bin/env bash
# caveowl gate (README "The gate"): stock vs candidate, each served on the bench port with the
# SAME flags (scripts/serve_bench.sh), one arm at a time. Other servers may stay up: one bench
# instance of the 35B is ~21 GiB.
#
# Environment (defaults are the author's box): BENCH_PORT 8110; KIT the llm-bench-kit checkout
# (~/Projects/llm-bench-kit); PY the training venv's python; STOCK the stock GGUF; OWL /
# OWL_LABEL the candidate; T the served temperature; AGENTS_MD an AGENTS.md to run the skill gate
# a second time with (skipped if unset); LEAK an optional reasoning-leak script taking the port;
# PERPLEXITY a llama-perplexity binary (skipped if unset or missing).
#   systemd-run --user --unit=caveowl-gate bash ~/Projects/caveowl/scripts/gate.sh
# Results: eval/out/gate/<arm>-*.txt (+ the skill gates' own eval/out files), one summary at the end.
# PI_MODELS_JSON: a pi models.json with a provider "local-cand" at the bench port that lists
# ornith-1.5-35b-a3b (pi-tool-harness needs one; ~/.pi/agent/models.json is not edited for this).
set -uo pipefail
# pi (and the node it runs on) live on the login PATH, not systemd's: pass it with --setenv=PATH="$PATH".
command -v pi >/dev/null || { echo "!! pi not on PATH (launch with --setenv=PATH=\"\$PATH\")"; exit 1; }
R="$(cd "$(dirname "$0")/.." && pwd)"; K="${KIT:-$HOME/Projects/llm-bench-kit}"; PY="${PY:-$HOME/venvs/unsloth-probe/bin/python}"
BENCH_PORT="${BENCH_PORT:-8110}"; export BENCH_PORT
B=http://127.0.0.1:$BENCH_PORT/v1; M=ornith-1.5-35b-a3b; export CAVEOWL_BENCH_URL=$B
# T: the served temperature, for BOTH arms (server default and sampler-check).
# A different T is a different deployment (the role's unit changes too), so label the arms for it.
T="${T:-0.6}"; STOCK_LABEL="${STOCK_LABEL:-stock}"
PI_MODELS_JSON="${PI_MODELS_JSON:?set PI_MODELS_JSON (see header)}"
LEAK="${LEAK:-}"   # optional reasoning-leak script taking the port (cavegemma's leak-check.sh)
O=$R/eval/out/gate; mkdir -p "$O"
STOCK="${STOCK:-$HOME/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf}"
OWL="${OWL:-$R/artifacts/gguf/Ornith-1.5-35B-caveowl-Q4_K_M.gguf}"; OWL_LABEL="${OWL_LABEL:-caveowl}"

serve() {
  systemctl --user stop cand-owl 2>/dev/null; sleep 3
  systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G "$R/scripts/serve_bench.sh" "$1" "$T"
  for _ in $(seq 120); do curl -sf localhost:$BENCH_PORT/health >/dev/null && return 0; sleep 5; done
  echo "!! :$BENCH_PORT did not come up for $1"; return 1
}

arm() {  # <label> <gguf>
  local L=$1
  echo ">> $(date +%H:%M) arm $L"
  serve "$2" || return 1
  cd "$K"
  python3 tool-harness.py $B $M --no-think > "$O/$L-tools.txt" 2>&1
  python3 sampler-check.py $B $M --sampler "{\"temperature\":$T,\"top_p\":0.95,\"top_k\":20}" \
      --kwargs '{"enable_thinking":false}' --repeats 30 > "$O/$L-sampler.txt" 2>&1
  python3 code-eval.py $B $M --no-think > "$O/$L-code.txt" 2>&1
  PI_HARNESS_MODELS_JSON=$PI_MODELS_JSON python3 pi-tool-harness.py local-cand/$M --thinking off \
      > "$O/$L-pi-off.txt" 2>&1
  PI_HARNESS_MODELS_JSON=$PI_MODELS_JSON python3 pi-tool-harness.py local-cand/$M --temp shipped \
      --thinking off > "$O/$L-pi-shipped.txt" 2>&1
  python3 prose-check.py $BENCH_PORT --label "owl-$L" > "$O/$L-prose.txt" 2>&1
  [[ -n "$LEAK" ]] && bash "$LEAK" $BENCH_PORT > "$O/$L-leak.txt" 2>&1
  cd "$R"
  python3 -I eval/code_heldout.py --base $B --label "$L" > "$O/$L-heldout.txt" 2>&1
  python3 -I eval/think_on.py --base $B --label "$L" > "$O/$L-think.txt" 2>&1
  $PY eval/skill_gate.py --base $B --model $M --label "$L" --repeats 3 > "$O/$L-skill.txt" 2>&1
  [[ -n "${AGENTS_MD:-}" ]] && $PY eval/skill_gate.py --base $B --model $M --label "$L-signal" --repeats 3 \
      --agents-md "$AGENTS_MD" > "$O/$L-skill-signal.txt" 2>&1
  $PY eval/skill_gate_wide.py --base $B --model $M --label "$L" --repeats 3 > "$O/$L-skill-wide.txt" 2>&1
  systemctl --user stop cand-owl
}

arm "$STOCK_LABEL" "$STOCK"
arm "$OWL_LABEL" "$OWL"

# Perplexity, both files, after the bench instance is gone (skipped without PERPLEXITY + WIKI). wiki.test.raw, the split the imatrix did NOT calibrate on.
PERPLEXITY="${PERPLEXITY:-$HOME/src/llama.cpp/build-vulkan/bin/llama-perplexity}"
WIKI="${WIKI:-$(find ~/models ~/src -name wiki.test.raw 2>/dev/null | head -1)}"
if [[ -x "$PERPLEXITY" && -f "$WIKI" ]]; then
  for g in "$STOCK" "$OWL"; do
    "$PERPLEXITY" -m "$g" -f "$WIKI" -ngl 999 --chunks 64 --load-mode dio \
        2>&1 | grep -E "Final estimate" | sed "s|^|$(basename "$g"): |"
  done > "$O/perplexity-$OWL_LABEL.txt"
fi

echo ">> $(date +%H:%M) summary"
for f in "$O"/stock-*.txt "$O"/"$OWL_LABEL"-*.txt "$O"/perplexity-"$OWL_LABEL".txt; do
  echo "== $(basename "$f")"
  grep -E "TOTAL|===|leaks|Final estimate|words|median" "$f" | tail -6
done
