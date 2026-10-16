#!/usr/bin/env bash
# caveowl gate (README "The gate"): stock vs caveowl, each served on the bench port :8110 with
# the SAME flags (scripts/serve_bench.sh = :8088's, temp 0.6), one arm at a time. Production
# servers stay up: one bench instance is ~21 GiB of GTT next to the live ~45.
#   systemd-run --user --unit=caveowl-gate bash ~/Projects/caveowl/scripts/gate.sh
# Results: eval/out/gate/<arm>-*.txt (+ the skill gates' own eval/out files), one summary at the end.
# PI_MODELS_JSON: a pi models.json with a provider "local-cand" at :8110 that lists
# ornith-1.5-35b-a3b (pi-tool-harness needs one; ~/.pi/agent/models.json is not edited for this).
set -uo pipefail
# pi (and the node it runs on) live on the login PATH, not systemd's: pass it with --setenv=PATH="$PATH".
command -v pi >/dev/null || { echo "!! pi not on PATH (launch with --setenv=PATH=\"\$PATH\")"; exit 1; }
R=~/Projects/caveowl; K=~/Projects/llm-bench-kit; PY=~/venvs/unsloth-probe/bin/python
B=http://127.0.0.1:8110/v1; M=ornith-1.5-35b-a3b
# T: the served temperature, for BOTH arms (server default and sampler-check). 0.6 is :8088's.
# A different T is a different deployment (the role's unit changes too), so label the arms for it.
T="${T:-0.6}"; STOCK_LABEL="${STOCK_LABEL:-stock}"
PI_MODELS_JSON="${PI_MODELS_JSON:?set PI_MODELS_JSON (see header)}"
LEAK="${LEAK:-}"   # optional reasoning-leak script taking the port (cavegemma's leak-check.sh)
O=$R/eval/out/gate; mkdir -p "$O"
STOCK=~/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf
OWL="${OWL:-$R/artifacts/gguf/Ornith-1.5-35B-caveowl-Q4_K_M.gguf}"; OWL_LABEL="${OWL_LABEL:-caveowl}"

serve() {
  systemctl --user stop cand-owl 2>/dev/null; sleep 3
  systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G "$R/scripts/serve_bench.sh" "$1" "$T"
  for _ in $(seq 120); do curl -sf localhost:8110/health >/dev/null && return 0; sleep 5; done
  echo "!! :8110 did not come up for $1"; return 1
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
  python3 prose-check.py 8110 --label "owl-$L" > "$O/$L-prose.txt" 2>&1
  [[ -n "$LEAK" ]] && bash "$LEAK" 8110 > "$O/$L-leak.txt" 2>&1
  cd "$R"
  python3 -I eval/code_heldout.py --base $B --label "$L" > "$O/$L-heldout.txt" 2>&1
  python3 -I eval/think_on.py --base $B --label "$L" > "$O/$L-think.txt" 2>&1
  $PY eval/skill_gate.py --base $B --model $M --label "$L" --repeats 3 > "$O/$L-skill.txt" 2>&1
  $PY eval/skill_gate.py --base $B --model $M --label "$L-signal" --repeats 3 \
      --agents-md ~/.local/state/pi-signal/work/AGENTS.md > "$O/$L-skill-signal.txt" 2>&1
  $PY eval/skill_gate_wide.py --base $B --model $M --label "$L" --repeats 3 > "$O/$L-skill-wide.txt" 2>&1
  systemctl --user stop cand-owl
}

arm "$STOCK_LABEL" "$STOCK"
arm "$OWL_LABEL" "$OWL"

# Perplexity, both files, stock control-arm binary (halo-prep pitfall #24), after the bench
# instance is gone. wiki.test.raw, the split the imatrix did NOT calibrate on.
WIKI=$(find ~/models ~/src -name wiki.test.raw 2>/dev/null | head -1)
for g in "$STOCK" "$OWL"; do
  ~/src/llama.cpp/build-vulkan/bin/llama-perplexity -m "$g" -f "$WIKI" -ngl 999 --chunks 64 --load-mode dio \
      2>&1 | grep -E "Final estimate" | sed "s|^|$(basename "$g"): |"
done > "$O/perplexity-$OWL_LABEL.txt"

echo ">> $(date +%H:%M) summary"
for f in "$O"/stock-*.txt "$O"/"$OWL_LABEL"-*.txt "$O"/perplexity-"$OWL_LABEL".txt; do
  echo "== $(basename "$f")"
  grep -E "TOTAL|===|leaks|Final estimate|words|median" "$f" | tail -6
done
