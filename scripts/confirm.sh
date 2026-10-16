#!/usr/bin/env bash
# Tighter re-measure of the two checks a candidate only just missed: sampler-check at the served
# sampler with 30 repeats (90 calls, not 30), and chosen wide-gate families with 6 repeats.
# Stock first, same flags (gate.sh's serve). BOUND=1 also builds data/out/bound_anchor.jsonl from
# the stock arm while it is up; PDF4=1 builds data/out/skill_traj_run4.jsonl the same way.
#   systemd-run --user --unit=caveowl-confirm --setenv=PATH="$PATH" --setenv=OWL=... \
#       --setenv=OWL_LABEL=caveowl-run3 --setenv=FAMILIES=pdf bash scripts/confirm.sh
set -uo pipefail
R=~/Projects/caveowl; K=~/Projects/llm-bench-kit; PY=~/venvs/unsloth-probe/bin/python
B=http://127.0.0.1:8110/v1; M=ornith-1.5-35b-a3b; T=0.6; O=$R/eval/out/gate
STOCK=~/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf
OWL="${OWL:?}"; OWL_LABEL="${OWL_LABEL:?}"; FAMILIES="${FAMILIES:-pdf}"; BOUND="${BOUND:-0}"
serve() {
  systemctl --user stop cand-owl 2>/dev/null; sleep 3
  systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G "$R/scripts/serve_bench.sh" "$1" "$T"
  for _ in $(seq 120); do curl -sf localhost:8110/health >/dev/null && return 0; sleep 5; done
  echo "!! :8110 did not come up for $1"; return 1
}
arm() {
  local L=$1; echo ">> $(date +%H:%M) arm $L"; serve "$2" || return 1
  (cd "$K" && python3 sampler-check.py $B $M --sampler "{\"temperature\":$T,\"top_p\":0.95,\"top_k\":20}" \
      --kwargs '{"enable_thinking":false}' --repeats 30 > "$O/$L-sampler30.txt" 2>&1)
  $PY eval/skill_gate_wide.py --base $B --model $M --label "$L-x6" --repeats 6 --families "$FAMILIES" \
      > "$O/$L-wide-x6.txt" 2>&1
  if [[ $L == stock && $BOUND == 1 ]]; then
    echo ">> $(date +%H:%M) bound anchors"
    python3 data/build_bound_anchor.py --base $B --n 400 > "$O/bound-anchor-build.txt" 2>&1
  fi
  if [[ $L == stock && ${PDF4:-0} == 1 ]]; then
    echo ">> $(date +%H:%M) run-4 pdf skill rows"
    $PY data/build_skill_anchor.py --set run4 --extra-attempts 4 --max-len 16384 > "$O/skill-run4-build.txt" 2>&1
  fi
  systemctl --user stop cand-owl
}
cd "$R"
arm stock "$STOCK"
arm "$OWL_LABEL" "$OWL"
for L in stock "$OWL_LABEL"; do echo "== $L"; grep -E "TOTAL|FAIL" "$O/$L-sampler30.txt"; grep "^===" "$O/$L-wide-x6.txt"; done
