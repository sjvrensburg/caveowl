#!/usr/bin/env bash
# Stock on :8110 (servers stay up): thinking-on baseline, think anchors, more bound anchors.
set -uo pipefail
R=~/Projects/caveowl; PY=~/venvs/unsloth-probe/bin/python; cd "$R"
systemctl --user stop cand-owl 2>/dev/null; sleep 3
systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G "$R/scripts/serve_bench.sh" \
    ~/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf 0.6
for _ in $(seq 120); do curl -sf localhost:8110/health >/dev/null && break; sleep 5; done
echo ">> $(date +%H:%M) think_on stock"; python3 -I eval/think_on.py --label stock | tail -1
echo ">> $(date +%H:%M) think anchors"; $PY data/build_think_anchor.py --n 320 | tail -1
echo ">> $(date +%H:%M) bound anchors"; python3 data/build_bound_anchor.py --base http://127.0.0.1:8110/v1 --n 900 | tail -1
systemctl --user stop cand-owl; echo ">> $(date +%H:%M) done"
