#!/usr/bin/env bash
# Memory smoke: take servers down -> train_unsloth.py on a smoke config -> restore on failure.
# Usage: bash scripts/smoke.sh [config]   (default config_smoke.toml). On success servers stay
# DOWN for the full run (scripts/run_full.sh); restore with bash restore_servers.sh otherwise.
set -uo pipefail
D=~/models/ornith-35b-hf; R=~/Projects/caveowl; PY=~/venvs/unsloth-probe/bin/python
# MemoryMax caps CPU-side memory only (GTT is not charged to the cgroup): a load that stages
# the checkpoint in RAM again dies in its own cgroup instead of taking the global OOM killer.
cd "$R"
PAUSE_TIMERS=1 bash takedown_servers.sh || { echo "TAKEDOWN FAILED"; exit 1; }
restore() { echo ">> restoring servers"; bash restore_servers.sh; }
( while true; do cat /sys/class/drm/card1/device/mem_info_gtt_used; sleep 2; done ) > artifacts/smoke_gtt.txt & MON=$!
t0=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=40G $PY train_unsloth.py --config "${1:-config_smoke.toml}" --no-resume > artifacts/smoke.log 2>&1
rc=$?
kill $MON 2>/dev/null
echo ">> smoke rc=$rc wall=$(( $(date +%s)-t0 ))s"
sort -n artifacts/smoke_gtt.txt | tail -1 | awk '{printf ">> peak GTT %.1f GiB (limit 112)\n",$1/2^30}'
tr '\r' '\n' < artifacts/smoke.log | grep -oE "\{'loss'[^}]*\}|train_runtime': '[^']+'|[0-9]+/[0-9]+ \[[^]]*\]|trainable params[^\n]*" | tail -10
grep -E "Traceback|Error|out of memory" artifacts/smoke.log | tail -5
if [[ $rc != 0 ]]; then restore; echo "SMOKE FAILED: servers restored"; exit 1; fi
echo "SMOKE OK: servers remain DOWN for stage B"
