#!/usr/bin/env bash
# Stage B: full caveowl run -> merge + GGUF -> ALWAYS restore servers + timers.
# Run detached:  systemd-run --user --unit=caveowl-stageB bash ~/Projects/caveowl/scripts/run_full.sh
# Resumes from the newest artifacts/adapter-owl/checkpoint-* (save_steps 25) after an interruption.
# Servers must already be down (stage A / owl_smoke.sh leaves them down after a good smoke).
# Run 2+: CONFIG / ADAPTER / TAG pick the config, its output_dir and the artifact names, so a new
# run never overwrites an earlier one's adapter, merge or GGUFs. GATE=1 runs scripts/gate.sh on
# the new GGUF after the servers are back (needs PATH and PI_MODELS_JSON, see gate.sh).
set -uo pipefail
R=~/Projects/caveowl; PY=~/venvs/unsloth-probe/bin/python
CONFIG="${CONFIG:-config.toml}"; ADAPTER="${ADAPTER:-artifacts/adapter}"; TAG="${TAG:-}"
GATE="${GATE:-0}"
cd "$R"
restore() { echo ">> $(date +%H:%M) restoring servers"; bash restore_servers.sh; }
finish() {
  local rc=$?
  restore
  if [[ $rc == 0 && $GATE == 1 ]]; then
    echo ">> $(date +%H:%M) gate"
    OWL="$R/artifacts/gguf/Ornith-1.5-35B-caveowl$TAG-Q4_K_M.gguf" OWL_LABEL="caveowl$TAG" \
        bash scripts/gate.sh > "artifacts/gate$TAG.log" 2>&1
    echo ">> $(date +%H:%M) gate rc=$?"; tail -40 "artifacts/gate$TAG.log"
  fi
}
trap finish EXIT
( while true; do cat /sys/class/drm/card1/device/mem_info_gtt_used; sleep 10; done ) > artifacts/train${TAG}_gtt.txt & MON=$!
echo ">> $(date +%H:%M) training"
# MemoryMax caps CPU-side memory only (GTT is not charged): a regression to staging the
# checkpoint in RAM dies in this scope instead of taking the global OOM killer.
systemd-run --user --scope -q -p MemoryMax=40G $PY train_unsloth.py --config "$CONFIG" \
    > "artifacts/train$TAG.log" 2>&1
rc=$?; kill $MON 2>/dev/null
echo ">> $(date +%H:%M) train rc=$rc"
sort -n artifacts/train${TAG}_gtt.txt | tail -1 | awk '{printf ">> peak GTT %.1f GiB\n",$1/2^30}'
[[ $rc == 0 && -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "TRAIN FAILED"; exit 1; }
echo ">> $(date +%H:%M) merge + GGUF"
ADAPTER="$ADAPTER" MERGED="artifacts/merged-owl$TAG" NAME="Ornith-1.5-35B-caveowl$TAG" \
    bash merge_and_gguf.sh > "artifacts/merge$TAG.log" 2>&1
rc=$?
echo ">> $(date +%H:%M) merge rc=$rc"; tail -5 "artifacts/merge$TAG.log"
[[ $rc == 0 ]] || { echo "MERGE FAILED"; exit 1; }
echo "STAGE B OK"
