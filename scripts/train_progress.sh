#!/usr/bin/env bash
# Sleep, then summarise the caveowl stage B run.
sleep "$1"
R=~/Projects/caveowl; U="${UNIT:-caveowl-stageB}"; T="${TAG:-}"; A="${ADAPTER:-$R/artifacts/adapter-owl}"
echo "at $(date +%H:%M); unit $(systemctl --user is-active $U)"
journalctl --user -u $U --no-pager -o cat | grep -E "^>>|OK|FAILED" | tail -4
L=$(tr '\r' '\n' < $R/artifacts/train$T.log 2>/dev/null)
echo "$L" | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1
echo "$L" | grep -oE "\{'loss'[^}]*\}" | tail -3 | grep -oE "'loss': '[^']+'|'grad_norm': '[^']+'" | paste -sd' '
echo "$L" | grep -oE "\{'eval_[^}]*\}" | tail -2
echo "$L" | grep -E "Traceback|Error|out of memory" | tail -3
awk '{printf "GTT now %.1f GiB\n",$1/2^30}' /sys/class/drm/card1/device/mem_info_gtt_used
C=$(ls -d $A/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$C" ] && jq -r '.log_history[] | select(.loss or .eval_loss) | "  step \(.step): \(if .loss then "loss \(.loss*1000|round/1000) gn \(.grad_norm*100|round/100)" else "EVAL \(to_entries|map(select(.key|startswith("eval_") and endswith("loss")))|map("\(.key) \(.value*1000|round/1000)")|join(" "))" end)"' $C/trainer_state.json | tail -4
exit 0
