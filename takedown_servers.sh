#!/usr/bin/env bash
# caveowl: stop every resident model server so the bf16 LoRA (~80-85 GiB) has the GTT, and
# record exactly which were running so restore_servers.sh brings back those and no others.
# Run by the OPERATOR's session, not by an agent:   bash takedown_servers.sh
#
# What it stops: every ACTIVE user service matching llama-* (2026-10-12 that is llama-ornith
# :8092 Gemma-4-E4B caveman v3, llama-ornith35 :8088, llama-winnow :8098, llama-embedding :8086;
# any on-demand llama-* someone started is caught the same way) and transient bench units
# cand-* (not recorded: a stopped transient unit no longer exists to restart).
#
# What it does NOT stop, and why:
#   - vibevoice-api: consumers lease it through halo-prep scripts/vibevoice-lease.sh, and a bare
#     stop cuts a running voice message mid-clip (halo-prep CLAUDE.md). If it is up, this script
#     refuses; wait for the lease holder to finish (it stops itself) and re-run.
#   - pi-signal, pi-web: hold no GTT. While :8092 is down their requests fail; the `judge`
#     extension fails open when :8098 is down. Nothing to restart for them afterwards.
#   - pi-job timers: by default left alone (a job firing during training fails and
#     notify-failure@ sends one Signal message). PAUSE_TIMERS=1 stops the active pi-job-*.timer
#     and memory-consolidate.timer units too and records them for restore.
#
# Every llama-* unit has Restart=always; that does not fire on an explicit `systemctl stop`.
set -euo pipefail

STATE_DIR="${STATE_DIR:-$HOME/.local/state/caveowl}"
STATE="$STATE_DIR/takedown.units"
mkdir -p "$STATE_DIR"

gtt() {
    local f
    for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
        [[ -r "$f" ]] && awk '{printf "%.1f", $1/2^30}' "$f" && return
    done
    echo "?"
}

if [[ -s "$STATE" && "${FORCE:-0}" != 1 ]]; then
    echo "$STATE already lists units taken down at $(stat -c %y "$STATE"):" >&2
    cat "$STATE" >&2
    echo "Run restore_servers.sh first (or FORCE=1 to overwrite the record)." >&2
    exit 1
fi

if systemctl --user is-active --quiet vibevoice-api; then
    echo "vibevoice-api is running: it is lease-managed (vibevoice-lease.sh) and may be mid-clip." >&2
    echo "Wait for its consumer to release it, then re-run. Not stopping it." >&2
    exit 1
fi

echo ">> GTT used before: $(gtt) GiB"
mapfile -t UNITS < <(systemctl --user list-units --no-legend --plain --state=active \
    --type=service 'llama-*' | awk '{print $1}')
mapfile -t CANDS < <(systemctl --user list-units --no-legend --plain --state=active \
    --type=service 'cand-*' | awk '{print $1}')
TIMERS=()
if [[ "${PAUSE_TIMERS:-0}" == 1 ]]; then
    mapfile -t TIMERS < <(systemctl --user list-units --no-legend --plain --state=active \
        --type=timer 'pi-job-*' 'memory-consolidate*' | awk '{print $1}')
fi

: > "$STATE.tmp"
for u in "${UNITS[@]}"; do
    # The port restore_servers.sh polls for /health, read from the unit's own ExecStart.
    port="$(systemctl --user cat "$u" | grep -oE -- '--port[ =]+[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    printf 'service %s %s\n' "$u" "${port:--}" >> "$STATE.tmp"
done
for t in "${TIMERS[@]}"; do
    printf 'timer %s -\n' "$t" >> "$STATE.tmp"
done
mv "$STATE.tmp" "$STATE"
echo ">> recorded in $STATE:"
sed 's/^/     /' "$STATE"

for u in "${UNITS[@]}" "${CANDS[@]}"; do
    echo ">> stopping $u"
    systemctl --user stop "$u"
done
for t in "${TIMERS[@]}"; do
    echo ">> pausing $t"
    systemctl --user stop "$t"
done

sleep 3
echo ">> GTT used after: $(gtt) GiB"
left="$(systemctl --user list-units --no-legend --plain --state=active --type=service 'llama-*' 'cand-*' | awk '{print $1}')"
if [[ -n "$left" ]]; then
    echo "!! still active: $left" >&2
    exit 1
fi
# Anything else holding the GPU (a ComfyUI, a stray llama-bench, a notebook):
users="$(fuser /dev/dri/renderD* 2>/dev/null | tr -s ' ' '\n' | grep -oE '^[0-9]+' | sort -un || true)"
if [[ -n "$users" ]]; then
    echo ">> processes with the GPU open (not stopped by this script; check their size):"
    # shellcheck disable=SC2086
    ps -o pid=,rss=,comm= -p $(echo $users | tr ' ' ',') | awk '{printf "     %s  %.1f GiB RSS  %s\n", $1, $2/2^20, $3}'
fi

cat <<NEXT

Down: ${#UNITS[@]} model server(s)$( [[ ${#TIMERS[@]} -gt 0 ]] && echo ", ${#TIMERS[@]} timer(s) paused").
GTT should now be a few GiB at most (the desktop). Then:
  ~/venvs/unsloth-probe/bin/python train_unsloth.py --config config_owl_smoke.toml   # first time
  ~/venvs/unsloth-probe/bin/python train_unsloth.py --config config_owl.toml
Afterwards (also after a failed run): bash restore_servers.sh
NEXT
