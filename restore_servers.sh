#!/usr/bin/env bash
# caveowl: bring back exactly the model servers takedown_servers.sh stopped, and prove each one
# answers. Run by the OPERATOR's session:   bash restore_servers.sh
#
# Reads ~/.local/state/caveowl/takedown.units ("service <unit> <port>" / "timer <unit> -"),
# starts the services in the recorded order, polls http://127.0.0.1:<port>/health until it
# returns 200 (a 35B with --load-mode dio takes up to a minute or two; TIMEOUT per unit,
# default 300 s), then re-starts any paused timers. The record is kept as takedown.units.last
# only if every unit came back healthy; otherwise it stays, so a re-run retries.
#
# Before restoring, it refuses if a training or merge process still holds the GPU memory the
# servers need (CHECK_GPU=0 to skip): starting :8088's 21 GiB into a full GTT fails slowly.
set -euo pipefail

STATE="${STATE_DIR:-$HOME/.local/state/caveowl}/takedown.units"
TIMEOUT="${TIMEOUT:-300}"

[[ -s "$STATE" ]] || { echo "no takedown record at $STATE: nothing to restore" >&2; exit 1; }

if [[ "${CHECK_GPU:-1}" == 1 ]] && pgrep -f 'train_unsloth\.py|merge_owl\.py|llama-imatrix|convert_hf_to_gguf\.py|llama-quantize' >/dev/null; then
    echo "a caveowl training / merge / imatrix process is still running:" >&2
    pgrep -af 'train_unsloth\.py|merge_owl\.py|llama-imatrix|convert_hf_to_gguf\.py|llama-quantize' >&2
    echo "let it finish (or stop it), then re-run. CHECK_GPU=0 to override." >&2
    exit 1
fi

fail=0
while read -r -u 3 kind unit port; do
    [[ "$kind" == service ]] || continue
    echo ">> starting $unit"
    systemctl --user start "$unit" || { echo "!! $unit failed to start" >&2; fail=1; continue; }
done 3< "$STATE"

while read -r -u 3 kind unit port; do
    [[ "$kind" == service ]] || continue
    if [[ "$port" == - ]]; then
        if systemctl --user is-active --quiet "$unit"; then
            echo "   $unit active (no --port in its unit file; /health not checked)"
        else
            echo "!! $unit not active" >&2; fail=1
        fi
        continue
    fi
    t0=$SECONDS
    until curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$port/health"; do
        if ! systemctl --user is-active --quiet "$unit"; then
            echo "!! $unit is not active (journalctl --user -u $unit -n 50)" >&2; fail=1; continue 2
        fi
        if (( SECONDS - t0 > TIMEOUT )); then
            echo "!! $unit :$port no /health after ${TIMEOUT}s" >&2; fail=1; continue 2
        fi
        sleep 3
    done
    echo "   $unit :$port healthy after $((SECONDS - t0)) s"
done 3< "$STATE"

while read -r -u 3 kind unit port; do
    [[ "$kind" == timer ]] || continue
    echo ">> resuming $unit"
    systemctl --user start "$unit" || { echo "!! $unit failed" >&2; fail=1; }
done 3< "$STATE"

for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
    [[ -r "$f" ]] && awk '{printf ">> GTT used now: %.1f GiB\n", $1/2^30}' "$f" && break
done

if (( fail )); then
    echo "!! not everything came back; record kept at $STATE so a re-run retries" >&2
    exit 1
fi
mv "$STATE" "$STATE.last"
echo ">> all recorded units restored and healthy (record moved to $STATE.last)"
