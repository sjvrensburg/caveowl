#!/usr/bin/env bash
# Serve a GGUF on the bench port (BENCH_PORT, default 8110) for anchor generation (stock) and the
# gate (stock vs candidate, same flags). Use a port nothing else relies on.
#
# Environment (defaults are the author's box):
#   BENCH_PORT    8110
#   LLAMA_SERVER  llama-server binary  ($HOME/src/llama.cpp-strix-076/build-vulkan/bin/llama-server)
#   MMPROJ        the base model's vision projector; empty = text only
#                 ($HOME/models/ornith-35b/mmproj-Ornith-1.5-35B-A3B-f16.gguf)
#
#   systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G \
#       ~/Projects/caveowl/scripts/serve_bench.sh [GGUF] [TEMP]
#   ... ; systemctl --user stop cand-owl
#
# Flags are the author's production Ornith server's (2026-10-12) except:
#   --port $BENCH_PORT     bench port (the skill builder's logging proxy uses BENCH_PORT+1)
#   --parallel 2           two pi workers / two MBPP requests at once
#   --ctx-size 262144      = 2 x 131072, so each slot gets :8088's window. --parallel set
#                          explicitly turns kv_unified OFF (halo-prep CLAUDE.md, :8088 note), so
#                          the two slots are independent 131072-token windows, which is what
#                          pi's contextWindow (131072) for this model assumes.
# Use the same binary and flags as whatever will serve the model for real, so stock behaviour
# here is production behaviour. --load-mode dio is load-bearing (large models hang on
# load without it). ~21 GiB of GTT while it runs.
set -euo pipefail
GGUF="${1:-$HOME/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf}"
TEMP="${2:-0.6}"
BENCH_PORT="${BENCH_PORT:-8110}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/src/llama.cpp-strix-076/build-vulkan/bin/llama-server}"
MMPROJ="${MMPROJ-$HOME/models/ornith-35b/mmproj-Ornith-1.5-35B-A3B-f16.gguf}"
MM=(); [[ -n "$MMPROJ" ]] && MM=(--mmproj "$MMPROJ" --image-min-tokens 1024)
exec env ROCBLAS_USE_HIPBLASLT=1 "$LLAMA_SERVER" \
  --model "$GGUF" \
  --n-gpu-layers 999 \
  -fit off \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --load-mode dio \
  --ctx-size 262144 \
  --parallel 2 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-ram 8192 \
  --jinja \
  "${MM[@]}" \
  --reasoning-format deepseek \
  --temp "$TEMP" \
  --top-k 20 \
  --top-p 0.95 \
  --host 127.0.0.1 \
  --port "$BENCH_PORT" \
  --alias ornith-1.5-35b-a3b
