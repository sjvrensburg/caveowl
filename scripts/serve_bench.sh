#!/usr/bin/env bash
# Serve a GGUF on the bench port (:8110) with production :8088's flags, for anchor generation
# (stock) and the gate (stock vs candidate, same flags). Never :8092 / :8088.
#
#   systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G \
#       ~/Projects/caveowl/scripts/serve_bench.sh [GGUF] [TEMP]
#   ... ; systemctl --user stop cand-owl
#
# Flags are copied from ~/.config/systemd/user/llama-ornith35.service (2026-10-12) except:
#   --port 8110            bench port (cavegemma's convention; the logging proxy uses 8111)
#   --parallel 2           two pi workers / two MBPP requests at once
#   --ctx-size 262144      = 2 x 131072, so each slot gets :8088's window. --parallel set
#                          explicitly turns kv_unified OFF (halo-prep CLAUDE.md, :8088 note), so
#                          the two slots are independent 131072-token windows, which is what
#                          pi's contextWindow (131072) for this model assumes.
# Same binary as :8088 (strix fork v0.7.6.1, halo-prep CLAUDE.md "Binary path"), so stock
# behaviour here is :8088's behaviour. --load-mode dio is load-bearing (large models hang on
# load without it). ~21 GiB of GTT while it runs.
set -euo pipefail
GGUF="${1:-$HOME/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf}"
TEMP="${2:-0.6}"
exec env ROCBLAS_USE_HIPBLASLT=1 "$HOME/src/llama.cpp-strix-076/build-vulkan/bin/llama-server" \
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
  --mmproj "$HOME/models/ornith-35b/mmproj-Ornith-1.5-35B-A3B-f16.gguf" \
  --image-min-tokens 1024 \
  --reasoning-format deepseek \
  --temp "$TEMP" \
  --top-k 20 \
  --top-p 0.95 \
  --host 127.0.0.1 \
  --port 8110 \
  --alias ornith-1.5-35b-a3b
