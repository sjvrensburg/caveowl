#!/usr/bin/env bash
# caveowl: merge the LoRA into Ornith-1.5-35B-A3B and produce a Q4_K_M GGUF for llama-server.
# Run: bash merge_and_gguf.sh       (from anywhere; NOT run by the setup; servers must be down:
#                                    the merge holds the ~67 GiB bf16 base in memory)
#
# Pipeline: adapter + bf16 base -> merged HF checkpoint (merge_owl.py, with its tensor checks)
#           -> BF16 GGUF (~/src/llama.cpp/convert_hf_to_gguf.py)
#           -> importance matrix (llama-imatrix, on the BF16 GGUF)
#           -> Q4_K_M GGUF (llama-quantize), all under artifacts/.
#
# Why these choices:
#   - convert_hf_to_gguf.py from ~/src/llama.cpp (b10969, 391fac164): conversion/qwen.py
#     registers Qwen3_5MoeForConditionalGeneration -> MODEL_ARCH.QWEN35MOE ("qwen35moe"),
#     reorders the DeltaNet V heads, and exports the MTP head as the nextn layer (blk.40),
#     which is why merge_owl.py restores mtp.* in the merged checkpoint. The vision tower is
#     skipped by the text conversion. Run with ~/venvs/unsloth-probe (transformers 5.17, torch).
#   - Q4_K_M = the stock file's type: ~/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf reports
#     general.file_type 15 (MOSTLY_Q4_K_M), tensors Q4_K x379 / Q6_K x64 / F32 x310.
#   - The stock file was quantized WITH an importance matrix (its metadata names the vendor's
#     imatrix, 3636 chunks of their own calibration set, which is not published). A plain
#     Q4_K_M would lose to stock on quantisation alone and confound the gate, so this computes
#     an imatrix too. It cannot be the vendor's, so the comparison is still not perfectly
#     like-for-like: set CONTROL=1 to also convert + quantize the UNMODIFIED base with the SAME
#     imatrix (stock-requant), the exact control arm, if a gate result is close.
#   - Calibration text: wikitext-2 TRAIN split (wiki.train.raw). Not the test split, which is
#     what the perplexity check below reads.
#   - llama-imatrix / llama-quantize from ~/src/llama.cpp/build-vulkan, the stock tree this box
#     trusts as its control arm (the ROCm trees there computed garbage on gfx1151).
#     Quantisation is CPU work; the backend only matters for imatrix, which runs the model.
#   - mmproj: NOT rebuilt. LoRA never touched the vision tower (merge_owl.py checks it is
#     byte-identical), so the stock mmproj-Ornith-1.5-35B-A3B-f16.gguf still matches.
#
# Nothing here installs into ~/models or touches a systemd unit.
set -euo pipefail

cd "$(dirname "$0")"

BASE="${BASE:-$HOME/models/ornith-35b-hf}"
ADAPTER="${ADAPTER:-artifacts/adapter}"
MERGED="${MERGED:-artifacts/merged}"
OUTDIR="${OUTDIR:-artifacts/gguf}"
NAME="${NAME:-Ornith-1.5-35B-caveowl}"
METHOD="${METHOD:-unsloth}"            # merge_owl.py --method (peft = fallback)
LLAMA="${LLAMA:-$HOME/src/llama.cpp}"
CALIB="${CALIB:-$LLAMA/wikitext-2-raw/wiki.train.raw}"
CHUNKS="${CHUNKS:-200}"                # 200 x 512 tokens of calibration
CONTROL="${CONTROL:-0}"
PY="${PY:-$HOME/venvs/unsloth-probe/bin/python}"
BIN="$LLAMA/build-vulkan/bin"
STOCK="$HOME/models/ornith-35b/Ornith-1.5-35B-Q4_K_M.gguf"
export HF_HUB_DISABLE_XET=1

for f in "$PY" "$BIN/llama-quantize" "$BIN/llama-imatrix" "$LLAMA/convert_hf_to_gguf.py" "$CALIB"; do
    [[ -e "$f" ]] || { echo "missing $f" >&2; exit 1; }
done
[[ -f "$BASE/model.safetensors.index.json" ]] || { echo "no bf16 weights at $BASE yet" >&2; exit 1; }
[[ -f "$ADAPTER/adapter_config.json" ]] || { echo "no adapter at $ADAPTER: train first" >&2; exit 1; }
if systemctl --user is-active --quiet llama-ornith llama-ornith35 2>/dev/null; then
    echo "resident model servers are up; run takedown_servers.sh first (the merge needs ~70 GiB)" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
BF16="$OUTDIR/$NAME-BF16.gguf"
IMAT="$OUTDIR/$NAME.imatrix.gguf"
Q4="$OUTDIR/$NAME-Q4_K_M.gguf"

echo ">> 1/4 merge adapter -> $MERGED ($METHOD)"
# Reuse an existing merge ONLY if it was made from this exact adapter (code review 2026-10-12:
# after a retrain the old merge was silently converted). The stamp records the adapter's
# weights hash; anything else (or METHOD changed) means remove and re-merge.
STAMP="$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -c1-16)-$METHOD"
if [[ -d "$MERGED" && "$(cat "$MERGED/.adapter-stamp" 2>/dev/null)" != "$STAMP" ]]; then
    echo "   $MERGED was made from a different adapter or method: removing it and re-merging"
    rm -rf "$MERGED"
fi
if [[ -d "$MERGED" ]]; then
    echo "   $MERGED exists for this adapter ($STAMP): re-running the checks only"
    "$PY" merge_owl.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" --check-only
else
    "$PY" merge_owl.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" --method "$METHOD"
    echo "$STAMP" > "$MERGED/.adapter-stamp"
fi

echo ">> 2/4 convert -> $BF16"
"$PY" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outtype bf16 --outfile "$BF16"

echo ">> 3/4 importance matrix -> $IMAT ($CHUNKS chunks of $(basename "$CALIB"))"
"$BIN/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IMAT" --chunks "$CHUNKS" -c 512 -ngl 999 -fa on

echo ">> 4/4 quantize -> $Q4"
"$BIN/llama-quantize" --imatrix "$IMAT" "$BF16" "$Q4" Q4_K_M

if [[ "$CONTROL" == 1 ]]; then
    echo ">> control: unmodified base, same converter, same imatrix"
    CB="$OUTDIR/Ornith-1.5-35B-stock-BF16.gguf"
    CQ="$OUTDIR/Ornith-1.5-35B-stock-requant-Q4_K_M.gguf"
    "$PY" "$LLAMA/convert_hf_to_gguf.py" "$BASE" --outtype bf16 --outfile "$CB"
    "$BIN/llama-quantize" --imatrix "$IMAT" "$CB" "$CQ" Q4_K_M
    ls -la "$CQ"
fi

ls -la "$BF16" "$IMAT" "$Q4"

cat <<NEXT

Verify:
  - $Q4 is about the size of $STOCK (~20.2 GiB) and has the same tensor-type mix and block
    count (41, the nextn layer included):
      PYTHONPATH=$LLAMA/gguf-py $PY -c 'from gguf import GGUFReader as R; from collections import Counter as C
      for p in ("$Q4", "$STOCK"): r = R(p); print(p, r.fields["qwen35moe.block_count"].contents(), C(t.tensor_type.name for t in r.tensors))'
  - the embedded chat template equals the stock GGUF's (it must: training rendered with it):
      PYTHONPATH=$LLAMA/gguf-py $PY -c 'from gguf import GGUFReader as R; t=lambda p: R(p).fields["tokenizer.chat_template"].contents(); print(t("$Q4")==t("$STOCK"))'
  - perplexity is sane against stock (same binary, same text, same flags):
      $HOME/src/llama.cpp-strix-076/build-vulkan/bin/llama-perplexity -m $Q4 -f $LLAMA/wikitext-2-raw/wiki.test.raw -ngl 999 -fa on
      (and the same for $STOCK)
  - serve it on the bench port with stock's flags and the stock mmproj, then run the gate:
      systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G scripts/serve_bench.sh $Q4

Next: README "The gate". Nothing here is deployed.
NEXT
