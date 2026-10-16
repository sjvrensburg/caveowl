"""Filter caveman-rewritten tool rows: keep a row only if nothing but its assistant prose changed.

Ported from cavegemma, MIT (https://github.com/JuliusBrussee/cavegemma): the tool-pair path of
its data/filter.py, with the helpers it needs, as a standalone CLI.

Input is data/synthesize_batch.py --tool-corpus output. Each row carries `source_messages` (the
corpus conversation) and `messages` (the same conversation with assistant prose rewritten).
The corpus file (--corpus) is the one the rows were rewritten from; with it, the tool
declarations and source messages are also checked against the original. Checks, first failure
wins, each counted:
  structure_altered        different length or roles
  tools_altered            the `tools` declarations differ from the corpus row
  source_mismatch          source_messages differ from the corpus row (synthesis bug guard)
  tool_call_altered        any assistant tool-call message not byte-identical (JSON, sorted keys)
  tool_result_altered      any tool message not byte-identical
  user_turn_altered        any user turn not byte-identical
  empty_segment            a rewritten prose segment is empty
  code_fence_mutated       a fence in a source segment is not byte-exact in its rewrite (CRLF and
                           common indentation normalised: a block nested under a list item may
                           come back un-indented when the list is compressed)
  identifier_altered       an identifier in the source prose (backticked span, CamelCase,
                           snake_case, dotted name) vanished while a near-identical spelling
                           appeared: the rewriter "fixed" a typo. Dropping it is allowed.
  compression_out_of_band  per-segment ratio tok(rewrite)/tok(source) outside
                           [0.10, --max-compression]. Measured on prose with code fences
                           removed from both sides, because fences must come back byte-exact and
                           cannot shrink; a segment with under PROSE_MIN_TOKENS of prose falls
                           back to the whole-text ratio, since a ratio over a handful of tokens
                           is noise.
  duplicate                key seen before
Rows with no prose (xLAM passthrough) only go through the identity checks. The 0.70 default
ceiling is upstream's own documented fix for its loose 1.00 bound (a model trained on 0.92
rewrites learns that 0.92 is acceptable).

Reproducing the released slice: caveowl's tool rows are this filter at --max-compression 0.80
with tiktoken installed (481 of 600 rewritten rows kept: all 300 xLAM rows, 181 Nemotron). At
the 0.70 default the same input keeps 384 with tiktoken, 432 with the word-count fallback.

Token counting: tiktoken's cl100k_base if installed (what the released slice was measured
with), else a word count. Both sides of a ratio use the same counter, so the fallback moves
ratios only slightly; the run prints which one it used.

Pipeline (from the repo root):
    python data/build_tool_corpus.py                       # -> data/out/tool_corpus.jsonl
    python data/synthesize_batch.py --tool-corpus data/out/tool_corpus.jsonl --fewshot 4 \\
        --confirm-spend                                    # -> data/out/tool_pairs.jsonl
    python data/filter_tool_pairs.py                       # -> data/in/tool.jsonl
Rejected rows, with `_reject_reason`, go to data/out/rejected_tool.jsonl.

The agent loops were NOT filtered for the release. Their calls, results and user turns pass
every identity check, but the prose checks would drop most of them: on the 244 released rows,
0.70 keeps 36 (tiktoken), with 133 over the ratio ceiling, 68 with a code block in the final
answer altered by the rewrite, and 7 with an identifier respelled. To apply it anyway:
    python data/filter_tool_pairs.py --in data/out/ext_traj_cave.jsonl \\
        --corpus data/out/ext_traj.jsonl --out data/out/ext_traj_cave_clean.jsonl \\
        --rejected data/out/rejected_ext.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-.]*\n?(.*?)```", re.DOTALL)
WORD_RE = re.compile(r"\b\w+\b")
IDENT_RE = re.compile(r"`([^`\n]{2,60})`|\b([A-Za-z_][A-Za-z0-9_]*(?:[a-z][A-Z]|_)[A-Za-z0-9_.]*)\b")
PROSE_MIN_TOKENS = 40
TOOL_SEGMENT_MIN_RATIO = 0.10  # upstream's qa band lower bound


def _token_counter():
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(enc.encode(text, disallowed_special=()))), "tiktoken cl100k_base"
    except Exception:
        return (lambda text: len(WORD_RE.findall(text))), "word count (tiktoken not installed)"


tok_count, TOK_NAME = _token_counter()


def strip_fences(text: str) -> str:
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def extract_fences(text: str) -> list[str]:
    return [m.group(1).rstrip("\n") for m in FENCE_RE.finditer(text)]


def _norm_fence(text: str) -> str:
    # CRLF -> LF and common leading indentation removed; nothing else may differ.
    return textwrap.dedent(text.replace("\r\n", "\n").replace("\r", "\n")).strip("\n")


def fences_preserved_whole(source: str, target: str) -> bool:
    src_fences = [f for f in extract_fences(source) if f.strip()]
    if not src_fences:
        return True
    tgt_norm = {_norm_fence(f) for f in extract_fences(target)}
    return all(f in target or _norm_fence(f) in tgt_norm for f in src_fences)


def identifier_altered(source: str, target: str) -> str | None:
    """First source identifier whose exact spelling vanished but a near-match appears."""
    src_ids = {a or b for a, b in IDENT_RE.findall(strip_fences(source))}
    tgt_ids = {a or b for a, b in IDENT_RE.findall(target)}
    for ident in sorted(src_ids):
        if len(ident) < 4 or ident in target:
            continue
        near = difflib.get_close_matches(ident, tgt_ids, n=1, cutoff=0.85)
        if near and near[0] != ident:
            return f"{ident}->{near[0]}"
    return None


def _canon(m: dict) -> str:
    return json.dumps(m, sort_keys=True, ensure_ascii=False)


def _is_prose(m: dict) -> bool:
    return (m.get("role") == "assistant" and not m.get("tool_calls")
            and isinstance(m.get("content"), str) and bool(m["content"].strip()))


def tool_pair_reason(rec: dict, corpus: dict[str, dict], max_compression: float) -> str | None:
    src, tgt = rec.get("source_messages") or [], rec.get("messages") or []
    if not src or len(src) != len(tgt) or [m.get("role") for m in src] != [m.get("role")
                                                                       for m in tgt]:
        return "structure_altered"
    base = corpus.get(rec.get("key"))
    if base is not None:
        if _canon({"t": base["tools"]}) != _canon({"t": rec.get("tools")}):
            return "tools_altered"
        if _canon({"m": base["messages"]}) != _canon({"m": src}):
            return "source_mismatch"
    for s, t in zip(src, tgt):
        if _is_prose(s):
            continue
        if _canon(s) != _canon(t):
            if s["role"] == "tool":
                return "tool_result_altered"
            if s.get("tool_calls") or t.get("tool_calls"):
                return "tool_call_altered"
            if s["role"] == "user":
                return "user_turn_altered"
            return "structure_altered"
    for s, t in zip(src, tgt):
        if not _is_prose(s):
            continue
        sc, tc = s["content"].strip(), (t.get("content") or "").strip()
        if t.get("tool_calls") or not tc:
            return "empty_segment"
        if not fences_preserved_whole(sc, tc):
            return "code_fence_mutated"
        if identifier_altered(sc, tc):
            return "identifier_altered"
        sp = tok_count(strip_fences(sc))
        if sp >= PROSE_MIN_TOKENS:
            ratio = tok_count(strip_fences(tc)) / sp
        else:
            ratio = tok_count(tc) / max(1, tok_count(sc))
        if not (TOOL_SEGMENT_MIN_RATIO <= ratio <= max_compression):
            return f"compression_out_of_band({ratio:.2f})"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", type=Path,
                    default=ROOT / "data" / "out" / "tool_pairs.jsonl")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "out" / "tool_corpus.jsonl",
                    help="the corpus the pairs were rewritten from, for identity checks")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "in" / "tool.jsonl")
    ap.add_argument("--rejected", type=Path,
                    default=ROOT / "data" / "out" / "rejected_tool.jsonl")
    ap.add_argument("--max-compression", type=float, default=0.80,
                    help="upper bound on a segment's tok(rewrite)/tok(source)")
    args = ap.parse_args()

    corpus: dict[str, dict] = {}
    if args.corpus and args.corpus.exists():
        for line in args.corpus.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                corpus[r["key"]] = r
    else:
        print(f"note: no corpus at {args.corpus}; identity checked against "
              "source_messages only")
    counts: Counter[str] = Counter()
    by_cat: Counter[str] = Counter()
    seen: set[str] = set()
    kept, rejected = [], []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            counts["bad_json"] += 1
            continue
        reason = "duplicate" if rec.get("key") in seen else tool_pair_reason(
            rec, corpus, args.max_compression)
        bucket = reason.split("(")[0] if reason else "kept"
        counts[bucket] += 1
        if reason:
            rec["_reject_reason"] = reason
            rejected.append(rec)
        else:
            seen.add(rec["key"])
            kept.append(rec)
            by_cat[rec.get("category", "?")] += 1
    for path, recs in ((args.out, kept), (args.rejected, rejected)):
        # Write beside, then rename over: if the path is a symlink (data/in/ often links to a
        # copy kept elsewhere) the link is replaced and its target is left untouched.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    total = sum(counts.values())
    print(f"tool pairs input: {total}  (max-compression {args.max_compression}, {TOK_NAME})")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v:5d}  ({v / max(1, total) * 100:.1f}%)")
    print("kept by category:", dict(by_cat))
    print(f"kept     -> {args.out.resolve()}")
    print(f"rejected -> {args.rejected.resolve()}")


if __name__ == "__main__":
    main()
