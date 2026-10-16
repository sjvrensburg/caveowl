"""Held-out Rust / shell / R code check: per-language pass counts on tasks no anchor trained on.

The other code gate (../llm-bench-kit/code-eval.py) is Python only, and so were the anchors
until 2026-10-12. This scores what the UltraData-Code L3 anchors are there to protect:
data/merge_anchors.py holds out 30 reference-validated tasks per language
(data/out/code_heldout.jsonl), picked from ALL validated tasks, so stock is not at 30/30 by
construction. Same prompt, sampler and sandboxed test run as data/build_anchor_ultra.py.

Usage (model on the bench port; owl_ports.refuse_live guards live servers):
    python3 -I eval/code_heldout.py --label stock
Pass rule: per language, candidate >= stock - 2 (temp 0, so reruns of one model agree; the
slack is for single-task flips), and total >= stock.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))
from owl_ports import BENCH_URL, refuse_live  # noqa: E402
from build_anchor_ultra import FENCE, ask, prompt_for, run_tests  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BENCH_URL)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", type=Path, default=ROOT / "data" / "out" / "code_heldout.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    refuse_live(args.base)

    tasks = [json.loads(l) for l in args.tasks.open()]

    def one(t: dict) -> dict:
        answer, finish = ask(args.base, prompt_for(t["lang"], t))
        m = re.search(rf"```(?:{FENCE[t['lang']]})?\s*\n(.*?)```", answer, re.S)
        code = m.group(1) if m else answer
        ok = finish != "length" and bool(code.strip()) and run_tests(t["lang"], code, t["test"])
        return {"lang": t["lang"], "uuid": t["uuid"], "pass": ok, "finish": finish, "answer": answer}

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, tasks))
    out = ROOT / "eval" / "out" / f"code_heldout-{args.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in res))
    n, ok = Counter(r["lang"] for r in res), Counter(r["lang"] for r in res if r["pass"])
    per = "  ".join(f"{l} {ok[l]}/{n[l]}" for l in ("sh", "r", "rust"))
    print(f"=== {args.label}: {sum(ok.values())}/{len(res)}  {per}  -> {out}")


if __name__ == "__main__":
    main()
