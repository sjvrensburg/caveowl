"""Merge the Python (MBPP) and Rust/sh/R (UltraData-Code L3) anchors into one anchor file, and
carve out the held-out per-language code check the gate runs (eval/code_heldout.py).

Operator's call (2026-10-12): same anchor_share (0.20), split roughly half Python, half
sh/R/Rust. train_unsloth.py's build_tool_splits orders anchor rows by key hash and keeps the
first `want` (= share/(1-share) x text rows), so the split is set here by row COUNTS: equal
numbers of Python and non-Python rows go in, and any cap then truncates both halves alike in
expectation. The trainer prints the resulting train count; check it says anchor ~20%.

Held-out: HOLDOUT tasks per language, picked by key hash from ALL reference-validated tasks
(data/out/ultra_tasks.jsonl), not just the ones stock Ornith passed, so the check has room to
move both ways. Their anchor rows (if any) are excluded from training.

Run (after build_anchor_ultra.py):
    python3 -I data/merge_anchors.py
"""
from __future__ import annotations

import json
from collections import Counter
from hashlib import sha1
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
HOLDOUT = 30


def h(s: str) -> str:
    return sha1(s.encode()).hexdigest()


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()]


def main() -> None:
    tasks = read(OUT / "ultra_tasks.jsonl")
    held = []
    for lang in ("sh", "r", "rust"):
        mine = sorted((t for t in tasks if t["lang"] == lang), key=lambda t: h("hold:" + t["uuid"]))
        held += mine[:HOLDOUT]
    held_keys = {f"ultracode-l3:{t['lang']}:{t['uuid']}" for t in held}
    with (OUT / "code_heldout.jsonl").open("w") as f:
        for t in held:
            f.write(json.dumps(t) + "\n")

    py = read(OUT / "anchor.jsonl")
    ultra = [r for r in read(OUT / "anchor_ultra.jsonl") if r["key"] not in held_keys]
    n = min(len(py), len(ultra))
    pick = lambda rows: sorted(rows, key=lambda r: h("mix:" + r["key"]))[:n]
    rows = pick(py) + pick(ultra)
    with (OUT / "anchor_all.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    langs = Counter("py" if r["key"].startswith("mbpp:") else r["key"].split(":")[1] for r in rows)
    print(f"held out {len(held)} tasks {dict(Counter(t['lang'] for t in held))} -> code_heldout.jsonl")
    print(f"anchor_all: {len(rows)} rows {dict(langs)} (python available {len(py)}, "
          f"ultra available {len(ultra)})")


if __name__ == "__main__":
    main()
