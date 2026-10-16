"""Build self-distilled code ANCHOR rows: MBPP tasks answered by the stock model, kept only if
the answer passes the task's own tests.

caveowl copy of ../cavegemma-e4b/data/build_anchor.py (9dd4267), regenerated against STOCK
Ornith-1.5-35B-A3B: an anchor is only an anchor if it is the model's OWN answer, so E4B's rows
cannot be reused. Changes: model id, two parallel requests (the bench unit runs --parallel 2),
and answers cut off by max_tokens are rejected (finish_reason "length"; training on them would
put an end-of-turn token after a truncated answer). Prompt, filter and gate-name exclusion are
unchanged. Provenance below is the E4B port's.

Why (v2, 2026-10-11): the v1 LoRA (caveman pairs + tool slice, 3 epochs, lr 2e-4) kept tool
calls at 9/9 but took ../llm-bench-kit/code-eval.py from 10/12 to 2/12 (epoch 2) / 4/12
(epoch 3). The terse style leaked from prose into code: it mutated an input the prompt said
not to touch and ignored "Code only." Nothing in the training data rewarded meeting a code
spec. Anchors are the standard fix for that kind of forgetting: the stock model's own answers
(its own style, so they pull toward "stay as you were", not toward a third style), filtered
to the ones that are verifiably correct, mixed into training at a fixed share.

Source: google-research-datasets/mbpp, config `sanitized` (CC-BY-4.0), all four splits.
Tasks whose text names one of code-eval.py's function names are dropped, so the gate is not
trained on. The prompt is MBPP's own convention (task + its first test as the signature hint)
plus "Code only.", matching code-eval.py's phrasing. All of a task's tests are shown (see
the comment in main()).

Needs STOCK Ornith served on --base (default http://127.0.0.1:8110/v1, the bench unit cand-owl;
:8088 serves the same file but is Open Notebook's worker: do not load it with this).

Run:
    ~/venvs/unsloth-probe/bin/python data/build_anchor.py --out data/out/anchor.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ("https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
           "refs%2Fconvert%2Fparquet/sanitized/{split}/0000.parquet")
SPLITS = ("train", "validation", "prompt", "test")
# code-eval.py's tasks: never train on anything that names them.
GATE_NAMES = ("chunk_ranges", "merge_interval", "parse_kv", "running_median", "retry",
              "dedupe", "backoff")
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def load_tasks() -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for split in SPLITS:
        data = urllib.request.urlopen(PARQUET.format(split=split), timeout=60).read()
        for r in pq.read_table(io.BytesIO(data)).to_pylist():
            r["split"] = split
            rows.append(r)
    return rows


def ask(base: str, prompt: str) -> tuple[str, str]:
    body = {"model": "ornith-1.5-35b-a3b", "temperature": 0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ch = json.load(r)["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason") or ""


def passes(code: str, task: dict) -> bool:
    prog = "\n".join([task.get("test_setup_code") or "", code, *task["test_list"]])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=10)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(f.name).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8110/v1")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "out" / "anchor.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="the bench unit runs --parallel 2")
    args = ap.parse_args()

    tasks = load_tasks()
    tasks = [t for t in tasks if not any(n in t["prompt"].lower() for n in GATE_NAMES)]
    if args.limit:
        tasks = tasks[:args.limit]
    stats = {"kept": 0, "failed": 0, "nocode": 0, "truncated": 0, "done": 0}
    rows: dict[int, dict] = {}
    lock = threading.Lock()

    def one(it: tuple[int, dict]) -> None:
        i, t = it
        # All of the task's tests, not just the first: MBPP task text often omits a
        # detail its tests pin (e.g. return None when nothing repeats). Measured on 10
        # tasks (E4B port): first test only kept 5/10, all failures genuine spec gaps.
        tests = "\n".join(t["test_list"])
        prompt = (f"{t['prompt'].strip()} Your code should pass these tests:\n"
                  f"{tests}\nCode only.")
        answer, finish = ask(args.base, prompt)
        m = FENCE.search(answer)
        code = m.group(1) if m else answer
        if finish == "length":
            res = "truncated"
        elif not code.strip():
            res = "nocode"
        elif not passes(code, t):
            res = "failed"
        else:
            res = "kept"
        with lock:
            stats[res] += 1
            stats["done"] += 1
            if res == "kept":
                rows[i] = {
                    "key": f"mbpp:{t['task_id']}",
                    "origin": f"mbpp-sanitized:{t['split']}:{t['task_id']}",
                    "license": "cc-by-4.0", "category": "anchor",
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": answer.strip()}]}
            if stats["done"] % 50 == 0:
                print(f"  {stats['done']}/{len(tasks)}  {stats}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, enumerate(tasks)))
    with args.out.open("w") as out:
        for i in sorted(rows):  # task order, not completion order: reruns diff cleanly
            out.write(json.dumps(rows[i]) + "\n")
    print(f"tasks {len(tasks)}  kept {stats['kept']}  failed tests {stats['failed']}  "
          f"no code {stats['nocode']}  truncated {stats['truncated']}  -> {args.out}")


if __name__ == "__main__":
    main()
