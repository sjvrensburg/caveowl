"""Build self-distilled Rust / shell / R code ANCHOR rows from openbmb/UltraData-Code L3.

Same idea as build_anchor.py (stock model's own answers, kept only if they pass tests), extended
past Python: MBPP is Python-only, and pi's day-to-day code is mostly shell, with R and Rust
next. Source: openbmb/UltraData-Code, config UltraData-Code-L3 (Apache-2.0 on the dataset; the
card also asks users to respect the source repos' licences, and L3 rows are model-generated
exercises derived from those files). Each row has task / analysis / solution / test fields.

Why two filters, not one (measured 2026-10-12 on 20 rows per language, offset 0): the dataset's
OWN reference solution passed its OWN generated tests in only 5/20 Rust, 6/20 sh, 5/20 R. The
failures were genuine defects (wrong expected values, solutions that disagree with their tests,
Rust that does not compile: stray `##`, unbalanced braces), plus a few environmental ones
(missing R packages, external crates). So the card's "test candidates" are candidates:
  1. --validate keeps a task only if the reference solution passes its tests here. That makes
     the tests trustworthy enough to grade with, and drops tasks needing absent deps.
  2. generation asks STOCK Ornith (bench unit :8110, never :8088/:8092) for the task with its
     tests shown, and keeps the answer only if it passes those tests.
The reference solution itself is never a training target: it is a third model's style.

All dataset code (reference and model answers) runs under bwrap: no network, its own work dir,
read-only system binds, a timeout.

Run:
    python3 -I data/build_anchor_ultra.py --validate --per-lang 600     # no server needed
    systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G scripts/serve_bench.sh
    python3 -I data/build_anchor_ultra.py --out data/out/anchor_ultra.jsonl
Generation appends one row per kept answer (stage A's takedown stops cand-* units; whatever
was kept by then survives) and skips keys already in --out on a rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=openbmb/UltraData-Code"
            "&config=UltraData-Code-L3&split={split}&offset={off}&length=100")
TOTAL = {"rust": 922564, "sh": 365222, "r": 304016}  # datasets-server /size, 2026-10-12
FENCE = {"rust": r"rust|rs", "sh": r"bash|sh|shell", "r": r"r|R"}
NAME = {"rust": "Rust", "sh": "Bash", "r": "R"}
# Tasks that need what the sandbox cannot (or should not) give them.
SKIP = {
    "rust": re.compile(r"^\s*(extern crate|use (?!std|core|alloc|crate|self|super)\w+)", re.M),
    "sh": re.compile(r"\b(curl|wget|ssh|scp|sudo|apt(-get)?|dnf|yum|docker|systemctl|nc)\b"),
    "r": re.compile(r"\b(install\.packages|download\.file|url\()"),
}
HOME = os.path.expanduser("~")
BWRAP = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
         "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/lib", "/lib",
         "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev",
         "--ro-bind", f"{HOME}/.rustup", f"{HOME}/.rustup",
         "--ro-bind", f"{HOME}/.cargo/bin", f"{HOME}/.cargo/bin",
         "--setenv", "RUSTUP_HOME", f"{HOME}/.rustup",
         "--setenv", "PATH", f"{HOME}/.cargo/bin:/usr/bin", "--tmpfs", "/tmp"]


def run_tests(lang: str, code: str, test: str, timeout: int = 60) -> bool:
    with tempfile.TemporaryDirectory() as d:
        sb = BWRAP + ["--bind", d, "/w", "--chdir", "/w"]
        try:
            if lang == "rust":
                body = code + "\n\n" + test
                if "fn main" not in body:
                    body += "\nfn main() {}\n"
                Path(d, "m.rs").write_text(body)
                if subprocess.run(sb + ["rustc", "--edition", "2021", "-O", "-o", "/w/m", "/w/m.rs"],
                                  capture_output=True, timeout=timeout).returncode:
                    return False
                cmd = sb + ["/w/m"]
            elif lang == "sh":
                Path(d, "t.sh").write_text(code + "\n\n" + test + "\n")
                cmd = sb + ["bash", "/w/t.sh"]
            else:
                Path(d, "t.R").write_text(code + "\n\n" + test + "\n")
                cmd = sb + ["Rscript", "--vanilla", "/w/t.R"]
            return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def fetch(split: str, per_lang: int, seed: int) -> list[dict]:
    """per_lang rows from spread-out blocks of 100 (rows are grouped by source repo)."""
    rng = random.Random(f"{seed}:{split}")
    offs = sorted(rng.sample(range(0, TOTAL[split] - 100, 100), per_lang // 100))
    rows = []
    for off in offs:
        url = ROWS_API.format(split=split, off=off)
        with urllib.request.urlopen(url, timeout=120) as r:
            rows += [x["row"] | {"_off": off + x["row_idx"] % 100} for x in json.load(r)["rows"]]
    return rows


def prompt_for(lang: str, t: dict) -> str:
    return (f"{t['task'].strip()}\n\nYour code should pass these tests:\n```\n{t['test'].strip()}\n```\n"
            f"Write the {NAME[lang]} code only: do not repeat the tests"
            + (" or add a `main` function." if lang == "rust" else "."))


def ask(base: str, prompt: str) -> tuple[str, str]:
    body = {"model": "ornith-1.5-35b-a3b", "temperature": 0, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ch = json.load(r)["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason") or ""


def validate(args) -> None:
    out = args.tasks.open("w")
    for lang in ("sh", "r", "rust"):
        rows = fetch(lang, args.per_lang, args.seed)
        rows = [t for t in rows if not SKIP[lang].search(t["solution"] + "\n" + t["test"])]
        ok = 0
        with ThreadPoolExecutor(args.jobs) as ex:
            for t, good in zip(rows, ex.map(lambda t: run_tests(lang, t["solution"], t["test"]), rows)):
                if good:
                    ok += 1
                    out.write(json.dumps({"lang": lang, "uuid": t["uuid"], "off": t["_off"],
                                          "task": t["task"], "test": t["test"]}) + "\n")
        out.flush()
        print(f"{lang}: {ok}/{len(rows)} reference solutions pass (after skip filter)", flush=True)


def generate(args) -> None:
    tasks = [json.loads(l) for l in args.tasks.open()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["key"] for l in args.out.open()}
    tasks = [t for t in tasks if f"ultracode-l3:{t['lang']}:{t['uuid']}" not in done]
    stats = {"kept": 0, "failed": 0, "nocode": 0, "truncated": 0, "done": 0}
    lock = threading.Lock()
    out = args.out.open("a")

    def one(t: dict) -> None:
        lang = t["lang"]
        prompt = prompt_for(lang, t)
        answer, finish = ask(args.base, prompt)
        m = re.search(rf"```(?:{FENCE[lang]})?\s*\n(.*?)```", answer, re.S)
        code = m.group(1) if m else answer
        if finish == "length":
            res = "truncated"
        elif not code.strip():
            res = "nocode"
        elif not run_tests(lang, code, t["test"]):
            res = "failed"
        else:
            res = "kept"
        with lock:
            stats[res] += 1
            stats["done"] += 1
            if res == "kept":
                out.write(json.dumps({
                    "key": f"ultracode-l3:{lang}:{t['uuid']}",
                    "origin": f"openbmb/UltraData-Code:L3:{lang}:{t['off']}",
                    "license": "apache-2.0", "category": "anchor",
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": answer.strip()}]}) + "\n")
                out.flush()
            if stats["done"] % 25 == 0:
                print(f"  {stats['done']}/{len(tasks)}  {stats}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, tasks))
    print(f"tasks {len(tasks)}  {stats}  -> {args.out}")


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from owl_ports import BENCH_URL, refuse_live  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--per-lang", type=int, default=600, help="multiple of 100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6, help="parallel sandboxed test runs")
    ap.add_argument("--tasks", type=Path, default=ROOT / "data" / "out" / "ultra_tasks.jsonl")
    ap.add_argument("--base", default=BENCH_URL)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "out" / "anchor_ultra.jsonl")
    ap.add_argument("--workers", type=int, default=2, help="the bench unit runs --parallel 2")
    args = ap.parse_args()
    refuse_live(args.base)
    validate(args) if args.validate else generate(args)


if __name__ == "__main__":
    main()
