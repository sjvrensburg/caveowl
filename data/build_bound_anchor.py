"""Self-distilled "bounded tool call" anchors (run 4 contingency, 2026-10-13).

Why: runs 1 and 2 both failed sampler-check at :8088's sampler (temp 0.6): 27/30, from a dropped
`max_results`, `terminal` called where a dedicated web_search tool existed, and once a wrong
unit. Stock is 30/30. Nothing in the mix teaches "the user named a bound/flag, carry it into the
call": the tool slice (smoltalk2 xlam) is single-tool calls with every argument spelled out, the
skill rows and synthtraces only have pi's read/bash/edit/write.

What: two-requirement prompts (pitfall #25's shape: a target AND a bound/flag) over toolsets of
3-5 tools that mix a shell tool with dedicated ones. Stock Ornith on :8110 answers each at the
served sampler, thinking off; a call is kept only if a checker confirms the right tool family and
every stated requirement in the arguments. Rows end at the tool call (the tool slice does the
same), so loss falls on the call alone.

Disjoint from llm-bench-kit/tool-harness.py on purpose: none of its three prompts' targets
(/etc/fstab, build b10582, llama-gemma26), its tool descriptions or its exact schemas. Tool
names overlap only where the name is generic (web_search, read_file); parameter names vary
(max_results / limit / count / top_k / max_lines / lines / n) so what is learnt is "carry the
bound", not one schema.

Run (stock on the bench port; owl_ports.refuse_live guards live servers):
    python3 data/build_bound_anchor.py --n 400
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))  # owl_ports
from owl_ports import BENCH_URL, refuse_live  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "bound_anchor.jsonl"
MODEL = "ornith-1.5-35b-a3b"
SAMPLER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
SHELLS = [fn("shell", "Execute a shell command and return stdout/stderr.", {"cmd": S}, ["cmd"]),
          fn("run_command", "Run a command in a bash shell on the host.", {"command": S}, ["command"]),
          fn("terminal", "Run a command in the user's terminal.", {"command": S}, ["command"])]
SEARCH = [("web_search", "Look something up on the internet.", "max_results"),
          ("search_web", "Search the public web and return result snippets.", "count"),
          ("internet_search", "Query a web search engine.", "limit")]
READ = [("read_file", "Return the contents of a text file.", "max_lines"),
        ("view_file", "Show a file's text.", "lines"),
        ("cat_file", "Read a file from the local disk.", "limit")]
EXTRA = [fn("get_weather", "Weather forecast for a city.", {"city": S, "days": I}, ["city"]),
         fn("send_message", "Send a chat message to a contact.", {"to": S, "text": S, "silent": B}, ["to", "text"]),
         fn("list_dir", "List a directory.", {"path": S, "depth": I}, ["path"]),
         fn("git_log", "Show recent commits of a repository.", {"repo": S, "n": I, "author": S}, ["repo"]),
         fn("db_query", "Run a read-only SQL query.", {"sql": S, "row_limit": I}, ["sql"]),
         fn("calendar_events", "List upcoming calendar events.", {"days_ahead": I, "calendar": S}, [])]

TOPICS = ["the Rust 1.90 release announcement", "Mesa 26.2 RADV changelog", "PostgreSQL 18 vacuum changes",
          "the Fedora 44 release schedule", "Python 3.15 free-threading status", "zstd long-range mode benchmarks",
          "systemd 259 new features", "the Wasmtime 40 release", "KDE Plasma 6.6 Wayland fixes",
          "Linux 7.2 scheduler changes", "SQLite JSONB performance", "Go 1.27 iterator proposals",
          "Debian trixie point release notes", "Podman 6 quadlet changes", "ffmpeg 8 AV1 encoder options",
          "the latest Tailscale release notes", "Firefox 150 privacy changes", "Node.js 26 LTS date",
          "uv lockfile format changes", "Blender 5.1 geometry nodes"]
FILES = ["/etc/hosts", "/etc/os-release", "~/.bashrc", "/var/log/dnf.log", "~/notes/todo.md",
         "/etc/ssh/sshd_config", "~/Projects/site/README.md", "/proc/cpuinfo", "/etc/resolv.conf",
         "~/.config/git/config", "/etc/passwd", "~/data/report.csv"]
UNITS = ["backup-sync", "photo-indexer", "mail-fetch", "rss-poller", "syncthing", "kopia-snapshot",
         "home-dashboard", "podcast-fetch"]
NUMS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 50]
SEARCH_P = ["Search the web for {t} and give me only the top {n} results.",
            "Look up {t}; I just want {n} results, no more.",
            "Find {t} online, limit it to {n} hits.",
            "web search: {t}. Max {n} results please.",
            "Can you search for {t}? Only {n} results."]
READ_P = ["Show me the first {n} lines of {f}.", "Print just the top {n} lines of {f}.",
          "Read {f} but only the first {n} lines.", "head of {f}, {n} lines please"]
UNIT_P = ["Is the {u} service running? It's a systemd user unit, not a system one.",
          "Check the status of my {u} user service (systemctl --user, not system).",
          "Restart-check: is {u} active? It runs as a user unit under my login.",
          "What does journalctl say about {u}? It's a --user unit; last {n} lines only."]


def make_items(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    seen, items = set(), []
    while len(items) < n:
        kind = rng.choice(["search", "search", "read", "read", "unit"])
        shell = rng.choice(SHELLS)
        tools = [shell] + rng.sample(EXTRA, rng.randint(1, 3))
        num = rng.choice(NUMS)
        if kind == "search":
            name, desc, bound = rng.choice(SEARCH)
            tools.append(fn(name, desc, {"query": S, bound: I}, ["query"]))
            prompt = rng.choice(SEARCH_P).format(t=rng.choice(TOPICS), n=num)
            spec = {"kind": kind, "tool": name, "bound": bound, "n": num}
        elif kind == "read":
            name, desc, bound = rng.choice(READ)
            tools.append(fn(name, desc, {"path": S, bound: I}, ["path"]))
            f = rng.choice(FILES)
            prompt = rng.choice(READ_P).format(f=f, n=num)
            spec = {"kind": kind, "tool": name, "bound": bound, "n": num, "path": f, "shell": shell["function"]["name"]}
        else:
            u = rng.choice(UNITS)
            prompt = rng.choice(UNIT_P).format(u=u, n=num)
            spec = {"kind": kind, "unit": u, "n": num if "last {n}" in prompt else None,
                    "shell": shell["function"]["name"]}
        rng.shuffle(tools)
        k = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if k in seen:
            continue
        seen.add(k)
        items.append({"key": f"bound:{k}", "prompt": prompt, "tools": tools, "spec": spec})
    return items


def check(call: dict, spec: dict) -> tuple[bool, str]:
    name, a = call["name"], call["args"]
    cmd = next((a[k] for k in ("cmd", "command") if isinstance(a.get(k), str)), "")
    if spec["kind"] == "search":
        if name != spec["tool"]:
            return False, f"called {name}, not {spec['tool']}"
        if not a.get("query"):
            return False, "empty query"
        return (a.get(spec["bound"]) == spec["n"]), f"{spec['bound']}={a.get(spec['bound'])!r}"
    if spec["kind"] == "read":
        p = spec["path"]
        if name == spec["tool"]:
            ok = a.get("path") in (p, p.replace("~", str(Path.home()))) and a.get(spec["bound"]) == spec["n"]
            return ok, json.dumps(a)
        if name == spec["shell"]:
            ok = p.split("/")[-1] in cmd and bool(re.search(rf"\b(head|sed)\b[^|;]*(?<!\d){spec['n']}(?!\d)", cmd))
            return ok, cmd
        return False, f"called {name}"
    if name != spec["shell"]:
        return False, f"called {name}"
    ok = spec["unit"] in cmd and "--user" in cmd
    if spec["n"]:
        ok = ok and bool(re.search(rf"(?<!\d){spec['n']}(?!\d)", cmd))
    return ok, cmd


def ask(base: str, item: dict) -> dict | None:
    body = {"model": MODEL, "max_tokens": 1024, **SAMPLER,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": item["prompt"]}], "tools": item["tools"]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if len(tcs) != 1:
        return None
    f = tcs[0]["function"]
    try:
        args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
    except json.JSONDecodeError:
        return None
    ok, why = check({"name": f["name"], "args": args}, item["spec"])
    if not ok:
        return None
    return {"key": item["key"], "origin": "self-distilled:ornith-1.5-35b-a3b@temp0.6",
            "license": "generated", "category": "tool_call", "tools": item["tools"],
            "messages": [{"role": "user", "content": item["prompt"]},
                         {"role": "assistant", "content": (msg.get("content") or "").strip(),
                          "tool_calls": [{"id": "call_0", "type": "function",
                                          "function": {"name": f["name"], "arguments": args}}]}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BENCH_URL)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="print items, call nothing")
    a = ap.parse_args()
    refuse_live(a.base)
    items = make_items(a.n)
    if a.dry:
        for it in items[:8]:
            print(it["spec"], "|", it["prompt"], "|", [t["function"]["name"] for t in it["tools"]])
        return
    done = {json.loads(l)["key"] for l in OUT.open()} if OUT.exists() else set()
    todo = [it for it in items if it["key"] not in done]
    kept = miss = 0
    with OUT.open("a") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        def one(it):
            for _ in range(a.tries):
                try:
                    r = ask(a.base, it)
                except Exception:
                    r = None
                if r:
                    return r
            return None
        for r in ex.map(one, todo):
            if r:
                fh.write(json.dumps(r) + "\n"); fh.flush(); kept += 1
            else:
                miss += 1
    print(f"kept {kept}, no correct call in {a.tries} tries: {miss} -> {OUT}")


if __name__ == "__main__":
    main()
