"""Convert julien-c/synthtraces pi sessions from Qwen3.6 agents into training rows (run 2).

Source: https://huggingface.co/datasets/julien-c/synthtraces (MIT), downloaded to
data/out/ext/synthtraces/ (`hf download julien-c/synthtraces --repo-type dataset`). Native pi
session JSONL: a remote open model is the coding agent (pi's default read/bash/edit/write
tools) inside one of 22 Hugging Face repos; a local model plays the user. As of 2026-10-13 the
repo holds 2,876 of the card's planned 24,000 sessions.

Why these, and only these (operator's call, 2026-10-13): Ornith-1.5-35B-A3B is a Qwen3.6-35B-A3B
fine-tune, so trajectories from Qwen3.6-35B-A3B (149 sessions) and Qwen3.6-27B (109) are the
closest public agent loops to the student's own distribution. They add many-step pi tool loops
(median ~9-10 calls, real tool output, real recoveries) that run 1's self-distilled rows have
few of. They contain NO skill reads, so they cannot teach skill follow-through; that stays the
job of data/build_skill_anchor.py.

Conversion, and why:
  - First exchange only: the opening question, the tool loop, the first answer with no tool
    call. Turns 2-5 are the simulated user, which mostly sends `null` (4 of 5 user turns in a
    typical session), and the agent then answers nothing in particular.
  - Thinking dropped: the sessions ran at thinking "medium"; caveowl trains thinking OFF.
  - System prompt and tool schemas: pi 1.1.0's, as captured in data/out/skill_traj.jsonl,
    with the <skills> block REMOVED and <cwd> set to the session's repo. A skill list the agent
    never consults would teach exactly the skill-skipping the run-1 gate failed on.
  - Paths: /home/gibbon/synthtraces/repos/<repo> -> ~/src/<repo> (this user's home), everywhere.
  - Dropped: any session whose exchange calls a tool pi does not have (the 35B emitted a few
    calls named after a shell command), ends with stopReason error/length, or has no answer.
  - Failed guess calls (command not found / no such file) are pruned as in build_skill_anchor.

Run:
    ~/venvs/unsloth-probe/bin/python data/build_synthtraces.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from build_skill_anchor import GUESS_ERROR, prune  # noqa: E402

MODELS = {"Qwen/Qwen3.6-35B-A3B": "qwen36-35b-a3b", "Qwen/Qwen3.6-27B": "qwen36-27b"}
TOOLS = {"read", "bash", "edit", "write"}
SRC_ROOT = "/home/gibbon/synthtraces/repos/"
DST_ROOT = str(Path.home() / "src") + "/"


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if c.get("type") == "text")


def template(skill_traj: Path) -> tuple[str, list]:
    row = json.loads(skill_traj.open().readline())
    sp = row["messages"][0]["content"]
    sp = re.sub(r"\n*<skills>.*?</skills>\n*", "\n\n", sp, flags=re.S)
    assert "<skills>" not in sp and "<cwd>" in sp
    return sp, row["tools"]


def convert(path: str, sp: str, tools: list) -> tuple[dict | None, str]:
    ev = [json.loads(l) for l in open(path)]
    mid = next((e["modelId"] for e in ev if e["type"] == "model_change"), None)
    if mid not in MODELS:
        return None, "other_model"
    cwd = next(e["cwd"] for e in ev if e["type"] == "session")
    fix = lambda s: s.replace(SRC_ROOT, DST_ROOT)  # noqa: E731
    msgs = [e["message"] for e in ev if e["type"] == "message"]
    if not msgs or msgs[0]["role"] != "user":
        return None, "no_question"
    q = text_of(msgs[0]["content"]).strip()
    if not q or q == "null":
        return None, "no_question"
    out = [{"role": "system", "content": re.sub(r"<cwd>\n.*?\n</cwd>", f"<cwd>\n{fix(cwd)}\n</cwd>", sp, flags=re.S)},
           {"role": "user", "content": q}]
    err_ids, failed_ids = set(), set()
    for m in msgs[1:]:
        if m["role"] == "user":
            return None, "user_before_answer"
        if m["role"] == "toolResult":
            body = fix(text_of(m.get("content")))
            if m.get("isError"):
                failed_ids.add(m["toolCallId"])
                if GUESS_ERROR.search(body):
                    err_ids.add(m["toolCallId"])
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": body})
            continue
        if m.get("stopReason") in ("error", "length", "aborted"):
            return None, f"stop_{m.get('stopReason')}"
        calls = [c for c in m["content"] if c.get("type") == "toolCall"]
        if any(c["name"] not in TOOLS for c in calls):
            return None, "bad_tool_name"
        txt = fix(text_of(m["content"]).strip())
        if not calls:
            if not txt:
                return None, "empty_answer"
            out.append({"role": "assistant", "content": txt})
            try:
                body, n_err, n_dup = prune(out[1:], err_ids, failed_ids)
            except ValueError as e:  # the answer talks about a pruned failed step
                return None, str(e)
            repo = Path(cwd).name
            sid = Path(path).stem.split("_")[-1]
            return {"key": f"synthtraces:{sid}", "origin": f"julien-c/synthtraces:{mid}:{repo}:{Path(path).name}",
                    "license": "mit", "category": "ext_traj", "family": f"repo-qa:{MODELS[mid]}",
                    "tools": tools, "messages": out[:1] + body,
                    "pruned": {"failed_calls": n_err, "repeats": n_dup}}, "kept"
        out.append({"role": "assistant", "content": txt, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.loads(fix(json.dumps(c.get("arguments") or {})))}}
            for c in calls]})
    return None, "no_answer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/out/ext/synthtraces")
    ap.add_argument("--template", type=Path, default=ROOT / "data/out/skill_traj.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/out/ext_traj.jsonl")
    args = ap.parse_args()
    sp, tools = template(args.template)
    why, rows = Counter(), []
    for f in sorted(glob.glob(str(args.src / "sessions/*/*.jsonl"))):
        row, reason = convert(f, sp, tools)
        why[reason] += 1
        if row:
            rows.append(row)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    del why["other_model"]
    print(f"kept {len(rows)}  {dict(why)}  by model {dict(Counter(r['family'] for r in rows))}  -> {args.out}")


if __name__ == "__main__":
    main()
