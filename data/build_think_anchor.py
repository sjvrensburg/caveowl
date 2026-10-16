"""Thinking-ON rows: stock Ornith's own reasoning, closed, then the caveman answer (run 5).

Why: every training row through run 4 is thinking-off (the empty think block is prompt). With
thinking on, the server's default when a request names no enable_thinking (pi's `local-ornith35`
at "medium", Open Notebook), run 4 wrote its whole caveman answer INSIDE the reasoning and
stopped without `</think>`: prose-check (which sends no kwarg) got empty content on 5/10 prompts
(stock 2/10). The model never saw a closed think block it had to produce itself.

What: prompts from the caveman text pairs (the same source as the text slice; single-turn QA
only, prompt under 400 characters: stock reasons for 3k+ tokens on long ones and most draws hit the
cap), answered by stock on :8110 with thinking ON at the served sampler. A draw is kept only if
the reasoning is non-empty and closed and the visible answer is non-empty (finish_reason stop),
with the reasoning under --max-reasoning tokens. The row is
    user -> assistant{reasoning_content: STOCK'S reasoning, content: the pair's CAVEMAN answer}
rendered thinking-on by owl_data (reasoning, </think>, answer and <|im_end|> all carry loss): it
keeps stock's way of thinking and puts the style where the style belongs, after </think>.

Run (stock on :8110; refuses :8092/:8088):
    ~/venvs/unsloth-probe/bin/python data/build_think_anchor.py --n 300
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import owl_data as D  # noqa: E402

OUT = ROOT / "data" / "out" / "think_anchor.jsonl"
SRC = ROOT.parent / "cavegemma-e4b" / "data" / "out" / "train.jsonl"


def _k(r: dict) -> str:  # the text pairs carry no key: hash the prompt
    import hashlib
    return hashlib.sha1((r.get("prompt") or "").encode()).hexdigest()[:16]


def ask(base: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": "ornith-1.5-35b-a3b", "max_tokens": max_tokens, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8110/v1")
    ap.add_argument("--n", type=int, default=300, help="prompts to try")
    ap.add_argument("--max-reasoning", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    assert not re.search(r":(8092|8088)\b", a.base), "bench port only"
    rows = [r for r in D.read_jsonl(SRC) if (r.get("category") or "qa").lower() != "dialogue"]
    random.Random("caveowl-run5-think").shuffle(rows)
    cands = []
    for r in rows:
        msgs = D.answer_messages(r)
        if msgs and len(msgs) == 2 and not D.has_marker(msgs) and len(msgs[0]["content"]) < 400:
            cands.append((r, msgs))
        if len(cands) >= a.n:
            break
    done = {json.loads(l)["key"] for l in OUT.open()} if OUT.exists() else set()
    todo = [(r, m) for r, m in cands if f"think:{_k(r)}" not in done]
    why: dict[str, int] = {}

    def one(rm):
        r, msgs = rm
        try:
            d = ask(a.base, msgs[0]["content"], a.max_reasoning + 1000)
        except Exception:
            return None, "error"
        ch = d["choices"][0]
        m = ch["message"]
        reasoning = (m.get("reasoning_content") or "").strip()
        content = (m.get("content") or "").strip()
        if ch.get("finish_reason") != "stop":
            return None, "not_stop"
        if not reasoning or not content:
            return None, "empty_reasoning_or_answer"
        if d["usage"]["completion_tokens"] > a.max_reasoning + 1000 or len(reasoning) > a.max_reasoning * 4:
            return None, "too_long"
        if any(mk in reasoning for mk in D.MARKERS):
            return None, "marker"
        return {"key": f"think:{_k(r)}", "origin": f"stock reasoning + {r.get('origin') or r.get('source_dataset') or 'cavegemma pair'}",
                "license": r.get("license", "see cavegemma"), "category": "think",
                "messages": [msgs[0], {"role": "assistant", "reasoning_content": reasoning,
                                       "content": msgs[1]["content"]}],
                "stock_answer": content}, "kept"

    with OUT.open("a") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        for row, reason in ex.map(one, todo):
            why[reason] = why.get(reason, 0) + 1
            if row:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    print(f"{why} -> {OUT}")


if __name__ == "__main__":
    main()
