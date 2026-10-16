"""Thinking-ON check: does the model close its reasoning and give a visible answer?

Run 4 (README "Run 4") wrote its caveman answer inside the reasoning and stopped without
`</think>` on 5 of llm-bench-kit/prose-check.py's 10 prompts (stock 2/10). prose-check sends no
enable_thinking, so the server's default (thinking on) applies, which is what pi's
`local-ornith35` at "medium" and Open Notebook get from :8088.

16 prompts (disjoint from prose-check's and from the text pairs' style of prompt), each sent
twice with NO chat_template_kwargs and no sampler fields (the server's own sampler, as a client
that sends none gets), max_tokens 4096. Counts per arm: empty visible answer, unfinished
(finish_reason length), median words of the answers that exist. Pass: empty <= stock's.

    python3 -I eval/think_on.py --base http://127.0.0.1:8110/v1 --label stock
"""
import argparse
import json
import re
import statistics
import urllib.request
from pathlib import Path

PROMPTS = [
    "Why do onions make people cry?",
    "What's the difference between a mutex and a semaphore?",
    "How does compound interest work? Give a short example.",
    "Explain what DNS does when I type a URL.",
    "What should I check before buying a second-hand bicycle?",
    "Why is the ocean salty but rivers are not?",
    "What is a hash table and why are lookups fast?",
    "Give me three tips for writing a clear commit message.",
    "How do vaccines train the immune system?",
    "What's the difference between TCP and UDP?",
    "Why does bread go stale?",
    "Explain big-O notation to a beginner.",
    "How do noise-cancelling headphones work?",
    "What are the pros and cons of working from home?",
    "Why do we have leap years?",
    "What is the difference between weather and climate?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8110/v1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    assert not re.search(r":(8092|8088)\b", a.base), "bench port only"
    out = Path(__file__).resolve().parent / "out" / f"think_on-{a.label}.jsonl"
    empty = unfinished = 0
    words = []
    with out.open("w") as fh:
        for p in PROMPTS:
            for _ in range(a.repeats):
                body = {"model": "m", "max_tokens": 4096, "messages": [{"role": "user", "content": p}]}
                req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                d = json.load(urllib.request.urlopen(req, timeout=900))
                ch = d["choices"][0]
                c = (ch["message"].get("content") or "").strip()
                fr = ch.get("finish_reason")
                empty += not c
                unfinished += fr != "stop"
                if c:
                    words.append(len(c.split()))
                fh.write(json.dumps({"prompt": p, "finish": fr, "content": c,
                                     "reasoning": ch["message"].get("reasoning_content")}) + "\n")
    n = len(PROMPTS) * a.repeats
    med = statistics.median(words) if words else 0
    print(f"=== {a.label}: empty {empty}/{n}  unfinished {unfinished}/{n}  median words {med}  -> {out}")


if __name__ == "__main__":
    main()
