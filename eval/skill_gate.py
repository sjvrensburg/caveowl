"""Skill follow-through gate: does a candidate, driven through REAL pi, finish multi-step work?

caveowl copy of ../cavegemma-e4b/eval/skill_gate.py (9dd4267): the pi provider/model are Ornith's
(data/pi_traj.py PROVIDER, MODEL); prompts, judge and pass rule are unchanged. Run stock and
candidate on the same bench flags (scripts/serve_bench.sh), one at a time on :8110.

Why this exists: v2 (2026-10-11) passed tools 9/9, code 12/12 and the reasoning-leak check, went
live on :8092, and was rolled back the next morning. On Signal it saw the weather skill and
never read SKILL.md or ran the script ("Skill exist. Not loadable now.", 0/3 against stock's
2/3). Every earlier check scores one request or the first tool call; this one scores whole pi
sessions with the same checks data/build_skill_anchor.py uses to keep training rows
(data/pi_traj.py judge_session): read the skill's SKILL.md, ran its script/command, final answer
non-empty and grounded in a tool result, no invented output, no unaddressed tool error, no
length-capped turn.

The prompts here are HELD OUT: 10 weather cities/phrasings and 5 generic task templates that
data/build_skill_anchor.py never uses (it imports these lists and refuses to start on a
collision). pi runs with its real, full system prompt and tool set (the same scratch agent dir
as the build: real skills, no extensions but data/pi_bench_guard.ts, no packages).

Usage (candidate served on a bench port, never :8092):
    ~/venvs/unsloth-probe/bin/python eval/skill_gate.py --base http://127.0.0.1:8110/v1 --model ornith-1.5-35b-a3b \\
        --label stock --repeats 3
    # Signal's own project context (AGENTS.md) instead of none:
    ~/venvs/unsloth-probe/bin/python eval/skill_gate.py ... --agents-md ~/.local/state/pi-signal/work/AGENTS.md
Pass rule (README "Before it touches :8092"): candidate total >= stock total, and weather
>= stock weather, on the same server flags, same repeats.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import pi_traj as P  # noqa: E402

GATE_WEATHER = [
    ("Gqeberha", "What's the weather like today in {c}?"),   # the exact Signal prompt v2 failed
    ("Summerstrand", "Any rain expected in {c} today?"),
    ("Durban", "Give me the {c} forecast."),
    ("Reykjavik", "How warm is it in {c} at the moment?"),
    ("Nairobi", "What should I wear in {c} today, weather-wise?"),
    ("Lima", "Weather check: {c}"),
    ("Osaka", "Is it a good day for a walk in {c}?"),
    ("Edinburgh", "Will I need sunglasses or a raincoat in {c} tomorrow?"),
    ("Perth", "What are the highs and lows in {c} for the next two days?"),
    ("Montreal", "Is it cold in {c} today?"),
]
GATE_GENERIC_IDS = ("g_ips", "g_slowest", "g_maxclients", "g_squares", "g_norders")
GATE_SEED = 7777


def gate_generic(f: dict) -> list[tuple[str, str, dict]]:
    n = 20 + f["port"] % 30
    sq = sum(k * k for k in range(1, n + 1))
    return [
        ("g_ips", "How many distinct IP addresses appear in logs/access.log?",
         {"expect": [str(f["n_unique_ips"])], "must_read_re": r"access\.log"}),
        ("g_slowest", "What was the slowest request time in logs/access.log?",
         {"expect": [str(f["slowest_ms"])], "must_read_re": r"access\.log"}),
        ("g_maxclients", "What's MAX_CLIENTS set to under src/, and in which file?",
         {"expect_all": [str(f["max_clients"]), "server.py"]}),
        ("g_squares", f"Create squares.py that prints the sum of the squares of 1..{n}, run it, and tell me the result.",
         {"expect": [str(sq), f"{sq:,}"], "must_bash_re": r"squares\.py"}),
        ("g_norders", "How many orders are in data/orders.csv, not counting the header?",
         {"expect": [str(f["n_orders"])], "must_read_re": r"orders\.csv"}),
    ]


def gate_items() -> list[dict]:
    items = [{"id": f"gw-{i}", "family": "weather", "prompt": p.format(c=c),
              "spec": {"skill": "weather", "script_re": r"weather\.py", "ground": "number"}}
             for i, (c, p) in enumerate(GATE_WEATHER)]
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="gatefx-"))
    facts = P.make_fixture(d, GATE_SEED)
    shutil.rmtree(d, ignore_errors=True)
    items += [{"id": tid, "family": "generic", "prompt": pr, "spec": spec,
               "fixture": {"seed": GATE_SEED}} for tid, pr, spec in gate_generic(facts)]
    assert {i["id"] for i in items if i["family"] == "generic"} == set(GATE_GENERIC_IDS)
    return items


def wilson(k: int, n: int) -> str:
    if not n:
        return "n/a"
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{max(0, c - h):.0%}-{min(1, c + h):.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8110/v1",
                    help="OpenAI-compatible endpoint of the candidate (not :8092)")
    ap.add_argument("--model", default=P.MODEL, help="model id the server answers to")
    ap.add_argument("--label", default=None, help="name for the results file")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--thinking", default="off")
    ap.add_argument("--agents-md", type=Path, default=None,
                    help="copy this file into each run's cwd as AGENTS.md and let pi load it")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "eval" / "out" / "skill_gate")
    args = ap.parse_args()
    if ":8092" in args.base or ":8088" in args.base:
        # :8088 is stock Ornith in production (Open Notebook's single slot); pi sessions there
        # stall podcasts and chat. Serve stock on the bench port for baselines (code review).
        sys.exit("refusing :8092/:8088 (live servers); serve the model on a bench port")

    label = args.label or args.model
    rd = args.run_dir.resolve() / label
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    # Short path beside the cwds (see build_skill_anchor.py): pi lists every skill location.
    agents = [P.make_agent_dir(P.WORK_ROOT / f"agent-gate-{label}-w{w}", args.base,
                               P.PROVIDER, args.model)
              for w in range(args.workers)]
    items = gate_items()
    jobs = [(it, r) for r in range(args.repeats) for it in items]
    slots, lock, results = list(range(args.workers)), threading.Lock(), []

    def one(job):
        it, r = job
        with lock:
            w = slots.pop()
        try:
            work = P.scratch_cwd("gate", label, f"{it['id']}-r{r}")
            if it.get("fixture"):
                P.make_fixture(work, it["fixture"]["seed"])
            if args.agents_md:
                shutil.copy(args.agents_md, work / "AGENTS.md")
            res = P.run_pi(agents[w], work, it["prompt"], P.PROVIDER, args.model,
                           thinking=args.thinking, context_files=bool(args.agents_md))
            if res["session"]:  # persist it: the cwd lives under /tmp
                keep = rd / f"{it['id']}-r{r}.session.jsonl"
                shutil.copy(res["session"], keep)
                res["session"] = str(keep)
                ok, why = P.judge_session(P.load_session(res["session"]), it["spec"])
            else:
                ok, why = False, [str(res["rc"]) if res["rc"] == "timeout" else "no_session"]
            rec = {"id": it["id"], "family": it["family"], "repeat": r, "prompt": it["prompt"],
                   "pass": ok, "reasons": why, "seconds": res["seconds"], "session": res["session"]}
            with lock:
                results.append(rec)
                print(f"  {it['id']:13s} r{r}  {'PASS' if ok else 'fail'}  {res['seconds']:6.1f}s  "
                      f"{','.join(why)}", flush=True)
        finally:
            with lock:
                slots.append(w)

    print(f">>> skill gate: {label}  {len(items)} prompts x {args.repeats} repeats  base={args.base}")
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))

    results.sort(key=lambda x: (x["id"], x["repeat"]))
    fam = Counter()
    fam_ok = Counter()
    reasons = Counter()
    for x in results:
        fam[x["family"]] += 1
        fam_ok[x["family"]] += x["pass"]
        reasons.update(x["reasons"])
    tot, ok = len(results), sum(x["pass"] for x in results)
    summary = {"label": label, "base": args.base, "model": args.model, "repeats": args.repeats,
               "agents_md": str(args.agents_md) if args.agents_md else None,
               "total": [ok, tot], **{f: [fam_ok[f], fam[f]] for f in fam},
               "reasons": dict(reasons.most_common()),
               "per_prompt": {i["id"]: sum(x["pass"] for x in results if x["id"] == i["id"])
                              for i in items},
               "results": results}
    (rd / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"=== {label}: {ok}/{tot} pass (95% CI {wilson(ok, tot)})  " +
          "  ".join(f"{f} {fam_ok[f]}/{fam[f]}" for f in sorted(fam)))
    print(f"    failure reasons: {dict(reasons.most_common())}")
    print(f"    per prompt (of {args.repeats}): {summary['per_prompt']}")
    print(f"    -> {rd / 'summary.json'}")


if __name__ == "__main__":
    main()
