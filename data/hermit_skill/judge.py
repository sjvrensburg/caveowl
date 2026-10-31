"""Hermit judge: skill_gate_wide.judge on the pi session, with its reasons explained as feedback.

The specs (expected values) live in .private/specs.json, written by build_tasks.py and read only
here, so the learner-visible tasks.jsonl carries no answers.

Score:
  1.0   pass: the skill was followed through (read SKILL.md, ran its command successfully,
        grounded final answer) and every per-family check held. The gate's primary score.
  0.25  the task outcome was right but not through the skill (skill_gate_wide.ROUTE_ONLY
        reasons only). Partial credit so GEPA can tell "right answer, wrong route" from
        "wrong", yet a skipped skill can never look as good as a followed one.
  0.0   anything else, including no session (timeout, pi crash).

Feedback names each failure reason and what it means. It never quotes the expected value: the
adapter should learn to follow skills, not memorise these fixtures' answers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / "data", ROOT / "eval", ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import pi_traj as P  # noqa: E402
from skill_gate_wide import judge as wide_judge, outcome_ok  # noqa: E402

from hermit.task import JudgeResult, RunOutput  # noqa: E402

SPECS = json.loads((HERE / ".private" / "specs.json").read_text())

EXPLAIN = {
    "skill_not_read": "the agent never read the relevant skill's SKILL.md before answering",
    "script_not_run": "the agent read the skill but never ran the command the skill prescribes",
    "script_failed": "the prescribed command ran but never succeeded (error or empty output)",
    "grounded_off_script": "the answer came from some other route than the skill's command",
    "no_final_answer": "the session ended without a final answer to the user",
    "answer_not_grounded": "the final answer does not contain the facts the tool output showed",
    "procedure_not_followed": "the answer skipped the structure the skill requires",
    "input_overwritten": "the agent overwrote the input file it was asked about",
    "invented_output": "the answer shows command output or file content no tool produced",
    "unaddressed_tool_error": "a tool call failed and the agent went on as if it had not",
    "length_capped": "the reply hit the output length cap",
    "turn_error": "a model turn ended in an error",
    "guard_block": "the bench guard blocked a command the agent tried",
    "file_not_read": "the agent never read the file the user asked about",
    "command_not_run": "the agent never ran a command it needed",
    "no_tool_calls": "the agent answered without using any tool",
    "timeout": "the session timed out",
    "no_session": "pi produced no session (crashed or refused to start)",
}


def score(task: dict, output: RunOutput) -> JudgeResult:
    spec = SPECS[task["id"]]
    out = output.output or {}
    if not out.get("session"):
        why = ["timeout" if out.get("rc") == "timeout" else "no_session"]
        ok, outcome = False, False
    else:
        ok, why = wide_judge(P.load_session(out["session"]), spec)
        outcome = outcome_ok(why)
    s = 1.0 if ok else (0.25 if outcome else 0.0)
    if ok:
        fb = "Pass: the skill was loaded and followed through, and the answer is grounded."
    else:
        fb = "Fail: " + "; ".join(f"{r} ({EXPLAIN.get(r, 'see the trace')})" for r in why)
        if outcome:
            fb += ". The answer itself was right, but not reached through the skill."
    return JudgeResult(score=s, feedback=fb, extra={"family": task.get("family"), "reasons": why})
