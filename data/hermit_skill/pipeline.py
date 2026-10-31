"""Hermit acting pipeline: one real pi session per task, against caveowl on the bench port.

Hermit has no pi acting agent (its built-ins are react, codex and claude-code; pi is only a
deploy target), so this is the task's own: `agent.name: pipeline.py:run` in run.yaml.

It is skill_gate_wide.py's loop for one item, so a Hermit score and a gate score mean the same:
  - data/pi_traj.py make_agent_dir: a scratch PI_CODING_AGENT_DIR with the real skills, ONE
    provider pointed at the bench URL, no extensions but data/pi_bench_guard.ts, no packages
    (memory and session search would read the operator's real data);
  - scratch_cwd: a fresh cwd outside any project, fixtures written by skill_gate_wide.setup_work;
  - run_pi: `pi -p` with every *_API_KEY dropped from its environment. Hermit runs under the
    operator's Anthropic key for the evolver; pi must never see it.

The adapter is deployed the way the operator deploys AGENTS.md by hand: the adapter's context
text (ADAPTER.md, through hermit.Session) written verbatim as the cwd's AGENTS.md, with pi's
context files on. hermit.hosts.pi.prepare is not used, because it wraps the text in Hermit's
own markers, which production would not have. adapter=None (Hermit's baseline) runs with no
AGENTS.md and context files off, exactly the gate's default arm.

params (run.yaml agent.params):
  base      bench URL; default owl_ports.BENCH_URL (:8110). refuse_live() stops a live port.
  model     the model id the bench server answers to; default pi_traj.MODEL
  thinking  pi --thinking; default "off", the setting caveowl's 38/81 wide-gate result used.
            Production pi-signal runs "medium" since the judge was retired (halo-prep CLAUDE.md).
  timeout   seconds per session; default 600
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / "data", ROOT / "eval", ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))  # appended, not prepended: never shadow the hermit package

import pi_traj as P  # noqa: E402
from owl_ports import BENCH_URL, refuse_live  # noqa: E402
from skill_gate_wide import setup_work  # noqa: E402

from hermit import Session  # noqa: E402
from hermit.task import RunOutput  # noqa: E402

# pi_traj.run_pi calls bare `pi`. A systemd unit's PATH lacks ~/.local/bin, and then every rollout
# raises FileNotFoundError, which Hermit records as an error scoring 0 and trains on anyway: the
# first training run (2026-10-31) spent 20 steps that way. Refuse to start instead.
if shutil.which("pi") is None:
    raise SystemExit("pipeline.py: `pi` is not on PATH (systemd units: pass --setenv=PATH=$PATH)")

_local = threading.local()
_RUN = uuid.uuid4().hex[:8]  # one tag per Hermit process, so concurrent runs never share dirs


def _agent_dir(base: str, model: str) -> Path:
    """One scratch agent dir per worker thread, built once (it copies the skills tree)."""
    if getattr(_local, "agent", None) is None:
        name = f"agent-hermit-{_RUN}-{os.getpid()}-{threading.get_ident()}"
        _local.agent = P.make_agent_dir(P.WORK_ROOT / name, base, P.PROVIDER, model)
    return _local.agent


def to_messages(msgs: list[dict]) -> list[dict]:
    """pi session messages as chat messages, the shape Hermit's evolvers render as a trace."""
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            calls = [{"id": c.get("id"), "type": "function",
                      "function": {"name": c.get("name"), "arguments": c.get("arguments")}}
                     for c in m.get("content") or [] if isinstance(c, dict) and c.get("type") == "toolCall"]
            out.append({"role": "assistant", "content": P._text(m.get("content")),
                        **({"tool_calls": calls} if calls else {})})
        elif role == "toolResult":
            text = P._text(m.get("content"))
            out.append({"role": "tool", "tool_call_id": m.get("toolCallId"),
                        "content": ("[error] " if m.get("isError") else "") + text})
        elif role in ("user", "system"):
            out.append({"role": role, "content": P._text(m.get("content"))})
    return out


def run(task: dict, adapter: Path | None, params: dict | None = None, base: Path | None = None) -> RunOutput:
    params = params or {}
    url = params.get("base") or BENCH_URL
    refuse_live(url)
    model = params.get("model") or P.MODEL
    agent = _agent_dir(url, model)
    work = P.scratch_cwd("hermit", _RUN, f"{task['id']}-{uuid.uuid4().hex[:6]}")
    setup_work(work, task.get("fixture") or {})
    context = ""
    if adapter is not None:
        context = Session(adapter).context() or ""
        (work / "AGENTS.md").write_text(context)
    res = P.run_pi(agent, work, task["prompt"], P.PROVIDER, model,
                   timeout=int(params.get("timeout", 600)),
                   thinking=params.get("thinking", "off"), context_files=adapter is not None)
    msgs = P.load_session(res["session"]) if res["session"] else []
    chat = to_messages(msgs)
    if context:
        chat = [{"role": "system", "content": context}] + chat
    return RunOutput(output={"session": res["session"], "rc": res["rc"]},
                     log=Path(res["session"]) if res["session"] else None, messages=chat, wall_s=res["seconds"],
                     cost_usd=0.0, tokens=None,
                     extra={"stderr_tail": res["stderr_tail"]} if res["rc"] not in (0, None) else {})
