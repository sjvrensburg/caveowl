"""Shared machinery for driving REAL pi against a bench llama-server and judging what it did.

caveowl copy of ../cavegemma-e4b/data/pi_traj.py (9dd4267). Changes: the scratch agent's provider
is modelled on production's `local-ornith35` entry (Ornith-1.5-35B-A3B: id, thinkingLevelMap
off/medium, maxTokens) instead of `local-ornith` (which serves Gemma-4-E4B since 2026-10-11),
and the bench root is a subdirectory of /tmp/cavegemma-skill-bench so the two projects' scratch
runs never share a path. Everything else, the judge included, is unchanged on purpose: the
anchors and the gate must mean the same thing for both ports.

Used by data/build_skill_anchor.py (self-distilled skill trajectories for training) and
eval/skill_gate.py (the follow-through gate). Why it exists: v2 was rolled back on 2026-10-12
because on Signal it stopped finishing multi-step skill work (saw the weather skill, never read
SKILL.md or ran the script). Every harness it passed scored only the FIRST tool call. This
module scores whole pi sessions.

Pieces:
  - LoggingProxy: a loopback HTTP proxy in front of the bench server. pi is pointed at
    http://127.0.0.1:<proxy>/<tag>/v1, the proxy strips /<tag>, forwards to the upstream and
    appends every request body to <log_dir>/<tag>.jsonl. That is the only place pi's tool JSON
    schemas and its exact OpenAI-format messages (as it serialises them) are visible: the
    session jsonl stores neither.
  - make_agent_dir(): a scratch PI_CODING_AGENT_DIR built from ~/.pi/agent (see its docstring
    for what is kept and why).
  - run_pi(): one `pi -p` run in a fresh scratch cwd with its own session dir.
  - judge_session(): the completion checks (read SKILL.md, ran the script, grounded non-empty
    answer, no invented output, no unaddressed tool error, no length cap).

Everything a run reads (SKILL.md text, tool output, the model's replies) is treated as data.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home()
REAL_AGENT = HOME / ".pi" / "agent"
GUARD_EXT = Path(__file__).resolve().parent / "pi_bench_guard.ts"
# What production pi calls Ornith-1.5-35B-A3B (~/.pi/agent/models.json, provider local-ornith35,
# :8088). The bench agent copies that entry's thinkingLevelMap (off -> off, medium -> medium) and
# token limits, and points its baseUrl at the bench port. Thinking stays OFF for anchors and the
# gate: pi-job and pi-signal drive Ornith with thinking off.
PROVIDER, MODEL = "local-ornith35", "ornith-1.5-35b-a3b"


# --------------------------------------------------------------------------- proxy

class LoggingProxy:
    """Forward /<tag>/v1/... to upstream /v1/...; log each JSON request body per tag.

    The response is read whole and returned with a Content-Length (pi's SSE parser does not
    care that the stream arrives in one piece), and also logged, so a run can be reconstructed
    without trusting pi's session file alone."""

    def __init__(self, upstream: str, port: int, log_dir: Path):
        m = re.match(r"https?://([^/:]+):(\d+)", upstream)
        if not m:
            raise ValueError(f"upstream must be http://host:port[/...], got {upstream}")
        self.up_host, self.up_port = m.group(1), int(m.group(2))
        self.port, self.log_dir = port, log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _fwd(self, method: str):
                parts = self.path.split("/", 2)  # ['', tag, rest]
                tag = parts[1] if len(parts) > 2 else "untagged"
                path = "/" + (parts[2] if len(parts) > 2 else "")
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                conn = http.client.HTTPConnection(proxy.up_host, proxy.up_port, timeout=900)
                hdrs = {k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "content-length", "connection")}
                conn.request(method, path, body=body, headers=hdrs)
                r = conn.getresponse()
                data = r.read()
                if body and path.endswith("/chat/completions"):
                    try:
                        req = json.loads(body)
                    except ValueError:
                        req = None
                    rec = {"ts": time.time(), "request": req,
                           "status": r.status, "response": data.decode("utf-8", "replace")}
                    with proxy._lock, (proxy.log_dir / f"{tag}.jsonl").open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                conn.close()

            def do_POST(self):
                self._fwd("POST")

            def do_GET(self):
                self._fwd("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


# --------------------------------------------------------------------------- agent dir

def make_agent_dir(dest: Path, base_url: str, provider: str, model_id: str,
                   template_provider: str = PROVIDER) -> Path:
    """Scratch PI_CODING_AGENT_DIR from ~/.pi/agent.

    Kept: skills/ (copied, so pi lists every installed skill exactly as it does in
    production; the SKILL.md files still point at ~/.pi/agent/skills/<x>/scripts, which is
    where the scripts really are), prompts/, settings.json (minus `packages`), and a
    models.json with ONE provider/model pointed at base_url.

    Dropped, on purpose:
      - extensions/: `judge` appends every message to ~/.local/share/pi-judge/decisions.jsonl
        (meant to become fine-tuning data; a bench must not pollute it) and can route to a
        paid provider; `camofox-browser` reaches the open web; `exec-guardrail` sends a
        Signal message on every block. data/pi_bench_guard.ts (stricter, silent, local) is
        loaded instead via -e.
      - settings `packages` (pify memory, session-search, pi-condense, ...): memory and
        session search would read the operator's real memory and Signal transcripts into a
        training row. With -ne they do not load anyway; removing them keeps that explicit.
      - memory/, sessions/, auth.json: private, and not needed by a local provider.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(REAL_AGENT / "skills", dest / "skills", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    if (REAL_AGENT / "prompts").exists():
        shutil.copytree(REAL_AGENT / "prompts", dest / "prompts")
    settings = json.loads((REAL_AGENT / "settings.json").read_text())
    settings.pop("packages", None)
    settings.update({"defaultProvider": provider, "defaultModel": model_id})
    (dest / "settings.json").write_text(json.dumps(settings, indent=2))
    real_models = json.loads((REAL_AGENT / "models.json").read_text())
    tmpl = real_models["providers"].get(template_provider)
    if tmpl is None:
        raise SystemExit(f"~/.pi/agent/models.json has no provider {template_provider!r} to copy")
    prov = {k: v for k, v in tmpl.items() if k != "models"}
    prov["baseUrl"] = base_url
    m0 = dict(next((m for m in tmpl["models"] if m["id"] == MODEL), tmpl["models"][0]))
    m0.update({"id": model_id, "name": f"{model_id} (bench)"})
    prov["models"] = [m0]
    (dest / "models.json").write_text(json.dumps({"providers": {provider: prov}}, indent=2))
    return dest


WORK_ROOT = Path(os.environ.get("CAVEOWL_BENCH_ROOT", "/tmp/cavegemma-skill-bench/owl"))


def scratch_cwd(*parts: str) -> Path:
    """A fresh, empty run cwd OUTSIDE any project. Learned the hard way (2026-10-12): a gate
    run's cwd sat inside this repo, the model ran `uv run python -c ...`, and uv found this
    repo's pyproject.toml and synced .venv from uv.lock (CUDA torch 2.12 over the ROCm .pth,
    transformers 5.17 -> 5.8.1). Session files are copied back into the repo by the callers."""
    d = WORK_ROOT.joinpath(*parts)
    for parent in [d, *d.parents]:
        if any((parent / m).exists() for m in ("pyproject.toml", "uv.lock", ".git")):
            raise SystemExit(f"bench cwd {d} is inside a project ({parent}); "
                             "set CAVEOWL_BENCH_ROOT elsewhere")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run_pi(agent_dir: Path, work: Path, prompt: str, provider: str, model_id: str,
           timeout: int = 600, thinking: str = "off", context_files: bool = False) -> dict:
    """One non-interactive pi run. Returns {session: path|None, rc, seconds, stderr_tail}."""
    sess = work.parent / (work.name + ".sessions")  # outside work: `find .` must not see it
    sess.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent_dir), "PI_OFFLINE": "1",
           # Belt and braces next to scratch_cwd() and the guard: no package manager may
           # resolve, download or sync anything from a bench run.
           "UV_NO_SYNC": "1", "UV_OFFLINE": "1", "PIP_NO_INDEX": "1"}
    for k in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX"):
        env.pop(k, None)
    # Never hand a bench run the operator's provider keys.
    for k in list(env):
        if k.endswith("_API_KEY") or k in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"):
            env.pop(k)
    cmd = ["pi", "-p", "-ne", "-e", str(GUARD_EXT),
           *([] if context_files else ["--no-context-files"]),
           "--provider", provider, "--model", model_id, "--thinking", thinking,
           "--session-dir", str(sess), prompt]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL,
                           timeout=timeout)
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        rc, err = "timeout", ""
    files = sorted(sess.glob("*.jsonl"))
    return {"session": str(files[-1]) if files else None, "rc": rc,
            "seconds": round(time.time() - t0, 1), "stderr_tail": (err or "")[-400:]}


# --------------------------------------------------------------------------- session parse

def load_session(path: str | Path) -> list[dict]:
    """The session's message entries (role user/assistant/toolResult/system), in order."""
    out = []
    for ln in Path(path).read_text().splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get("type") == "message":
            out.append(d["message"])
    return out


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if c.get("type") == "text")


def tool_calls(msgs: list[dict]) -> list[dict]:
    return [c for m in msgs if m.get("role") == "assistant"
            for c in m.get("content") or [] if c.get("type") == "toolCall"]


NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")
YEAR = re.compile(r"^20\d\d$")


def judge_session(msgs: list[dict], spec: dict) -> tuple[bool, list[str]]:
    """Return (kept, reasons). spec keys:
         skill        skill name the prompt targets (None for generic tasks)
         script_re    regex a bash command must match to count as "ran the skill"
         expect       list of strings; the final answer must contain at least one
                      (case-insensitive). Generic tasks know their answer by construction.
         expect_all   list of strings; the final answer must contain every one
         must_read_re regex a read path or bash command must match (the task's input file)
         ground       "number": a number in the final answer must appear in a tool result
         must_bash_re regex a bash command must match (e.g. running the created file)
    """
    reasons: list[str] = []
    asst = [m for m in msgs if m.get("role") == "assistant"]
    results = [m for m in msgs if m.get("role") == "toolResult"]
    calls = tool_calls(msgs)
    if not asst:
        return False, ["no_assistant"]
    if any(m.get("stopReason") == "length" for m in asst):
        reasons.append("length_capped")
    if any(m.get("stopReason") in ("error", "aborted") for m in asst):
        reasons.append("turn_error")
    final = asst[-1]
    final_text = _text(final.get("content")).strip()
    if not final_text or any(c.get("type") == "toolCall" for c in final.get("content") or []):
        reasons.append("no_final_answer")
    tool_out = "\n".join(_text(r.get("content")) for r in results)
    bash_cmds = [str((c.get("arguments") or {}).get("command", "")) for c in calls
                 if c.get("name") == "bash"]
    if any("bench-guard" in _text(r.get("content")) for r in results):
        reasons.append("guard_block")

    read_paths = [str((c.get("arguments") or {}).get("path", "")) for c in calls
                  if c.get("name") == "read"]
    # Only a SKILL.md read (or cat) that SUCCEEDED counts as reading the skill: a guessed path
    # that returned ENOENT used to count, which inflated pdf-explore (stock 9/9 -> 7/9) and let
    # rows that never opened the skill through. Fix 2026-10-13 (run-1 gate review).
    ok_ids = {r.get("toolCallId") for r in results if not r.get("isError")}
    ok_reads = [str((c.get("arguments") or {}).get("path", "")) for c in calls
                if c.get("name") == "read" and c.get("id") in ok_ids]
    ok_bash = [str((c.get("arguments") or {}).get("command", "")) for c in calls
               if c.get("name") == "bash" and c.get("id") in ok_ids]
    skills = spec.get("skill")
    if skills:
        skills = [skills] if isinstance(skills, str) else skills
        if not any(p.endswith(f"{s}/SKILL.md") for p in ok_reads for s in skills) and \
                not any(f"{s}/SKILL.md" in b for b in ok_bash for s in skills):
            reasons.append("skill_not_read")
    if spec.get("must_read_re"):
        rx = spec["must_read_re"]
        if not (any(re.search(rx, p) for p in read_paths) or any(re.search(rx, b) for b in bash_cmds)):
            reasons.append("file_not_read")
    if spec.get("script_re") and not any(re.search(spec["script_re"], b) for b in bash_cmds):
        reasons.append("script_not_run")
    if spec.get("must_bash_re") and not any(re.search(spec["must_bash_re"], b) for b in bash_cmds):
        reasons.append("command_not_run")

    # (d) a tool error must be followed by at least one later successful tool result.
    last_err = max((i for i, r in enumerate(results) if r.get("isError")), default=-1)
    if last_err >= 0 and not any(not r.get("isError") for r in results[last_err + 1:]):
        reasons.append("unaddressed_tool_error")

    # (b) invented output: a fenced block in the model's prose that is neither a tool argument
    # nor present in any tool result, in a run that is supposed to report command output.
    call_args = "\n".join(json.dumps(c.get("arguments") or {}) for c in calls)
    for m in asst:
        for block in re.findall(r"```[^\n]*\n(.*?)```", _text(m.get("content")), re.S):
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            unseen = [ln for ln in lines if ln not in tool_out and json.dumps(ln)[1:-1] not in call_args]
            if len(unseen) > len(lines) / 2 and spec.get("strict_fences", True):
                reasons.append("invented_output")
                break
        else:
            continue
        break
    if not calls:
        reasons.append("no_tool_calls")

    if final_text:
        exp = spec.get("expect") or []
        if exp and not any(e.lower() in final_text.lower() for e in exp):
            reasons.append("answer_not_grounded")
        if spec.get("expect_all") and not all(e.lower() in final_text.lower()
                                              for e in spec["expect_all"]):
            reasons.append("answer_not_grounded")
        if spec.get("ground") == "number":
            nums = set(NUM.findall(final_text))
            seen = set(NUM.findall(tool_out))
            # Two-digit-or-more numbers (or decimals), not bare years: "1", "3" and "2026"
            # ground nothing.
            if not any((len(n) >= 2 or "." in n) and not YEAR.match(n) and n in seen
                       for n in nums):
                reasons.append("answer_not_grounded")
    return (not reasons), sorted(set(reasons))


# --------------------------------------------------------------------------- conversion

def last_request(log_path: Path) -> dict | None:
    """The final chat/completions request pi sent in this run (it carries the whole history)."""
    if not log_path.exists():
        return None
    last = None
    for ln in log_path.read_text().splitlines():
        rec = json.loads(ln)
        if rec.get("request") and rec["request"].get("messages"):
            last = rec["request"]
    return last


# --------------------------------------------------------------------------- fixtures
#
# Generic multi-step tasks run against a small seeded project tree, so every task knows its
# answer by construction and every prompt sees different numbers (nothing to memorise).

FIRST = ["Ayesha", "Bongani", "Chen", "Dmitri", "Elena", "Femi", "Grace", "Hiro", "Ines",
         "Jabu", "Kofi", "Lerato", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa",
         "Sipho", "Tamsin", "Uma", "Victor", "Wen", "Xolani", "Yusuf", "Zanele"]
ERRORS = ["connection reset by peer", "timeout waiting for upstream", "disk quota exceeded",
          "invalid token signature", "database is locked", "certificate has expired"]
SERVICES = ["billing", "search", "auth", "reports", "mailer", "ingest"]


def write_pdf(path: Path, lines: list[str]) -> None:
    """Minimal one-page PDF (Helvetica, one text line per entry) that pdftotext reads."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    ops = ["BT", "/F1 11 Tf", "14 TL", "56 780 Td"]
    for ln in lines:
        ops.append(f"({esc(ln)}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1", "replace")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


def make_fixture(work: Path, seed: int) -> dict:
    """Populate `work` with a seeded mini project; return the facts the tasks ask about."""
    import random
    rng = random.Random(seed)
    f: dict = {}
    (work / "src" / "app").mkdir(parents=True, exist_ok=True)
    (work / "data").mkdir(exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    (work / "notes").mkdir(exist_ok=True)
    (work / "config").mkdir(exist_ok=True)

    port = rng.choice([8433, 9120, 7002, 5150, 8890, 6060, 9443])
    max_clients = rng.choice([16, 32, 48, 64, 96, 128])
    f.update(port=port, max_clients=max_clients)
    (work / "src/app/__init__.py").write_text("")
    (work / "src/app/config.py").write_text(
        f'"""Configuration loading."""\nimport json\n\nDEFAULT_PORT = {port}\n\n\n'
        "def parse_config(path):\n    \"\"\"Read a JSON config; fill in defaults.\"\"\"\n"
        "    with open(path) as fh:\n        cfg = json.load(fh)\n"
        "    cfg.setdefault(\"port\", DEFAULT_PORT)\n    return cfg\n")
    todo1 = rng.choice(["handle SIGTERM gracefully", "add request logging",
                        "move secrets out of config", "support IPv6 listeners"])
    todo2 = rng.choice(["cache parsed configs", "add a health endpoint",
                        "rate-limit failed logins", "write tests for shutdown"])
    f.update(todos=[todo1, todo2])
    (work / "src/app/server.py").write_text(
        "from .config import parse_config\nfrom .utils import retry\n\n"
        f"MAX_CLIENTS = {max_clients}\n\n\nclass Server:\n    # TODO: {todo1}\n"
        "    def __init__(self, cfg_path):\n        self.cfg = parse_config(cfg_path)\n\n"
        "    def start(self):\n        # TODO: " + todo2 + "\n"
        "        return retry(lambda: self.cfg[\"port\"], attempts=3)\n")
    (work / "src/app/utils.py").write_text(
        "import re\nimport time\n\n\ndef slugify(text):\n"
        "    return re.sub(r\"[^a-z0-9]+\", \"-\", text.lower()).strip(\"-\")\n\n\n"
        "def retry(fn, attempts=3, delay=0.1):\n    for i in range(attempts):\n        try:\n"
        "            return fn()\n        except Exception:\n            if i == attempts - 1:\n"
        "                raise\n            time.sleep(delay)\n")
    n_extra = rng.randint(0, 3)
    for i in range(n_extra):
        (work / f"src/app/plugin_{i}.py").write_text(f"NAME = 'plugin_{i}'\n")
    f["n_py_src"] = 4 + n_extra

    # orders.csv
    customers = rng.sample(FIRST, 6)
    rows, counts, shipped_total = [], {}, 0.0
    n_rows = rng.randint(28, 44)
    for i in range(n_rows):
        c = rng.choices(customers, weights=[6, 3, 3, 2, 2, 1])[0]
        st = rng.choices(["shipped", "failed", "pending"], weights=[6, 2, 2])[0]
        amt = round(rng.uniform(5, 400), 2)
        rows.append(f"{1000 + i},{c},{st},{amt:.2f}")
        counts[c] = counts.get(c, 0) + 1
        if st == "shipped":
            shipped_total += amt
    (work / "data/orders.csv").write_text("order_id,customer,status,amount\n" + "\n".join(rows) + "\n")
    top = max(counts.items(), key=lambda kv: kv[1])
    if list(counts.values()).count(top[1]) > 1:
        # make the top customer unique
        rows.append(f"{1000 + n_rows},{top[0]},pending,10.00")
        (work / "data/orders.csv").write_text("order_id,customer,status,amount\n" + "\n".join(rows) + "\n")
        counts[top[0]] += 1
    f["top_customer"] = max(counts.items(), key=lambda kv: kv[1])[0]
    f["n_failed"] = sum(r.split(",")[2] == "failed" for r in rows)
    f["n_orders"] = len(rows)
    f["shipped_total"] = round(sum(float(r.split(",")[3]) for r in rows if r.split(",")[2] == "shipped"), 2)
    (work / "data/readme.txt").write_text("orders.csv is exported nightly.\n")

    # app.log
    errs = rng.sample(ERRORS, 3)
    weights = [5, 2, 1]
    lines, n_err, err_count = [], 0, {e: 0 for e in errs}
    for i in range(rng.randint(60, 90)):
        lvl = rng.choices(["INFO", "WARN", "ERROR"], weights=[7, 2, 2])[0]
        ts = f"2026-10-0{rng.randint(1, 9)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"
        if lvl == "ERROR":
            e = rng.choices(errs, weights=weights)[0]
            err_count[e] += 1
            n_err += 1
            lines.append(f"{ts} ERROR worker-{rng.randint(1, 4)}: {e}")
        elif lvl == "WARN":
            lines.append(f"{ts} WARN slow request {rng.randint(500, 3000)}ms")
        else:
            lines.append(f"{ts} INFO request ok")
    (work / "logs/app.log").write_text("\n".join(lines) + "\n")
    f["n_error_lines"] = n_err
    top_err = max(err_count.items(), key=lambda kv: kv[1])
    f["top_error"] = top_err[0] if list(err_count.values()).count(top_err[1]) == 1 else None

    # access.log
    ips = [f"10.{rng.randint(0, 9)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}" for _ in range(rng.randint(5, 11))]
    acc, ms = [], []
    for i in range(rng.randint(40, 70)):
        ip = rng.choice(ips)
        t = rng.randint(20, 900)
        ms.append(t)
        acc.append(f'{ip} - - [05/Oct/2026:10:{i % 60:02d}:00] "GET /api/v1/items HTTP/1.1" 200 {t}ms')
    (work / "logs/access.log").write_text("\n".join(acc) + "\n")
    f["n_unique_ips"] = len({a.split()[0] for a in acc})
    f["slowest_ms"] = max(ms)

    # meeting notes
    owner, owner2 = rng.sample(FIRST, 2)
    day = rng.randint(14, 30)
    notes = (f"# Platform sync, 2026-10-0{rng.randint(1, 9)}\n\nAttendees: {owner}, {owner2}, ops\n\n"
             "## Decisions\n- Freeze schema changes until the migration lands.\n"
             f"- Keep the {rng.choice(SERVICES)} service on the old cluster for now.\n\n"
             "## Action items\n"
             f"- {owner}: run the database migration by 2026-10-{day}.\n"
             f"- {owner2}: update the on-call rota.\n")
    (work / "notes/meeting.md").write_text(notes)
    f.update(migration_owner=owner, migration_day=str(day), notes_words=len(notes.split()))

    # settings.json
    svc = {s: {"timeout_s": rng.choice([5, 10, 15, 20, 30, 45, 60, 90]),
               "replicas": rng.randint(1, 6)} for s in SERVICES}
    (work / "config/settings.json").write_text(json.dumps({"services": svc}, indent=2) + "\n")
    f["services"] = svc

    # a buggy script for fix-and-run tasks
    a, b = rng.randint(10, 99), rng.randint(2, 9)
    (work / "calc.py").write_text(
        "def average(values):\n    return sum(values) / len(value)\n\n\n"
        f"print(average([{a}, {a + b}, {a + 2 * b}]))\n")
    f["calc_out"] = f"{(3 * a + 3 * b) / 3:.1f}"

    sizes = {}
    for name in ("orders.csv", "readme.txt"):
        sizes[name] = (work / "data" / name).stat().st_size
    f["largest_data"] = max(sizes.items(), key=lambda kv: kv[1])
    return f
