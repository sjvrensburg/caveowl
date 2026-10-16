"""Build self-distilled SKILL TRAJECTORY anchors: real pi sessions driven against the STOCK
model, kept only if the model followed through, converted to training rows.

caveowl copy of ../cavegemma-e4b/data/build_skill_anchor.py (9dd4267), driven against STOCK
Ornith-1.5-35B-A3B on the bench unit cand-owl (:8110). Changes from the E4B port: provider/model
are production's local-ornith35 / ornith-1.5-35b-a3b (pi_traj.PROVIDER, MODEL; thinkingLevelMap
off/medium, thinking OFF here); rows are rendered and length-checked with Ornith's tokenizer and
the GGUF's chat template through owl_data.tokenize_chat (not Gemma's); outputs land in this
repo's data/out. Prompt set, judge, pruning and scrubbing are unchanged. The docstring below is
the E4B port's and its numbers are E4B's.

Why (v3, 2026-10-12): v2 passed every harness (tools 9/9, code 12/12, no reasoning leak) and was
rolled back the same day. On Signal, asked "What's the weather like today in Gqeberha?", stock
E4B read ~/.pi/agent/skills/weather/SKILL.md, ran its script and answered (2/3); v2 saw the
skill and stopped ("Skill exist. Not loadable now.", 0/3). The harnesses only score the first
tool call, and nothing in v2's data showed a whole multi-step agent loop. These rows are the
same kind of anchor as data/build_anchor.py's MBPP rows: the stock model's own behaviour,
filtered to what verifiably worked, so they pull toward "keep doing what you did", not toward
a third style. The final prose is left as stock wrote it (verbose); see README "Skill
trajectories" for why the share stays modest.

How a trajectory is produced: REAL pi (`pi -p`, non-interactive) with a scratch
PI_CODING_AGENT_DIR (data/pi_traj.py make_agent_dir: real skills, no extensions but
data/pi_bench_guard.ts, no packages, one provider pointed at a logging proxy in front of the
bench server). The proxy records every request, which is the only place pi's tool JSON schemas
and its OpenAI-format messages are visible. Each prompt runs in its own fresh scratch cwd
(generic tasks get a seeded mini project, so their answers are known by construction).

Needs the STOCK model on --upstream (default :8110, the bench port; never :8092):
    systemd-run --user --unit=cand-owl -p MemoryHigh=26G -p MemoryMax=30G scripts/serve_bench.sh
    ~/venvs/unsloth-probe/bin/python data/build_skill_anchor.py --extra-attempts 2   # generate + convert
    ~/venvs/unsloth-probe/bin/python data/build_skill_anchor.py --extra-attempts 2 --convert-only
    systemctl --user stop cand-owl

Run 2 (2026-10-13): data/skill_run2.py adds pdf-explore, humanizer and coffee prompts with
fixtures disjoint from eval/skill_gate_wide.py's, judged with that gate's own judge
(spec["judge"] == "wide"). They are only built with --set run2 (or all), which writes to
data/out/skill_runs_run2/ and data/out/skill_traj_run2.jsonl by default, so run 1's files are
never touched:
    ~/venvs/unsloth-probe/bin/python data/build_skill_anchor.py --set run2 --extra-attempts 2 --max-len 16384
Coffee and humanizer rows cannot fit 4096 tokens (their SKILL.md files alone are ~3.3k and ~7k
tokens); see data/skill_run2.py. Convert prints per-family length percentiles so the training
max_seq_length can be chosen from them.

Safety: prompts were chosen so that every skill they reach is read-only (weather: public
forecast APIs the script already calls; system-health-check: uptime/free/df/ps; coffee: the
tracker's report.py over its SQLite; PDFs and text files are generated locally). No prompt sends
a message, spends money, starts a podcast/deep research, or controls a service, and the bench
guard refuses those commands anyway (a refusal ends the run and the row is rejected).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))
from owl_ports import BENCH_URL, refuse_live  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
import pi_traj as P  # noqa: E402

PROVIDER, MODEL = P.PROVIDER, P.MODEL
PROD_AGENT = str(Path.home() / ".pi" / "agent")
PROD_CWD = str(Path.home() / "scratch")

# --------------------------------------------------------------------------- prompt set

WEATHER_CITIES = [
    "Cape Town", "Johannesburg", "Pretoria", "Bloemfontein", "East London", "Stellenbosch",
    "George", "Knysna", "Pietermaritzburg", "Polokwane", "Kimberley", "Mthatha", "London",
    "Paris", "Berlin", "Madrid", "Rome", "Amsterdam", "Lisbon", "Vienna", "Oslo", "Stockholm",
    "Helsinki", "Dublin", "New York", "Chicago", "Toronto", "Vancouver", "Mexico City",
    "Buenos Aires", "Sao Paulo", "Cairo", "Lagos", "Tokyo", "Kyoto", "Seoul", "Singapore",
    "Sydney", "Auckland", "Mumbai"]
WEATHER_PHRASES = [
    "What's the weather like today in {c}?",
    "Will it rain in {c} tomorrow?",
    "Should I take an umbrella in {c} today?",
    "weather {c}",
    "How hot will it get in {c} over the next few days?",
    "What's the forecast for {c} for the next 3 days?",
    "Is it going to be windy in {c} today?",
    "What's the temperature in {c} right now?",
    "Do I need a jacket in {c} this evening?",
    "{c} weather this week?",
]
HEALTH_PROMPTS = [
    "How's the box doing?", "Run a health check on this machine.",
    "Is the disk filling up?", "How much memory is free right now?",
    "What's the load average like?", "Check CPU, memory and disk for me.",
    "Is anything hogging the CPU?", "Which processes use the most memory?",
    "Quick system health report please.", "Is this host healthy?",
    "How much disk space is left on /?", "Any resource problems on the box right now?",
    "How long has the machine been up, and how busy is it?",
    "Give me a health summary of this server.", "Is the machine swapping or short on RAM?",
    "Check the box's health and tell me if anything needs attention.",
    "How many CPU cores does this machine have and how loaded are they?",
    "Are the network interfaces up?", "Health check: memory and disk only.",
    "Status of this machine?",
]
COFFEE_PROMPTS = [
    "How have Jacobs Kronung prices moved over the last 30 days?",
    "What does the coffee price tracker's history say for the last 90 days?",
    "Which retailer has been cheapest for Jacobs Kronung lately, according to the tracker?",
    "Which retailer's Kronung prices are the most volatile?",
    "Summarise the coffee tracker report for the past two weeks.",
    "From the tracked history, what's the usual Pick n Pay price for the 200g Kronung?",
    "How many Kronung specials did the tracker record in the last 30 days?",
    "What's the Takealot price range for Jacobs Kronung over the last month, per the tracker?",
    "Coffee tracker: how has the 95g jar been priced recently?",
    "Read the coffee price history and tell me where Kronung is cheapest per the last 60 days.",
]

PDF_DOCS = {
    "invoice": lambda r: (["TAX INVOICE", f"Invoice number: INV-{r.randint(1000, 9999)}",
                           "Supplier: Bay Office Supplies (Pty) Ltd",
                           f"Total due: R {r.randint(1, 9)},{r.randint(100, 999)}.{r.randint(10, 99)}",
                           f"Due date: 2026-11-{r.randint(10, 28)}",
                           "Payment reference: use the invoice number"],
                          [("What's the total due on invoice.pdf?", 3),
                           ("When is invoice.pdf due?", 4),
                           ("What's the invoice number in invoice.pdf?", 1),
                           ("Pull the text out of invoice.pdf and tell me who the supplier is and the total.", 3)]),
    "lab": lambda r: (["LABORATORY REPORT", f"Sample ID: S-{r.randint(10000, 99999)}",
                       f"Collected: 2026-09-{r.randint(10, 30)}",
                       f"Nitrate (mg/L): {r.randint(2, 48)}.{r.randint(0, 9)}",
                       f"pH: {r.randint(6, 8)}.{r.randint(0, 9)}",
                       "Comment: within municipal limits"],
                      [("What nitrate level does lab_report.pdf give?", 3),
                       ("What's the sample ID in lab_report.pdf?", 1),
                       ("Read lab_report.pdf: what pH was measured?", 4)]),
    "lease": lambda r: (["RESIDENTIAL LEASE - SCHEDULE", f"Monthly rent: R {r.randint(6, 19)} {r.randint(100, 999)}",
                         f"Lease start: 2026-{r.randint(1, 12):02d}-01",
                         f"Deposit: R {r.randint(10, 30)} 000",
                         f"Notice period: {r.choice([1, 2, 3])} calendar months"],
                        [("How much is the monthly rent in lease.pdf?", 1),
                         ("What notice period does lease.pdf specify?", 4),
                         ("Extract lease.pdf and tell me the deposit.", 3)]),
    "agenda": lambda r: (["WORKSHOP AGENDA", f"Keynote: {r.randint(8, 10)}:{r.choice(['00', '15', '30', '45'])} - Dr {r.choice(P.FIRST)} Ndlovu",
                          f"Room: Lecture Hall {r.randint(1, 12)}",
                          f"Lunch: 12:{r.choice(['00', '30'])}",
                          f"Registration closes: 2026-10-{r.randint(15, 29)}"],
                         [("Who gives the keynote in agenda.pdf, and when?", 1),
                          ("Which room is the workshop in, per agenda.pdf?", 2),
                          ("When does registration close according to agenda.pdf?", 4)]),
}

ARTICLES = [
    ("notes/article.md", "# Solar uptake in small municipalities\n\n"
     "A survey of {n} municipalities found that rooftop solar capacity grew by {p}% in the year to "
     "June. The largest single installation was {k} kW at a water treatment works. Officials said "
     "the main obstacle was not cost but the time taken to approve grid connections, which "
     "averaged {d} days. The report recommends a single online application form and a fixed "
     "approval deadline. Several councils also asked for clearer rules on feed-in tariffs, which "
     "currently differ from one town to the next. The authors expect growth to slow next year as "
     "the easiest rooftops are already covered.\n", "p"),
    ("report.txt", "Quarterly operations report\n\nThe support desk closed {n} tickets this "
     "quarter, {p}% of them within one working day. Median first response time was {d} minutes. "
     "The largest category was password resets. Two outages occurred, the longer lasting {k} "
     "minutes after a certificate expired on the load balancer. Actions: automate certificate "
     "renewal, add a status page, and retire the legacy VPN by year end.\n", "n"),
    ("notes/postmortem.md", "# Postmortem: billing delays\n\nOn 3 October invoices were sent "
     "{d} hours late. Root cause: a queue worker ran out of memory after a dependency upgrade "
     "doubled its cache size. {n} customers were affected; none were double-charged. The fix "
     "capped the cache at {k} MB and added an alert when the queue is older than {p} minutes. "
     "Follow-up: pin dependency versions in the worker image.\n", "d"),
    ("notes/field_trip.md", "# Field trip notes\n\nWe counted {n} bird species at the estuary "
     "over two mornings, including {k} flamingos on the second day. Water temperature was "
     "{p} degrees at 07:00. The boardwalk is closed for repairs until mid-November, so access "
     "was from the northern car park. Next visit: bring a spotting scope and arrive before "
     "sunrise; the tide table shows low water at {d}:00.\n", "k"),
]
SUMMARY_PHRASES = ["Summarize {f} in three bullet points.", "tl;dr {f}",
                   "What's {f} about?", "Give me a short summary of {f}."]


def generic_train(f: dict) -> list[tuple[str, str, dict]]:
    """(template id, prompt, spec) for the TRAIN generic tasks. eval/skill_gate.py has its own,
    disjoint templates."""
    import math
    N = 30 + (f["port"] % 50)
    n_primes = sum(all(k % d for d in range(2, int(math.isqrt(k)) + 1)) for k in range(2, N))
    fibn = 10 + (f["max_clients"] % 15)
    a, b = 1, 1
    for _ in range(fibn - 2):
        a, b = b, a + b
    fib = b if fibn > 2 else 1
    svc = sorted(f["services"])[f["port"] % len(f["services"])]
    st = f["shipped_total"]
    out = [
        ("orders_failed", "How many orders in data/orders.csv have status failed?",
         {"expect": [str(f["n_failed"])], "must_read_re": r"orders\.csv"}),
        ("orders_failed2", "Count the failed orders in data/orders.csv.",
         {"expect": [str(f["n_failed"])], "must_read_re": r"orders\.csv"}),
        ("orders_top", "Which customer has the most orders in data/orders.csv?",
         {"expect": [f["top_customer"]], "must_read_re": r"orders\.csv"}),
        ("orders_total", "What's the total amount of all shipped orders in data/orders.csv?",
         {"expect": [f"{st:.2f}", f"{st:,.2f}", f"{st:,.2f}".replace(",", " ")],
          "must_read_re": r"orders\.csv"}),
        ("log_errors", "How many ERROR lines are in logs/app.log?",
         {"expect": [str(f["n_error_lines"])], "must_read_re": r"app\.log"}),
        ("port", "What default port does the app under src/ use, and which file defines it?",
         {"expect_all": [str(f["port"]), "config.py"]}),
        ("todos", "List the TODO comments in src/.", {"expect_all": f["todos"]}),
        ("primes", f"Create primes.py that prints all primes below {N}, run it, and tell me how many there are.",
         {"expect": [str(n_primes)], "must_bash_re": r"primes\.py"}),
        ("fib", f"Write fib.py that prints the {fibn}th Fibonacci number (fib(1) = fib(2) = 1), run it and tell me the result.",
         {"expect": [str(fib)], "must_bash_re": r"fib\.py"}),
        ("wordcount", "Write a Python script that counts the words in notes/meeting.md (split on whitespace), run it, and report the count.",
         {"expect": [str(f["notes_words"])], "must_bash_re": r"python"}),
        ("calc_fix", "Run calc.py. If it fails, fix it, run it again and tell me what it prints.",
         {"expect": [f["calc_out"]], "must_bash_re": r"calc\.py"}),
        ("meeting_owner", "In notes/meeting.md, who owns the database migration and by when?",
         {"expect_all": [f["migration_owner"], f["migration_day"]]}),
        ("svc_timeout", f"What timeout is configured for the {svc} service in config/settings.json?",
         {"expect": [str(f["services"][svc]["timeout_s"])], "must_read_re": r"settings\.json"}),
        ("count_py", "Write a bash script count_py.sh that counts the .py files under src/, run it, and tell me the number.",
         {"expect": [str(f["n_py_src"])], "must_bash_re": r"count_py"}),
        ("retry_caller", "Which file in src/ calls retry(), and how many attempts does it pass?",
         {"expect_all": ["server.py", "3"]}),
        ("replicas", "What's the total number of replicas across all services in config/settings.json?",
         {"expect": [str(sum(v["replicas"] for v in f["services"].values()))],
          "must_read_re": r"settings\.json"}),
    ]
    if f["top_error"]:
        out.append(("log_top", "What's the most frequent error message in logs/app.log?",
                    {"expect": [f["top_error"]], "must_read_re": r"app\.log"}))
    return out


def build_prompts(sets: tuple[str, ...] = ("run1",)) -> list[dict]:
    """Every training prompt of the given sets: {id, family, prompt, spec, fixture}.
    "run1" is the original prompt set; "run2" is data/skill_run2.py's."""
    run1 = _build_run1()
    items = run1 if "run1" in sets else []
    if "run4" in sets:  # data/skill_run4.py: more pdf (ocr-and-documents) rows
        import skill_run2
        import skill_run4
        r4 = skill_run4.items()
        skill_run2.check_disjoint(r4, run1)
        items = items + r4
    if "run2" in sets:
        import skill_run2
        r2 = skill_run2.items()
        skill_run2.check_disjoint(r2, run1)
        items = items + r2
    return items


def _build_run1() -> list[dict]:
    from skill_gate import GATE_WEATHER, GATE_GENERIC_IDS
    gate_cities = {c.lower() for c, _ in GATE_WEATHER}
    items: list[dict] = []
    # weather: every city once, every other city a second phrasing.
    for i, c in enumerate(WEATHER_CITIES):
        assert c.lower() not in gate_cities, c
        for j in ([i % 10, (i + 5) % 10] if i % 2 == 0 else [i % 10]):
            items.append({"id": f"wx-{i:02d}-{j}", "family": "weather",
                          "prompt": WEATHER_PHRASES[j].format(c=c),
                          "spec": {"skill": "weather", "script_re": r"weather\.py",
                                   "ground": "number"}})
    for i, p in enumerate(HEALTH_PROMPTS):
        items.append({"id": f"health-{i:02d}", "family": "system-health", "prompt": p,
                      "spec": {"skill": "system-health-check", "ground": "number",
                               "script_re": r"\b(uptime|free|df|nproc|lscpu|ps|ip)\b"}})
    for i, p in enumerate(COFFEE_PROMPTS):
        items.append({"id": f"coffee-{i:02d}", "family": "coffee", "prompt": p,
                      "spec": {"skill": "coffee-price-tracker", "script_re": r"report\.py",
                               "ground": "number"}})
    for name, mk in PDF_DOCS.items():
        for k, seed in enumerate((11, 23, 37, 41)):
            lines, qs = mk(random.Random(f"{name}-{seed}"))
            q, li = qs[k % len(qs)]
            fname = {"invoice": "invoice.pdf", "lab": "lab_report.pdf", "lease": "lease.pdf",
                     "agenda": "agenda.pdf"}[name]
            # The answer is the value after the line's first colon (or the line itself).
            val = lines[li].split(":", 1)[1].strip() if ":" in lines[li] else lines[li]
            val = val.split(" - ")[0] if name == "agenda" and li == 1 else val
            items.append({"id": f"pdf-{name}-{k}", "family": "pdf", "prompt": q,
                          "fixture": {"pdf": fname, "lines": lines},
                          "spec": {"skill": ["ocr-and-documents", "summarize", "pdf-explore"],
                                   "script_re": r"pdftotext|extract_pymupdf|fitz",
                                   "expect": [val, val.replace(" ", ""), val.replace(",", "")]}})
    for a, (path, text, key) in enumerate(ARTICLES):
        for k, phrase in enumerate(SUMMARY_PHRASES[: 4 if a < 2 else 3]):
            r = random.Random(f"art-{a}-{k}")
            vals = {"n": r.randint(12, 480), "p": r.randint(11, 64), "k": r.randint(14, 900),
                    "d": r.randint(3, 59)}
            items.append({"id": f"sum-{a}-{k}", "family": "summarize",
                          "prompt": phrase.format(f=path),
                          "fixture": {"file": path, "text": text.format(**vals)},
                          "spec": {"skill": "summarize", "must_read_re": re.escape(path.split("/")[-1]),
                                   "expect": [str(vals[key])]}})
    for s in range(4):
        seed = 1000 + s  # facts only here; the fixture is rebuilt in each run's own cwd
        for tid, prompt, spec in generic_train(_facts(seed)):
            if tid in GATE_GENERIC_IDS:
                raise SystemExit(f"train template {tid} collides with a gate template")
            items.append({"id": f"gen-{tid}-{s}", "family": "generic", "prompt": prompt,
                          "fixture": {"seed": seed}, "spec": spec})
    return items


_FACT_CACHE: dict[int, dict] = {}


def _facts(seed: int) -> dict:
    """make_fixture's facts for a seed, computed once in a throwaway dir."""
    if seed not in _FACT_CACHE:
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="skillfx-"))
        try:
            _FACT_CACHE[seed] = P.make_fixture(d, seed)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return _FACT_CACHE[seed]


def setup_work(work: Path, item: dict) -> None:
    fx = item.get("fixture") or {}
    if "seed" in fx:
        P.make_fixture(work, fx["seed"])
    if "pdf" in fx:
        P.write_pdf(work / fx["pdf"], fx["lines"])
    if "file" in fx:
        (work / fx["file"]).parent.mkdir(parents=True, exist_ok=True)
        (work / fx["file"]).write_text(fx["text"])
    if "files" in fx or "pdfs" in fx:  # run 2: skill_gate_wide's fixture shape
        import skill_gate_wide as W
        W.setup_work(work, fx)


def judge_item(msgs: list[dict], spec: dict) -> tuple[bool, list[str]]:
    """Run-2 items carry spec["judge"] = "wide": the wide gate's judge (exact-value grounding,
    numbers surviving a rewrite, coffee's ground_alt and script_failed) on top of
    judge_session. Everything else keeps judge_session, as in run 1."""
    if spec.get("judge") == "wide":
        import skill_gate_wide as W
        return W.judge(msgs, {k: v for k, v in spec.items() if k != "judge"})
    return P.judge_session(msgs, spec)


# --------------------------------------------------------------------------- generate

def generate(items: list[dict], run_dir: Path, upstream: str, proxy_port: int, workers: int,
             force: bool) -> None:
    plog = run_dir / "_proxy"
    if plog.exists():
        shutil.rmtree(plog)
    todo = [it for it in items if force or not (run_dir / it["id"] / "meta.json").exists()]
    print(f">>> generate: {len(todo)} of {len(items)} prompts to run, {workers} workers")
    lock = threading.Lock()
    done = [0]
    with P.LoggingProxy(upstream, proxy_port, plog):
        # Agent dirs live beside the cwds, on a short path: pi lists every skill's location
        # in the system prompt, and a long repo path costs ~1k tokens and is one more thing
        # unlike production's ~/.pi/agent/skills (to_row rewrites it back to that anyway).
        agents = [P.make_agent_dir(P.WORK_ROOT / f"agent-build-w{w}",
                                   f"http://127.0.0.1:{proxy_port}/w{w}/v1", PROVIDER, MODEL)
                  for w in range(workers)]
        slots = list(range(workers))

        def one(it: dict) -> None:
            with lock:
                w = slots.pop()
            try:
                d = run_dir / it["id"]
                d.mkdir(parents=True, exist_ok=True)
                work = P.scratch_cwd("build", it["id"])
                setup_work(work, it)
                (plog / f"w{w}.jsonl").unlink(missing_ok=True)
                res = P.run_pi(agents[w], work, it["prompt"], PROVIDER, MODEL)
                src = plog / f"w{w}.jsonl"
                if src.exists():
                    shutil.move(src, d / "requests.jsonl")
                if res["session"]:  # persist it: the cwd lives under /tmp
                    shutil.copy(res["session"], d / "session.jsonl")
                    res["session"] = str(d / "session.jsonl")
                meta = {**it, "run": res, "agent_dir": str(agents[w]), "work": str(work)}
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
                with lock:
                    done[0] += 1
                    print(f"  [{done[0]}/{len(todo)}] {it['id']}  rc={res['rc']}  "
                          f"{res['seconds']}s", flush=True)
            finally:
                with lock:
                    slots.append(w)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))


# --------------------------------------------------------------------------- convert

MAC = re.compile(r"\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b", re.I)
PRIVATE_IP = re.compile(r"\b(?:192\.168|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}\b")


def scrub(s: str, agent_dir: str, work: str) -> str:
    """Production paths in place of scratch ones; MACs and LAN/tailnet IPs redacted (the
    system-health rows run `ip -s link`, and docs/06 in halo-prep is the only place the
    operator wants their topology)."""
    s = s.replace(agent_dir, PROD_AGENT).replace(work, PROD_CWD)
    s = MAC.sub("02:00:00:00:00:00", s)
    return PRIVATE_IP.sub("192.0.2.1", s)


def trim_system_prompt(sp: str, keep: set[str]) -> str:
    """pi's real system prompt with the <docs> block dropped and <available_skills> cut to
    `keep` (same order, same markup, descriptions untouched). See README "Skill trajectories"
    for the token arithmetic behind this."""
    sp = re.sub(r"\n<docs>.*?</docs>\n", "\n", sp, flags=re.S)

    def keep_skill(m: re.Match) -> str:
        name = re.search(r"<name>(.*?)</name>", m.group(0)).group(1)
        return m.group(0) if name in keep else ""
    sp = re.sub(r"  <skill>\n.*?</skill>\n", keep_skill, sp, flags=re.S)
    return sp


def skill_names(sp: str) -> list[tuple[str, int]]:
    return [(m.group(1), len(m.group(2))) for m in
            re.finditer(r"<name>(.*?)</name>\s*<description>(.*?)</description>", sp, re.S)]


REFERS_TO_FAILURE = re.compile(
    r"\b(fail\w*|not found|not (a )?recogni[sz]ed|error|did ?n[o']t work|previous (attempt|command)|"
    r"again|instead|ENOENT|no such file|unable|couldn't|could not|cannot execute|can't execute|"
    r"not available|directly)\b", re.I)


GUESS_ERROR = re.compile(r"command not found|ENOENT|No such file or directory", re.I)


def prune(msgs: list[dict], err_ids: set[str], failed_ids: set[str]) -> tuple[list[dict], int, int]:
    """Drop guessed-and-failed tool calls (with their results) and immediate repeats of a
    successful call, so a row never puts loss on a step that went nowhere.

    err_ids: calls whose result was a GUESS error (see GUESS_ERROR); failed_ids: every call
    whose result was an error. Stock E4B's commonest detour is `bash: weather Cape Town` ->
    "command not found" -> read SKILL.md; training on that teaches the detour. A repeat is the
    same tool with the same arguments as the call right before it, which succeeded (stock
    sometimes reads SKILL.md or runs the script twice in a row); an identical call after
    something else happened (a rerun after an edit) is not a repeat. Text that talks about a
    dropped step ("the `weather` command was not found, so I will use the skill") would make
    the row contradict itself: in an intermediate message (one that also makes calls) that
    text is blanked and the calls kept; in the final answer the row is rejected
    (ValueError `prune_inconsistent`). An assistant message whose calls are all dropped goes
    too, including any intent line in front of them, because that text announced the step.
    Returns (messages, n_failed_dropped, n_repeats_dropped)."""
    drop_ids: set[str] = set()
    n_err = n_dup = 0
    prev_sig, prev_ok = None, False
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            sig = tc["function"]["name"] + json.dumps(tc["function"]["arguments"], sort_keys=True)
            if tc["id"] in err_ids:
                drop_ids.add(tc["id"])
                n_err += 1
                continue
            if sig == prev_sig and prev_ok:
                drop_ids.add(tc["id"])
                n_dup += 1
                continue
            prev_sig, prev_ok = sig, tc["id"] not in failed_ids
    if not drop_ids:
        return msgs, 0, 0
    out, pending_check, n_narr = [], False, [0]
    for m in msgs:
        if m["role"] == "tool" and m["tool_call_id"] in drop_ids:
            continue
        if m["role"] == "assistant":
            calls = m.get("tool_calls") or []
            if pending_check and m.get("content") and REFERS_TO_FAILURE.search(m["content"]):
                if not calls:
                    # The final answer talks about the dropped step: cannot be made consistent.
                    raise ValueError("prune_inconsistent")
                # An intermediate narration of the dropped failure ("the `weather` command was
                # not found, I will use the skill") goes; the calls it introduced stay.
                m = {**m, "content": ""}
                n_narr[0] += 1
            kept = [tc for tc in calls if tc["id"] not in drop_ids]
            if calls and len(kept) < len(calls):
                pending_check = True
                if not kept:
                    continue  # the whole step (and the line announcing it) goes
                m = {**m, "tool_calls": kept}
            elif m.get("content"):
                pending_check = False
        out.append(m)
    return out, n_err, n_dup


BENCH_ROOT = "/tmp/cavegemma-skill-bench"


def to_row(meta: dict, req: dict, msgs: list[dict], n_distract: int) -> dict:
    agent_dir, work = meta["agent_dir"], meta["work"]
    sc = lambda s: scrub(s, agent_dir, work)  # noqa: E731
    sp = req["messages"][0]["content"]
    assert req["messages"][0]["role"] == "system"
    target = meta["spec"].get("skill") or []
    target = {target} if isinstance(target, str) else set(target)
    # A read SKILL.md must stay listed, whichever of the accepted skills was used.
    for c in P.tool_calls(msgs):
        m = re.search(r"skills/([^/]+)/SKILL\.md", json.dumps(c.get("arguments") or {}))
        if m:
            target.add(m.group(1))
    pool = [n for n, ln in skill_names(sp) if n not in target and ln <= 400]
    pool.sort(key=lambda n: sha1(f"{meta['id']}:{n}".encode()).hexdigest())
    keep = target | set(pool[:n_distract])
    out = [{"role": "system", "content": sc(trim_system_prompt(sp, keep))}]
    for m in req["messages"][1:]:
        role = m["role"]
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        content = sc(content or "")
        if role == "assistant":
            nm = {"role": "assistant", "content": content}
            if m.get("tool_calls"):
                nm["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(sc(tc["function"]["arguments"]) or "{}")}}
                    for tc in m["tool_calls"]]
            out.append(nm)
        elif role == "tool":
            # Other runs' scratch copies (stock runs `find / -name x.pdf`, which lists every
            # earlier build cwd under the bench root): drop those lines rather than teach the
            # bench layout. This run's own paths were rewritten by scrub() already.
            content = "\n".join(ln for ln in content.split("\n") if BENCH_ROOT not in ln)
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
        else:
            out.append({"role": role, "content": content})
    final = [m for m in msgs if m.get("role") == "assistant"][-1]
    final_text = sc(P._text(final.get("content")).strip())
    out.append({"role": "assistant", "content": final_text})
    # Only GUESS errors are pruned: a command or path that does not exist. An error from the
    # task itself (a traceback from the script the user asked to run and fix) is information
    # the next step depends on, so it stays; judge_session already requires a later success.
    err_ids = {r["toolCallId"] for r in msgs if r.get("role") == "toolResult" and r.get("isError")
               and GUESS_ERROR.search(P._text(r.get("content")))}
    failed_ids = {r["toolCallId"] for r in msgs if r.get("role") == "toolResult" and r.get("isError")}
    body, n_err, n_dup = prune(out[1:], err_ids, failed_ids)
    out = out[:1] + body
    tools = json.loads(sc(json.dumps(req.get("tools") or [])))
    return {"key": f"skill:{meta['id']}", "origin": f"pi-1.1.0+stock-ornith-35b-a3b:{meta['family']}:{meta['id']}",
            "license": "self-distilled", "category": "skill_traj", "family": meta["family"],
            "tools": tools, "messages": out, "pruned": {"failed_calls": n_err, "repeats": n_dup}}


def row_checks(row: dict, spec: dict) -> list[str]:
    """Checks on the finished row, after pruning.

    skill_read_pruned: judge_session counts a FAILED read of SKILL.md (e.g. a guessed relative
    path -> ENOENT) as "read the skill", and prune() then drops that failed call, leaving a row
    that answers without ever opening the skill: the exact behaviour these rows exist to
    train out (found in run 2: pdfx-8-2-a2). judge_session is shared with the gates and is left
    alone; the row is refused here instead.
    bench_path_leak: a scratch path survived scrub() outside a tool result."""
    reasons = []
    skills = spec.get("skill")
    if skills:
        skills = [skills] if isinstance(skills, str) else skills
        args = [json.dumps(tc["function"]["arguments"]) for m in row["messages"]
                if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
        if not any(f"{sk}/SKILL.md" in a for a in args for sk in skills):
            reasons.append("skill_read_pruned")
    if BENCH_ROOT in json.dumps(row["messages"]):
        reasons.append("bench_path_leak")
    return reasons


def convert(items: list[dict], run_dir: Path, out_path: Path, n_distract: int,
            no_prune: bool, max_len: int, tokenizer: str | None) -> None:
    tok = None
    sys.path.insert(0, str(ROOT))
    import owl_data as D  # has_marker() below runs with or without a tokenizer
    if tokenizer:
        tok, _ = D.load_tokenizer(tokenizer, None, ROOT / "template" / "ornith-gguf-chat-template.jinja")

        def tokenize_tool_chat(tok, row):
            return D.tokenize_chat(tok, row["messages"], row.get("tools"))
    fam_total, fam_kept, reasons_by_fam, draws = Counter(), Counter(), Counter(), Counter()
    reasons_all, lens, full_sys, trim_sys = Counter(), [], [], []
    pruned_stats = Counter()
    fam_lens: dict[str, list[int]] = {}
    kept_base: set[str] = set()
    # Attempt 0 of every prompt first, so extra draws only fill prompts that had no row.
    items = sorted(items, key=lambda i: "base_id" in i)
    rows = []
    for it in items:
        fam = it["family"]
        draws[fam] += 1
        if "base_id" not in it:
            fam_total[fam] += 1
        d = run_dir / it["id"]
        if not (d / "meta.json").exists():
            reasons_all["not_run"] += 1
            reasons_by_fam[(fam, "not_run")] += 1
            continue
        meta = json.loads((d / "meta.json").read_text())
        run = meta["run"]
        rs: list[str] = []
        if not run.get("session"):
            rs = ["no_session" if run["rc"] != "timeout" else "timeout"]
        else:
            msgs = P.load_session(run["session"])
            ok, rs = judge_item(msgs, meta["spec"])
            if run["rc"] == "timeout":
                rs.append("timeout")
            if no_prune and any(m.get("isError") for m in msgs if m.get("role") == "toolResult"):
                rs.append("recovered_error")
            req = P.last_request(d / "requests.jsonl")
            if req is None:
                rs.append("no_request_log")
        if not rs:
            try:
                row = to_row(meta, req, msgs, n_distract)
            except ValueError as e:
                rs.append(str(e))
        if not rs:
            rs += row_checks(row, meta["spec"])
        if not rs:
            if no_prune and any(row["pruned"].values()):
                rs.append("would_need_pruning")
        if not rs:
            blob = json.dumps(row["messages"])
            if D.has_marker(row["messages"]):  # unconditional: a missing tokenizer must not skip it
                rs.append("marker_in_content")
            elif tok is not None:
                ex = tokenize_tool_chat(tok, row)
                if ex is None:
                    rs.append("render_failed")
                else:
                    n = len(ex["input_ids"])
                    full = D.render(tok, [{"role": "system", "content": req["messages"][0]["content"]},
                                          {"role": "user", "content": "x"}], req.get("tools"))
                    full_sys.append(len(tok(full, add_special_tokens=False)["input_ids"]))
                    trimmed = D.render(tok, [row["messages"][0], {"role": "user", "content": "x"}],
                                       row["tools"])
                    trim_sys.append(len(tok(trimmed, add_special_tokens=False)["input_ids"]))
                    lens.append(n)
                    fam_lens.setdefault(fam, []).append(n)
                    if n > max_len:
                        rs.append("too_long")
        base = it.get("base_id", it["id"])
        if not rs and base in kept_base:
            rs.append("prompt_already_kept")
        for r in set(rs):
            reasons_all[r] += 1
            reasons_by_fam[(fam, r)] += 1
        if not rs:
            pruned_stats.update(row["pruned"])
            pruned_stats["kept_rows_pruned"] += any(row["pruned"].values())
            kept_base.add(base)
            fam_kept[fam] += 1
            rows.append(row)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f">>> kept {len(rows)} rows from {sum(fam_total.values())} prompts "
          f"({sum(draws.values())} draws) -> {out_path}")
    for fam in fam_total:
        rej = {r: n for (f2, r), n in reasons_by_fam.items() if f2 == fam}
        print(f"  {fam:14s} kept {fam_kept[fam]:3d} / {fam_total[fam]:3d} prompts "
              f"({draws[fam]} draws)   rejected draws for: {rej}")
    print(f"  rejection reasons per draw (a draw can have several): "
          f"{dict(reasons_all.most_common())}")
    if lens:
        s = sorted(lens)
        print(f"  tokens (trimmed rows, Ornith GGUF template): n={len(s)} median={s[len(s) // 2]} "
              f"p90={s[int(len(s) * .9)]} max={s[-1]}  (limit {max_len})")
    for fam, fl in sorted(fam_lens.items()):
        s = sorted(fl)
        print(f"    {fam:14s} judged-ok draws n={len(s)} min={s[0]} median={s[len(s) // 2]} "
              f"p90={s[int(len(s) * .9)]} max={s[-1]}  over limit: {sum(x > max_len for x in s)}")
    if full_sys:
        s, t = sorted(full_sys), sorted(trim_sys)
        print(f"  system prompt + tool declarations alone: pi's full one median={s[len(s) // 2]} "
              f"max={s[-1]}; trimmed median={t[len(t) // 2]} max={t[-1]} tokens")
    print(f"  pruning (kept rows): {dict(pruned_stats)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=BENCH_URL.removesuffix("/v1"))
    ap.add_argument("--proxy-port", type=int, default=8111)
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel pi runs; the bench server runs --parallel 2")
    ap.add_argument("--set", default="run1", choices=["run1", "run2", "run4", "all"],
                    help="prompt set; run2 = data/skill_run2.py (pdf-explore, humanizer, coffee)")
    ap.add_argument("--families", default="", help="comma list: keep only these families")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="default data/out/skill_runs (run1/all) or skill_runs_run2 (run2)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/out/skill_traj.jsonl (run1/all) or skill_traj_run2.jsonl (run2)")
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run prompts that already have a run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="regex on prompt ids")
    ap.add_argument("--distractors", type=int, default=5,
                    help="non-target skills kept in the trimmed system prompt")
    ap.add_argument("--no-prune", action="store_true",
                    help="reject runs with any tool error or repeated call instead of pruning "
                         "the failed/repeated steps out (see prune())")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizer"))
    ap.add_argument("--list", action="store_true", help="print the prompt set and exit")
    ap.add_argument("--extra-attempts", type=int, default=0,
                    help="extra draws per prompt for the --resample families")
    ap.add_argument("--resample", default=None,
                    help="default system-health,coffee,pdf,summarize (run1/all); "
                         "pdf-explore,humanizer,coffee (run2)")
    args = ap.parse_args()
    refuse_live(args.upstream)
    run2_only = args.set == "run2"
    out_dir = ROOT / "data" / "out"
    if args.set == "run4":
        args.run_dir = args.run_dir or out_dir / "skill_runs_run4"
        args.out = args.out or out_dir / "skill_traj_run4.jsonl"
        args.resample = args.resample or "pdf"
    args.run_dir = args.run_dir or out_dir / ("skill_runs_run2" if run2_only else "skill_runs")
    args.out = args.out or out_dir / ("skill_traj_run2.jsonl" if run2_only else "skill_traj.jsonl")
    if args.resample is None:
        args.resample = ("pdf-explore,humanizer,coffee" if run2_only
                         else "system-health,coffee,pdf,summarize")

    items = build_prompts(("run1", "run2") if args.set == "all" else (args.set,))
    if args.families:
        keep = set(args.families.split(","))
        items = [i for i in items if i["family"] in keep]
    if args.extra_attempts:
        # Rejection sampling, the same thing at temperature 1.0 again: for families where stock
        # rarely completes (it often skips SKILL.md), more draws of the same prompt. Convert
        # keeps at most one row per prompt (the first attempt that passes), so this adds
        # coverage, not duplicates.
        fams = set(args.resample.split(","))
        items += [{**it, "id": f"{it['id']}-a{a}", "base_id": it["id"]}
                  for a in range(1, args.extra_attempts + 1)
                  for it in list(items) if it["family"] in fams and "base_id" not in it]
    if args.only:
        items = [i for i in items if re.search(args.only, i["id"])]
    if args.limit:
        items = items[: args.limit]
    if args.list:
        for it in items:
            print(f"{it['id']:24s} {it['prompt']}")
        print(Counter(i["family"] for i in items), len(items))
        return
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.convert_only:
        generate(items, args.run_dir, args.upstream, args.proxy_port, args.workers, args.force)
    convert(items, args.run_dir, args.out, args.distractors, args.no_prune, args.max_len,
            args.tokenizer if Path(args.tokenizer).exists() else None)


if __name__ == "__main__":
    main()
