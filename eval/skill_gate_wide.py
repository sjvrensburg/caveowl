"""Wide skill follow-through gate: does skill follow-through hold on skills a candidate never

caveowl copy of cavegemma's eval/skill_gate_wide.py (the Gemma-4-E4B port): the pi provider/model are Ornith's
(data/pi_traj.py PROVIDER, MODEL); prompts, judge and pass rule are unchanged. Run stock and
candidate on the same bench flags (scripts/serve_bench.sh), one at a time on :8110.
trained on, and is it at least stock's per skill?

Why this exists: eval/skill_gate.py (weather + generic shell tasks) is exactly the families v3's
skill rows were drawn from (data/out/skill_traj.jsonl: weather 48, system-health 9, generic 48;
coffee, PDF and summarize produced 0 kept rows because stock itself rarely loads those skills).
A model can pass that gate by having learned "weather -> read SKILL.md -> run weather.py" and
nothing general. This gate drives REAL pi (same machinery as skill_gate.py: data/pi_traj.py,
scratch agent dir, data/pi_bench_guard.ts the only extension, a fresh scratch cwd under
/tmp/cavegemma-skill-bench) through skills that are SAFE and CHEAP to exercise for real.

Included (all read-only, local, no money, no sending, no service changes, no long jobs):
  system-health   trained family, NEW prompts (uptime/free/df/ps)            -> held-out phrasing
  coffee          coffee-price-tracker's report.py over its existing SQLite   -> untrained
  pdf             ocr-and-documents: pdftotext on tiny generated PDFs         -> untrained
  pdf-explore     multi-page generated PDFs, answer needs 2 pages             -> untrained
  summarize       summarize: a local text/markdown file                       -> untrained
  eli5            eli5: purely instructional (its 6 section headings)         -> untrained
  humanizer       humanizer: rewrite a local paragraph, numbers unchanged     -> untrained
  weather         control only, 2 new cities
Excluded, and why: podcasts / deep research / autoresearch / literature-review / paper-writing /
replication / ml-training-recipe / research-review / source-comparison / session-log /
paper-code-audit (long jobs, paid or rate-limited APIs, or `/workflow` slash commands that are
not available under `pi -p`); docker (its procedure is `docker run`/`build`/`pip install`, i.e.
writes and pulls); session-search (~/.feynman/sessions does not exist: nothing to find);
preference-mining (its read path runs pref_summarize.py, which rewrites the operator's store);
preview (`xdg-open` would open a window on the desktop); railmark (writes annotated PDFs);
alpha-research (alphaXiv API); sepia/academic-humanizer (humanizer covers the instructional
rewrite case; their prompts would overlap it).

Every prompt here is held out: the script refuses to start if one equals a prompt in
data/build_skill_anchor.py, eval/skill_gate.py or data/out/skill_traj.jsonl.

Judge: data/pi_traj.py judge_session (read the target SKILL.md, ran the prescribed command,
non-empty final answer, no invented fenced output, no unaddressed tool error, no length cap),
plus per-family extras below. Grounding checks are chosen so a TERSE correct answer passes
(a key value or name, normalised for spaces and thousands separators), because a caveman model
answers in a few words.

Usage (candidate served on the bench port, never :8092):
    ~/venvs/unsloth-probe/bin/python eval/skill_gate_wide.py --model ornith-1.5-35b-a3b --label stock --repeats 3
    ~/venvs/unsloth-probe/bin/python eval/skill_gate_wide.py ... --agents-md ~/.local/state/pi-signal/work/AGENTS.md
    ~/venvs/unsloth-probe/bin/python eval/skill_gate_wide.py --rescore stock      # re-judge saved sessions
    ~/venvs/unsloth-probe/bin/python eval/skill_gate_wide.py --table stock cave-v3 # per-family comparison
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))
from owl_ports import BENCH_URL, refuse_live  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
import pi_traj as P  # noqa: E402
from skill_gate import wilson  # noqa: E402

RUN_DIR = ROOT / "eval" / "out" / "skill_gate_wide"
PDF_SKILLS = ["ocr-and-documents", "pdf-explore", "summarize"]
RETAILERS = ["Checkers", "Pick n Pay", "Takealot"]
ELI5_HEADINGS = ["one-sentence summary", "big idea", "how it works", "why it matters",
                 "what to be skeptical of", "if you remember 3 things"]


# --------------------------------------------------------------------------- fixtures

def write_pdf_pages(path: Path, pages: list[list[str]]) -> None:
    """Multi-page version of pi_traj.write_pdf (Helvetica, one text line per entry)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()]
    font_id = 3 + 2 * n
    for i, lines in enumerate(pages):
        ops = ["BT", "/F1 11 Tf", "14 TL", "56 780 Td"] + [f"({esc(ln)}) Tj T*" for ln in lines] + ["ET"]
        stream = "\n".join(ops).encode("latin-1", "replace")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
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


def setup_work(work: Path, fx: dict) -> None:
    for rel, content in (fx.get("files") or {}).items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(content)
    for rel, pages in (fx.get("pdfs") or {}).items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        write_pdf_pages(work / rel, pages)


# --------------------------------------------------------------------------- items

def items_all() -> list[dict]:
    r = random.Random(4242)
    items: list[dict] = []

    def add(fam, iid, prompt, spec, fx=None):
        items.append({"id": f"{fam}-{iid}", "family": fam, "prompt": prompt, "spec": spec,
                      "fixture": fx or {}})

    # weather: control only
    for i, (c, p) in enumerate([("Windhoek", "What's the weather doing in {c} this afternoon?"),
                                ("Hobart", "Do I need a coat in {c} tomorrow morning?")]):
        add("weather", i, p.format(c=c),
            {"skill": "weather", "script_re": r"weather\.py", "ground": "number"})

    # system-health: trained family, held-out prompts
    hs = r"\b(uptime|free|df|nproc|lscpu|ps|ip)\b"
    for i, p in enumerate(["Is RAM or disk the tighter resource on this machine right now?",
                           "What's the 1-minute load average, and is that high for this many cores?",
                           "How full is the root filesystem, as a percentage?",
                           "Has this box been rebooted recently? What's the uptime?"]):
        add("health", i, p, {"skill": "system-health-check", "script_re": hs, "ground": "number"})

    # coffee: report.py over the tracker's existing SQLite (read-only). Grounded by a 2+-digit
    # number from the report OR a retailer name the report printed.
    for i, p in enumerate([
            "Per the coffee price tracker, which retailer had the most Kronung specials in the last 90 days?",
            "Give me a quick coffee tracker summary for the last 45 days.",
            "According to the tracker's history, is Kronung usually cheaper at Pick n Pay or Takealot?",
            "What's the lowest price the coffee tracker has recorded for the 95g Kronung jar?"]):
        add("coffee", i, p, {"skill": "coffee-price-tracker", "script_re": r"report\.py",
                             "ground_alt": {"names": RETAILERS}})

    # pdf (ocr-and-documents): single-page generated documents
    trk = f"ZA{r.randint(100000, 999999)}KX"
    amt = f"{r.randint(1, 4)} {r.randint(100, 999)}.{r.randint(10, 99)}"
    holder = f"{r.choice(P.FIRST)} {r.choice(['Mokoena', 'van Wyk', 'Pillay', 'Botha', 'Dlamini'])}"
    net = f"{r.randint(18, 39)} {r.randint(100, 999)}.{r.randint(10, 99)}"
    docs = [
        ("shipping_notice.pdf", ["SHIPPING NOTICE", "Carrier: Karoo Freight", f"Tracking number: {trk}",
                                 "Parcels: 3", "Estimated delivery: 2026-10-21"],
         "What's the tracking number in shipping_notice.pdf?", [trk]),
        ("municipal_bill.pdf", ["MUNICIPAL ACCOUNT", "Account: 5501 2231 09", "Electricity: R 612.40",
                                "Water: R 288.75", f"Amount due: R {amt}", "Pay by: 2026-11-05"],
         "How much is due on municipal_bill.pdf?", [amt]),
        ("certificate.pdf", ["CERTIFICATE OF COMPLETION", "This certifies that", holder,
                             "has completed: First Aid Level 2", "Date: 2026-09-18"],
         "Who is the certificate in certificate.pdf issued to?", [holder]),
        ("payslip.pdf", ["PAYSLIP - SEPTEMBER 2026", "Employee no: 40217", "Gross pay: R 41 250.00",
                         "PAYE: R 9 870.15", f"Net pay: R {net}"],
         "What net pay does payslip.pdf show?", [net]),
    ]
    for i, (fn, lines, q, exp) in enumerate(docs):
        add("pdf", i, q, {"skill": PDF_SKILLS, "script_re": r"pdftotext|extract_pymupdf|fitz|pymupdf",
                          "expect_norm": exp}, {"pdfs": {fn: [lines]}})

    # pdf-explore: the answer needs more than one page
    n, acc_abs, acc_tab = r.randint(110, 190), "86.1", "84.3"
    gse = f"GSE{r.randint(200000, 299999)}"
    paper = [["Sparse probes for leaf disease detection", "Abstract",
              f"We train a sparse probe on leaf images and report {acc_abs}% accuracy",
              "on held-out farms. Methods are on page 2, results on page 3."],
             ["2. Methods", f"We recruited {n} smallholder farms across three provinces.",
              "Each farm contributed 40 images. Labels were assigned by two agronomists.",
              f"Data availability: raw images are deposited under accession {gse}."],
             ["3. Results", "Table 2. Held-out accuracy by province",
              "Limpopo 82.0%   Mpumalanga 85.9%   KwaZulu-Natal 85.0%",
              f"Overall accuracy: {acc_tab}%", "Figure 3 shows the confusion matrix."]]
    for i, (q, spec) in enumerate([
            ("In study.pdf, how many farms were recruited and what overall accuracy does the results table give?",
             {"expect_all_norm": [str(n), acc_tab]}),
            ("Does the accuracy in study.pdf's abstract match its results table?",
             {"expect_all_norm": [acc_abs, acc_tab]}),
            ("Which data accession does study.pdf give, and on which page?",
             {"expect_norm": [gse]})]):
        add("pdf-explore", i, q, {"skill": PDF_SKILLS, "script_re": r"pdftotext|extract_pymupdf|fitz|pymupdf",
                                  **spec}, {"pdfs": {"study.pdf": paper}})

    # summarize: a local file; grounded by the text's one distinctive name
    arts = [
        ("docs/proposal.md", "Can you summarise docs/proposal.md for me?", "Kestrel",
         "# Proposal: Project Kestrel\n\nWe propose replacing the three ageing diesel generators at "
         "the Hillcrest clinic with a 60 kWh battery bank and rooftop solar. Kestrel would cut fuel "
         "spend by an estimated R 310 000 a year and remove the night-time noise complaints. The "
         "main risk is the eight-week lead time on inverters. We ask the board to approve phase one "
         "(battery bank only) in November and review phase two after one winter of data.\n"),
        ("minutes.txt", "Boil minutes.txt down to the key decisions.", "Thursday",
         "Residents' association minutes, 4 October\n\nPresent: 14 members. The committee agreed to "
         "move the monthly meeting to the first Thursday of each month. The pothole petition will be "
         "sent to the ward councillor with 212 signatures. The treasurer reported R 8 450 in the "
         "account; the security camera upgrade was deferred to next year. Next meeting: Thursday 6 "
         "November, community hall.\n"),
        ("notes/incident.md", "What are the main points of notes/incident.md?", "certificate",
         "# Incident: login failures on the staff portal\n\nFrom 06:40 to 09:15 on 9 October staff "
         "could not log in. The identity provider's signing certificate had been rotated, but the "
         "portal still trusted the old one. 1 140 login attempts failed. Fix: the new certificate was "
         "added to the trust store and the portal restarted. Follow-up: fetch signing keys from the "
         "provider's metadata URL instead of pinning them by hand.\n"),
        ("essay.md", "Short summary of essay.md please.", "baobab",
         "# Why the baobab matters\n\nThe baobab stores up to 120 000 litres of water in its trunk, "
         "which lets it survive droughts that kill most other trees in the lowveld. Elephants strip "
         "its bark in dry years, and some old trees have collapsed since 2005, a die-off researchers "
         "link to hotter, drier seasons. Protecting the baobab means protecting the water and shade "
         "that dozens of other species rely on.\n"),
    ]
    for i, (fn, q, key, text) in enumerate(arts):
        add("summarize", i, q, {"skill": "summarize", "must_read_re": re.escape(fn.split("/")[-1]),
                                "expect": [key], "strict_fences": False}, {"files": {fn: text}})

    # eli5: purely instructional; the procedure is its six section headings
    abstracts = [
        ("notes/abstract.md", "ELI5 notes/abstract.md",
         "# Abstract\n\nWe present a retrieval-augmented transformer that conditions generation on "
         "passages fetched from a dense vector index at every decoding step. On open-domain question "
         "answering it improves exact match by 6.2 points over a parametric baseline of equal size, "
         "while the index can be updated without retraining.\n"),
        ("notes/quantum.md", "Explain notes/quantum.md to me like I'm five.",
         "# Abstract\n\nWe demonstrate error-corrected logical qubits using a surface code of distance "
         "seven on a superconducting processor. Logical error per cycle falls by a factor of 2.1 each "
         "time the code distance increases by two, showing operation below the fault-tolerance "
         "threshold.\n"),
        ("paper_abstract.txt", "What does paper_abstract.txt actually mean, in plain English?",
         "Abstract. We show that mixture-of-experts language models route tokens to a small subset "
         "of expert sub-networks, so that only 3 billion of 35 billion parameters are active per "
         "token. Under a fixed compute budget this sparsity yields lower perplexity than a dense "
         "model with the same number of active parameters.\n"),
    ]
    for i, (fn, q, text) in enumerate(abstracts):
        add("eli5", i, q, {"skill": "eli5", "must_read_re": re.escape(fn.split("/")[-1]),
                           "headings_min": 3, "strict_fences": False}, {"files": {fn: text}})

    # humanizer: rewrite a local paragraph; the skill forbids changing what it says, so the
    # paragraph's one number must survive.
    posts = [
        ("notes/launch_post.md",
         "notes/launch_post.md reads like ChatGPT wrote it. Rewrite it so it sounds like a person.",
         ["47"], "In today's fast-paced world, collaboration is not just a nice-to-have, it's a "
         "game-changer. That's why we're thrilled to unveil Tandem, a seamless, cutting-edge "
         "workspace that empowers teams to unlock their full potential. Early pilots saw meetings "
         "drop by 47 percent. Tandem isn't just a tool. It's a revolution.\n"),
        ("blurb.txt", "Make blurb.txt sound less like AI wrote it.", ["12", "twelve"],
         "Nestled in the heart of the Karoo, our guesthouse offers a truly unforgettable escape. "
         "Whether you're seeking adventure, relaxation, or both, our 12 thoughtfully curated rooms "
         "deliver an experience that is nothing short of magical. Come for the stars, stay for the "
         "memories.\n"),
        ("notes/update.md", "Can you de-AI the wording in notes/update.md?", ["3", "three"],
         "Exciting news! We've delved deep into your feedback and are thrilled to share that the "
         "mobile app now loads 3 times faster. This isn't just an update - it's a testament to our "
         "unwavering commitment to excellence. Stay tuned for more groundbreaking improvements!\n"),
    ]
    for i, (fn, q, num, text) in enumerate(posts):
        add("humanizer", i, q, {"skill": ["humanizer", "sepia", "academic-humanizer"],
                                "must_read_re": re.escape(fn.split("/")[-1]),
                                "expect_num": num, "strict_fences": False}, {"files": {fn: text}})
    return items


def check_held_out(items: list[dict]) -> None:
    import build_skill_anchor as B
    from skill_gate import gate_items
    # run2 = data/skill_run2.py (pdf-explore, humanizer, coffee training prompts, 2026-10-13);
    # building it also checks its fixtures against this gate's.
    seen = {it["prompt"] for it in B.build_prompts(("run1", "run2"))} | {it["prompt"] for it in gate_items()}
    # Hermit's training set (data/hermit_skill/, halo-prep docs/22): an adapter evolved on it is
    # gated here, so its prompts must never be this gate's either. build_tasks.py checks the
    # other direction (fixtures and answer values) before it writes tasks.jsonl.
    hx = ROOT / "data" / "hermit_skill" / "tasks.jsonl"
    if hx.exists():
        seen |= {json.loads(ln)["prompt"].strip() for ln in hx.read_text().splitlines() if ln.strip()}
    for traj in (ROOT / "data" / "out" / "skill_traj.jsonl",
                 ROOT / "data" / "out" / "skill_traj_run2.jsonl"):
        if not traj.exists():
            continue
        for ln in traj.read_text().splitlines():
            msgs = json.loads(ln)["messages"]
            msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
            for m in msgs:
                if m.get("role") == "user":
                    c = m.get("content")
                    c = c if isinstance(c, str) else P._text(c)
                    seen.add(c.strip())
    bad = [it["prompt"] for it in items if it["prompt"].strip() in seen]
    if bad:
        raise SystemExit(f"prompts collide with training/gate prompts: {bad}")


# --------------------------------------------------------------------------- judge

def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def judge(msgs: list[dict], spec: dict) -> tuple[bool, list[str]]:
    base = {k: v for k, v in spec.items()
            if k not in ("expect_norm", "expect_all_norm", "ground_alt", "headings_min", "expect_num")}
    ok, reasons = P.judge_session(msgs, base)
    reasons = set(reasons)
    asst = [m for m in msgs if m.get("role") == "assistant"]
    final = P._text(asst[-1].get("content")).strip() if asst else ""
    # Grounding evidence excludes SKILL.md reads: the coffee SKILL.md names every retailer and
    # quotes prices, so a run that read it and then handed the command back to the user
    # ("Would you like me to run report.py?") would otherwise look grounded. Judge fix after
    # reading stock-agents coffee sessions, 2026-10-12.
    skill_reads = {c.get("id") for c in P.tool_calls(msgs) if c.get("name") == "read"
                   and str((c.get("arguments") or {}).get("path", "")).endswith("SKILL.md")}
    tool_out = "\n".join(P._text(m.get("content")) for m in msgs if m.get("role") == "toolResult"
                         and m.get("toolCallId") not in skill_reads)
    # The prescribed command must also have SUCCEEDED at least once: stock ran
    # `.venv/bin/python scripts/report.py` from the scratch cwd (exit 127), which P's
    # script_re check counts as "ran". Its output is then the only grounding evidence for
    # ground_alt (stock-agents coffee-3-r2 answered "1600 cents" from an empty sqlite3 result
    # and passed on a stray number). Judge fix, 2026-10-12.
    script_out = ""
    if spec.get("script_re"):
        ids = {c.get("id") for c in P.tool_calls(msgs) if c.get("name") == "bash"
               and re.search(spec["script_re"], str((c.get("arguments") or {}).get("command", "")))}
        script_out = "\n".join(P._text(m.get("content")) for m in msgs
                               if m.get("role") == "toolResult" and m.get("toolCallId") in ids
                               and not m.get("isError"))
        if ids and not script_out.strip():
            reasons.add("script_failed")
    if final:
        nf = _norm(final)
        if spec.get("expect_norm") and not any(_norm(e) in nf for e in spec["expect_norm"]):
            reasons.add("answer_not_grounded")
        if spec.get("expect_all_norm") and not all(_norm(e) in nf for e in spec["expect_all_norm"]):
            reasons.add("answer_not_grounded")
        # A rewrite may be delivered by editing the file in place (stock does this): the
        # number then has to survive in the edit/write arguments instead of the reply.
        written = final + "\n" + "\n".join(
            json.dumps(c.get("arguments") or {}) for c in P.tool_calls(msgs)
            if c.get("name") in ("edit", "write"))
        if spec.get("expect_num") and not any(
                re.search(rf"(?<![\d.]){re.escape(e)}(?![\d])", written, re.I) for e in spec["expect_num"]):
            reasons.add("answer_not_grounded")
        if spec.get("ground_alt"):
            # Numbers echoed from the prompt ("the 95g jar", "last 45 days") ground nothing.
            asked = set(P.NUM.findall(" ".join(P._text(m.get("content")) for m in msgs
                                               if m.get("role") == "user")))
            nums = {n for n in P.NUM.findall(final)
                    if (len(n) >= 2 or "." in n) and not P.YEAR.match(n) and n not in asked}
            def grounded(ev: str) -> bool:
                return bool(nums & set(P.NUM.findall(ev))) or any(
                    nm.lower() in final.lower() and nm.lower() in ev.lower()
                    for nm in spec["ground_alt"]["names"])
            if not grounded(script_out):
                # Grounded in some other (non-SKILL.md) tool output, e.g. its own read-only
                # sqlite3 query after report.py failed: right answer, off-procedure route.
                reasons.add("grounded_off_script" if grounded(tool_out) else "answer_not_grounded")
        if spec.get("headings_min"):
            # Headings must BE headings (a markdown heading, a bold line, or a line that is just
            # the heading), not names listed in a sentence: stock eli5-0-r0 passed by writing
            # "Followed the required structure: One-Sentence Summary, Big Idea, ..." after
            # putting the ELI5 into the input file. Fix 2026-10-13.
            lines = [re.sub(r"[*_#`:]", "", ln).strip().lower() for ln in final.splitlines()
                     if re.match(r"\s*(#{1,6}\s|\*\*|__)", ln) or len(ln.strip()) <= 40]
            if sum(any(ln.startswith(h) for ln in lines) for h in ELI5_HEADINGS) < spec["headings_min"]:
                reasons.add("procedure_not_followed")
    # Overwriting the input the user asked about destroys it (stock eli5-0-r0 wrote its ELI5
    # over notes/abstract.md). Only for read-and-explain specs: a rewrite task (humanizer)
    # edits its input on purpose and has no headings_min.
    if spec.get("headings_min") and spec.get("must_read_re") and any(
            c.get("name") in ("edit", "write") and re.search(
                spec["must_read_re"], str((c.get("arguments") or {}).get("path", "")))
            for c in P.tool_calls(msgs)):
        reasons.add("input_overwritten")
    return (not reasons), sorted(reasons)


# A run whose only failures are these did the TASK correctly but not through the skill: the
# secondary "outcome" score. The primary score (skill follow-through) is the question asked.
# procedure_not_followed (eli5's required headings) is procedure too: since 2026-10-13 it no
# longer counts against the outcome score.
ROUTE_ONLY = {"skill_not_read", "script_not_run", "script_failed", "grounded_off_script",
              "procedure_not_followed"}


def outcome_ok(reasons: list[str]) -> bool:
    return set(reasons) <= ROUTE_ONLY


# --------------------------------------------------------------------------- run / report

def summarise(label: str, rd: Path, results: list[dict], items: list[dict], meta: dict) -> dict:
    results.sort(key=lambda x: (x["id"], x["repeat"]))
    fam, fam_ok, fam_out, reasons = Counter(), Counter(), Counter(), Counter()
    for x in results:
        x["outcome"] = outcome_ok(x["reasons"]) if x["session"] else False
        fam[x["family"]] += 1
        fam_ok[x["family"]] += x["pass"]
        fam_out[x["family"]] += x["outcome"]
        reasons.update(x["reasons"])
    tot, ok = len(results), sum(x["pass"] for x in results)
    summary = {**meta, "label": label, "total": [ok, tot],
               "families": {f: [fam_ok[f], fam[f]] for f in fam},
               "outcome": {f: [fam_out[f], fam[f]] for f in fam},
               "reasons": dict(reasons.most_common()),
               "per_prompt": {i["id"]: sum(x["pass"] for x in results if x["id"] == i["id"])
                              for i in items},
               "results": results}
    (rd / "summary.json").write_text(json.dumps(summary, indent=1))
    nw = sum(fam[f] for f in fam if f != "weather")
    kw = sum(fam_ok[f] for f in fam if f != "weather")
    print(f"=== {label}: {ok}/{tot} (95% CI {wilson(ok, tot)}); excluding weather control "
          f"{kw}/{nw} ({wilson(kw, nw)})")
    print("    skill:   " + "  ".join(f"{f} {fam_ok[f]}/{fam[f]}" for f in sorted(fam)))
    print("    outcome: " + "  ".join(f"{f} {fam_out[f]}/{fam[f]}" for f in sorted(fam)))
    print(f"    failure reasons: {dict(reasons.most_common())}")
    return summary


def rescore(label: str, run_dir: Path) -> None:
    rd = run_dir.resolve() / label
    s = json.loads((rd / "summary.json").read_text())
    items = {i["id"]: i for i in items_all()}
    changed = 0
    for x in s["results"]:
        if x["session"] and Path(x["session"]).exists():
            ok, why = judge(P.load_session(x["session"]), items[x["id"]]["spec"])
            changed += ok != x["pass"]
            x["pass"], x["reasons"] = ok, why
    meta = {k: s[k] for k in ("base", "model", "repeats", "agents_md") if k in s}
    meta["rescored"] = True
    print(f"rescored {label}: {changed} verdicts changed")
    summarise(label, rd, s["results"], list(items.values()), meta)


def table(labels: list[str], run_dir: Path) -> None:
    sums = [json.loads((run_dir.resolve() / lb / "summary.json").read_text()) for lb in labels]
    fams = sorted({f for s in sums for f in s["families"]})
    print("skill follow-through pass/runs (task outcome correct by any route)")
    print("| family | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    for f in fams + ["TOTAL"]:
        cells = []
        for s in sums:
            if f == "TOTAL":
                k, n = s["total"]
                o = sum(v[0] for v in s.get("outcome", {}).values())
            else:
                k, n = s["families"].get(f, [0, 0])
                o = s.get("outcome", {}).get(f, [0, 0])[0]
            cells.append(f"{k}/{n} ({o})")
        print(f"| {f} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BENCH_URL)
    ap.add_argument("--model", default=P.MODEL)
    ap.add_argument("--label", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--thinking", default="off")
    ap.add_argument("--agents-md", type=Path, default=None)
    ap.add_argument("--families", default=None, help="comma list to run a subset")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--rescore", metavar="LABEL", help="re-judge a saved run; no pi runs")
    ap.add_argument("--table", nargs="+", metavar="LABEL", help="print a per-family table")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.rescore, args.run_dir)
    if args.table:
        return table(args.table, args.run_dir)
    refuse_live(args.base)

    items = items_all()
    check_held_out(items)
    if args.families:
        keep = set(args.families.split(","))
        items = [i for i in items if i["family"] in keep]
    label = args.label or args.model
    rd = args.run_dir.resolve() / label
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    agents = [P.make_agent_dir(P.WORK_ROOT / f"agent-wide-{label}-w{w}", args.base,
                               P.PROVIDER, args.model) for w in range(args.workers)]
    jobs = [(it, r) for r in range(args.repeats) for it in items]
    slots, lock, results = list(range(args.workers)), threading.Lock(), []

    def one(job):
        it, r = job
        with lock:
            w = slots.pop()
        try:
            work = P.scratch_cwd("wide", label, f"{it['id']}-r{r}")
            setup_work(work, it["fixture"])
            if args.agents_md:
                shutil.copy(args.agents_md, work / "AGENTS.md")
            res = P.run_pi(agents[w], work, it["prompt"], P.PROVIDER, args.model,
                           thinking=args.thinking, context_files=bool(args.agents_md))
            if res["session"]:
                keep = rd / f"{it['id']}-r{r}.session.jsonl"
                shutil.copy(res["session"], keep)
                res["session"] = str(keep)
                ok, why = judge(P.load_session(res["session"]), it["spec"])
            else:
                ok, why = False, [str(res["rc"]) if res["rc"] == "timeout" else "no_session"]
            rec = {"id": it["id"], "family": it["family"], "repeat": r, "prompt": it["prompt"],
                   "pass": ok, "reasons": why, "seconds": res["seconds"], "session": res["session"]}
            with lock:
                results.append(rec)
                print(f"  {it['id']:14s} r{r}  {'PASS' if ok else 'fail'}  {res['seconds']:6.1f}s  "
                      f"{','.join(why)}", flush=True)
        finally:
            with lock:
                slots.append(w)

    print(f">>> wide skill gate: {label}  {len(items)} prompts x {args.repeats}  base={args.base}")
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))
    summarise(label, rd, results, items,
              {"base": args.base, "model": args.model, "repeats": args.repeats,
               "agents_md": str(args.agents_md) if args.agents_md else None})


if __name__ == "__main__":
    main()
