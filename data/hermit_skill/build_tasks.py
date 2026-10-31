"""Hermit training set for caveowl's skill follow-through (halo-prep docs/22, Phase 1).

Hermit evolves pi-signal's AGENTS.md around a frozen caveowl and a frozen pi. It trains on THIS
set; eval/skill_gate_wide.py stays the held-out final gate and Hermit must never see its prompts.
So, like data/skill_run2.py, every item here is checked against the gate before anything is
written (skill_run2.check_disjoint: no shared prompt, fixture name, 8-word run of fixture text,
or gate answer value). It is also checked against every LoRA training prompt (run1, run2, run4),
so caveowl has never been trained on any of these prompts and the failure signal is honest.
eval/skill_gate_wide.py's check_held_out() reads tasks.jsonl back, so the guard covers this set too.

Families: the five the plan trains on (coffee, humanizer, pdf, pdf-explore, summarize), 8
prompts each. eli5, system-health and weather are held out whole: the wide gate scores all
eight, so a gain there is transfer, not fit.

Outputs (beside this file):
  tasks.jsonl            {id, family, prompt, fixture}: learner-visible, so no expected answers
  train.txt, val.txt     6 + 2 ids per family; GEPA's Pareto selection runs on val
  .private/specs.json    the judge's specs (expected values), read only by judge.py
  seed_adapter/ADAPTER.md  adapter v0 = halo-prep's config/pi-signal/AGENTS.md, byte for byte

Run:  python3 data/hermit_skill/build_tasks.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

PDF_SKILLS = ["ocr-and-documents", "pdf-explore", "summarize"]
PDF_SCRIPT_RE = r"pdftotext|extract_pymupdf|fitz|pymupdf"
RETAILERS = ["Checkers", "Pick n Pay", "Takealot"]
HUMANIZER_SKILLS = ["humanizer", "sepia", "academic-humanizer"]
AGENTS_MD = Path.home() / "Projects" / "halo-prep" / "config" / "pi-signal" / "AGENTS.md"
N_VAL = 2  # per family, the last N_VAL items


def items() -> list[dict]:
    out: list[dict] = []

    def add(fam: str, prompt: str, spec: dict, fx: dict | None = None) -> None:
        n = sum(1 for it in out if it["family"] == fam)
        out.append({"id": f"hx-{fam}-{n}", "family": fam, "prompt": prompt, "spec": spec,
                    "fixture": fx or {}})

    # coffee: report.py over the tracker's real SQLite (read-only; the bench guard refuses the
    # write paths). Grounding as in the gate: a number or retailer name report.py printed.
    for p in ["Using the coffee tracker, how many Kronung specials did Checkers run in the past two months?",
              "What does the coffee price tracker show for the last 30 days?",
              "Which shop does the coffee tracker show as cheapest for Kronung lately?",
              "Check the coffee tracker: has anyone had a Kronung special in the past fortnight?",
              "From the coffee price history, what's the usual special price for Kronung at Takealot?",
              "Pull the coffee tracker report for the past 60 days and tell me the best deal in it.",
              "Coffee tracker question: how often did Pick n Pay discount Kronung over the last quarter?",
              "Is there a Kronung special worth buying right now, going by the coffee tracker?"]:
        add("coffee", p, {"skill": "coffee-price-tracker", "script_re": r"report\.py",
                          "ground_alt": {"names": RETAILERS}})

    # pdf (ocr-and-documents): one-page generated documents, one fact each
    docs = [
        ("fuel_receipt.pdf", ["FUEL RECEIPT", "Station: Sasolburg North", "Pump: 6", "Litres: 41.27",
                              "Total: R 1 018.53"],
         "What did the fuel cost according to fuel_receipt.pdf?", ["1 018.53"]),
        ("booking_confirmation.pdf", ["BOOKING CONFIRMATION", "Guest house: Die Hoek, Clarens",
                                      "Booking reference: KQ7741-MB", "Nights: 2"],
         "What's the booking reference in booking_confirmation.pdf?", ["KQ7741-MB"]),
        ("lease_summary.pdf", ["LEASE SUMMARY", "Unit 14B, Rosebank Gardens", "Lease term: 18 months",
                               "Monthly rent: R 9 350.00", "Deposit: R 18 700.00"],
         "How much is the monthly rent in lease_summary.pdf?", ["9 350.00"]),
        ("lab_report.pdf", ["PATHOLOGY REPORT", "Full blood count", "Haemoglobin: 13.8 g/dL",
                            "White cell count: 6.1 x10^9/L", "Platelets: 254 x10^9/L"],
         "What haemoglobin value does lab_report.pdf give?", ["13.8"]),
        ("warranty_card.pdf", ["WARRANTY CARD", "Product: cordless drill DX-18",
                               "Serial number: WX-55120-QT", "Warranty: 24 months from purchase"],
         "Find the serial number on warranty_card.pdf.", ["WX-55120-QT"]),
        ("traffic_fine.pdf", ["NOTICE OF INFRINGEMENT", "Offence: speeding, 78 km/h in a 60 zone",
                              "Vehicle registration: CA 482-917", "Fine: R 1 250.00"],
         "Which vehicle registration is on traffic_fine.pdf?", ["CA 482-917"]),
        ("school_report.pdf", ["TERM REPORT", "Learner: Thandi Nkosi", "English: 71%",
                               "Mathematics: 83%", "Life Sciences: 76%"],
         "What mark did the learner get for Mathematics in school_report.pdf?", ["83"]),
        ("insurance_renewal.pdf", ["POLICY RENEWAL", "Policy: household contents",
                                   "Renewal date: 2026-11-30", "Premium: R 734.90 per month"],
         "When does the policy in insurance_renewal.pdf renew?",
         ["2026-11-30", "30 November 2026", "November 30"]),
    ]
    for fn, lines, q, exp in docs:
        add("pdf", q, {"skill": PDF_SKILLS, "script_re": PDF_SCRIPT_RE, "expect_norm": exp},
            {"pdfs": {fn: [lines]}})

    # pdf-explore: multi-page generated reports; every answer needs more than one page
    grant = [["Reading for Meaning: final grant report", "Summary",
              "The programme spent 91.4% of its budget and reached every target district.",
              "Participation is on page 2; the expenditure table is on page 3."],
             ["2. Participation", "236 primary schools took part across four districts.",
              "Each school nominated two teachers for the coaching cycle.",
              "Queries: evaluations@ukhanyo.org.za"],
             ["3. Expenditure", "Table 4. Spend by line item",
              "Coaching R 2 904 100   Books R 1 377 800   Travel R 530 400",
              "Share of budget spent: 89.7%"]]
    survey = [["Household water survey 2026", "Introduction",
               "With a 64% response rate, the survey gives a reliable picture of access.",
               "Page 2 covers sampling; page 3 has the findings."],
              ["2. Sampling", "We sampled 1 870 households in eleven wards.",
               "Fieldworkers revisited each non-responding household once."],
              ["3. Results", "Final response rate: 61.2%",
               "Median household income: R 7 400 per month",
               "Households with a yard tap: 58%"]]
    pumps = [["Solar irrigation pump field trial", "Abstract",
              "Pumps reached a mean efficiency of 72.5% under field conditions.",
              "Page 2 has the method and page 3 the findings."],
             ["2. Methods", "We tested 38 pumps on smallholdings in the Eastern Cape.",
              "Each pump ran for six weeks with a logging flow meter."],
             ["3. Results", "Table 1. Efficiency by model",
              "Mean efficiency: 70.8%",
              "Lowest: Aqualine AL-200 (61.0%)   Highest: Sunflo SF-5 (78.9%)"]]
    for fn, pages, q, spec in [
            ("grant_report.pdf", grant,
             "In grant_report.pdf, how many schools took part and what share of the budget does the final table say was spent?",
             {"expect_all_norm": ["236", "89.7"]}),
            ("grant_report.pdf", grant,
             "Does the budget figure in grant_report.pdf's summary agree with its expenditure table?",
             {"expect_all_norm": ["91.4", "89.7"]}),
            ("grant_report.pdf", grant,
             "What contact email does grant_report.pdf give for queries, and on which page is it?",
             {"expect_norm": ["evaluations@ukhanyo.org.za"]}),
            ("survey_results.pdf", survey,
             "How many households were sampled in survey_results.pdf, and what final response rate does it report?",
             {"expect_all_norm": ["1 870", "61.2"]}),
            ("survey_results.pdf", survey,
             "Is the response rate in survey_results.pdf's introduction the same as the one in its results?",
             {"expect_all_norm": ["64", "61.2"]}),
            ("pump_trial.pdf", pumps,
             "In pump_trial.pdf, how many pumps were tested and what mean efficiency do the results give?",
             {"expect_all_norm": ["38", "70.8"]}),
            ("survey_results.pdf", survey,
             "What median household income does survey_results.pdf report, and how big was the sample?",
             {"expect_all_norm": ["7 400", "1 870"]}),
            ("pump_trial.pdf", pumps,
             "Which pump model did worst in pump_trial.pdf, and does the abstract's efficiency match the results?",
             {"expect_all_norm": ["Aqualine", "72.5", "70.8"]})]:
        add("pdf-explore", q, {"skill": PDF_SKILLS, "script_re": PDF_SCRIPT_RE, **spec},
            {"pdfs": {fn: pages}})

    # summarize: a local text file, grounded by its one distinctive name
    arts = [
        ("docs/memo.md", "Summarise docs/memo.md.", "Marula",
         "# Memo: moving the Marula office\n\nFrom 1 February the Marula team moves from the third "
         "floor of Riverside Park to the new building on Main Road. Parking drops from 40 bays to 22, "
         "so staff are asked to use the shuttle from the station. IT will move desks over the weekend "
         "of 24 January; anyone with a desktop machine must label it by the Friday.\n"),
        ("report.txt", "Give me the gist of report.txt.", "Nightjar",
         "Quarterly report: Project Nightjar\n\nThe bird-call recognition app reached 18 400 monthly "
         "users, up from 11 900. Accuracy on the new owl recordings is still poor at 64%, because the "
         "training set has few night recordings. The team will run a recording drive with two birding "
         "clubs in the next quarter. Spend is on budget.\n"),
        ("notes/retro.md", "TL;DR of notes/retro.md?", "Osprey",
         "# Sprint retro: Osprey release\n\nWhat went well: the payment flow shipped on time and support "
         "tickets fell by a third. What went badly: the release was held two days by a flaky end-to-end "
         "test nobody owned. Action: each test suite gets a named owner, and a test that fails twice "
         "without a code change is quarantined automatically.\n"),
        ("letter.txt", "What's letter.txt about, briefly?", "Mopani",
         "Dear residents of Mopani Close,\n\nThe municipality will replace the water main under our street "
         "between 10 and 21 November. Water will be off from 08:00 to 16:00 on weekdays during that time. "
         "A tanker will stand at the corner with Acacia Road. Please move cars off the verge by 07:30 each "
         "morning.\n\nThe street committee\n"),
        ("notes/plan.md", "Summarize notes/plan.md in a few lines.", "Pangolin",
         "# Plan: Pangolin data migration\n\nWe will move the customer database from the old MySQL server "
         "to managed Postgres in 3 steps: copy and verify a snapshot, run both databases side by side "
         "for two weeks with writes mirrored, then switch reads over. Rollback is a DNS change back to the "
         "old server. The risky part is the 2 million rows with mixed character encodings.\n"),
        ("review.md", "Can you condense review.md into its main points?", "Fynbos",
         "# Review: Fynbos guesthouse\n\nThe rooms are spotless and the breakfast is the best we had on the "
         "trip, with homemade bread and local honey. The downside is the road: the last 6 km are gravel and "
         "rough after rain. Wi-Fi only works in the lounge. We would stay again, but in a higher car.\n"),
        ("docs/postmortem.txt", "What happened, according to docs/postmortem.txt? Keep it short.", "Hornbill",
         "Postmortem: Hornbill checkout outage\n\nFor 52 minutes on 14 October no card payments went through. "
         "A configuration push set the payment provider's timeout to 50 milliseconds instead of 50 seconds. "
         "Alerts fired after 9 minutes but went to an old on-call rota. Fixes: unit suffixes are now "
         "required in timeout settings, and the rota is generated from the HR system.\n"),
        ("article.md", "Quick summary of article.md, please.", "Okavango",
         "# Floods arrive early in the Okavango\n\nThe annual flood reached Maun 21 days earlier than "
         "usual this year, after heavy rain in the Angolan highlands. Guides expect good game viewing as "
         "animals gather along the channels, but some mokoro routes are already too deep to pole. Lodges "
         "advise visitors to confirm their activities before travelling.\n"),
    ]
    for fn, q, key, text in arts:
        add("summarize", q, {"skill": "summarize", "must_read_re": re.escape(fn.split("/")[-1]),
                             "expect": [key], "strict_fences": False}, {"files": {fn: text}})

    # humanizer: rewrite a local paragraph; its distinctive number must survive
    posts = [
        ("notes/newsletter.md", "notes/newsletter.md sounds robotic. Make it read like a human wrote it.",
         ["28"], "In an ever-evolving landscape, our community continues to flourish. This month, a "
         "remarkable 28 volunteers rolled up their sleeves to transform the park into a vibrant hub of "
         "connection. Together, we are not just building gardens - we are cultivating hope.\n"),
        ("pitch.txt", "Rewrite pitch.txt so it doesn't sound AI-generated.", ["90", "ninety"],
         "Introducing LedgerLite: the revolutionary, all-in-one solution that seamlessly streamlines your "
         "bookkeeping journey. Small businesses using LedgerLite reclaim up to 90 minutes a week, "
         "empowering them to focus on what truly matters. Elevate your finances today.\n"),
        ("notes/about.md", "Humanize notes/about.md.", ["15", "fifteen"],
         "At our core, we are passionate innovators dedicated to delivering excellence. For over 15 years, "
         "our team has leveraged cutting-edge expertise to craft bespoke solutions that resonate. We don't "
         "just meet expectations; we redefine them.\n"),
        ("promo.md", "Take the AI gloss off promo.md.", ["64"],
         "Unlock a world of flavour with our artisanal coffee subscription! Each month, discover beans "
         "curated from 64 handpicked farms, delivered straight to your door. It's more than coffee - it's "
         "a sensory adventure that awakens your passion for life.\n"),
        ("notes/bio.txt", "notes/bio.txt is full of buzzwords; rewrite it in a natural voice.", ["20", "twenty"],
         "A visionary thought leader with 20 years of experience, Sam is a results-driven change agent who "
         "thrives at the intersection of strategy and execution. Sam is passionate about empowering teams "
         "to unlock synergies and drive transformative growth.\n"),
        ("release.md", "Can you make release.md sound like a real person wrote it?", ["2.4"],
         "We're thrilled to announce version 2.4, a game-changing update that takes your experience to the "
         "next level! With a host of exciting enhancements, this release underscores our unwavering "
         "dedication to delivering seamless, delightful moments for every user.\n"),
        ("notes/welcome.md", "Strip the AI cliches out of notes/welcome.md.", ["300"],
         "Welcome aboard! We're beyond excited to have you join our thriving family of 300 passionate "
         "members. Here, every voice matters, and together we embark on a transformative journey of "
         "growth, discovery and meaningful connection.\n"),
        ("listing.txt", "listing.txt reads like a bot wrote it. Fix the tone.", ["4", "four"],
         "Welcome to your dream home! This stunning 4-bedroom residence seamlessly blends timeless elegance "
         "with modern convenience. Nestled in a sought-after neighbourhood, it offers an unparalleled "
         "lifestyle that you truly deserve.\n"),
    ]
    for fn, q, num, text in posts:
        add("humanizer", q, {"skill": HUMANIZER_SKILLS, "must_read_re": re.escape(fn.split("/")[-1]),
                             "expect_num": num, "strict_fences": False}, {"files": {fn: text}})
    return out


def check(hx: list[dict]) -> None:
    """Refuse to write anything that would leak into the held-out gate, or that caveowl trained on."""
    import build_skill_anchor as B
    import skill_run2
    trained = B.build_prompts(("run1", "run2", "run4"))
    skill_run2.check_disjoint(hx, trained)
    trained_prompts = {it["prompt"].strip() for it in trained}
    dup = [it["id"] for it in hx if it["prompt"].strip() in trained_prompts]
    if dup:
        raise SystemExit(f"prompts equal a LoRA training prompt: {dup}")


def main() -> None:
    hx = items()
    check(hx)
    with (HERE / "tasks.jsonl").open("w") as f:
        for it in hx:
            f.write(json.dumps({k: it[k] for k in ("id", "family", "prompt", "fixture")}) + "\n")
    fams: dict[str, list[str]] = {}
    for it in hx:
        fams.setdefault(it["family"], []).append(it["id"])
    train = [i for ids in fams.values() for i in ids[:-N_VAL]]
    val = [i for ids in fams.values() for i in ids[-N_VAL:]]
    (HERE / "train.txt").write_text("\n".join(train) + "\n")
    (HERE / "val.txt").write_text("\n".join(val) + "\n")
    # every id, for `hermit eval ... data.test=all.txt` (Phase 1 baseline; eval skips check's overlap rule)
    (HERE / "all.txt").write_text("\n".join(it["id"] for it in hx) + "\n")
    priv = HERE / ".private"
    priv.mkdir(exist_ok=True)
    (priv / "specs.json").write_text(json.dumps({it["id"]: it["spec"] for it in hx}, indent=1))
    shutil.copyfile(AGENTS_MD, HERE / "seed_adapter" / "ADAPTER.md")
    print(f"{len(hx)} items ({len(train)} train, {len(val)} val) in {len(fams)} families; "
          f"disjoint from the wide gate and from LoRA runs 1, 2 and 4")


if __name__ == "__main__":
    main()
