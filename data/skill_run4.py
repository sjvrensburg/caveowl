"""Run-4 skill trajectory prompts: the `pdf` family (ocr-and-documents on small single-page PDFs).

Why: run 3 (README "Run 3") passed every family of eval/skill_gate_wide.py at or above stock
except pdf: 2/12 vs stock 5/12. The answers were right; the model read the tiny generated PDF
raw (`read` on the file, or `pdftotext` itself) and never opened SKILL.md, so the judge failed
`skill_not_read`. pdf is the family with the fewest skill rows: run 1 built 14 prompts, 10 passed,
7 survived data/filter_run2.py. pdf-explore, with 23 rows, went 6/9 -> 9/9.

Same rules as data/skill_run2.py (its check_disjoint() is reused): the gate's fixture shape
(`{"pdfs": {name: [lines]}}`), the gate's judge (spec["judge"] == "wide"), the gate's skills and
script regex, and prompts, file names and answer values disjoint from the gate's and run 1's.
Twelve documents of kinds the gate does not use, two or three questions each.

Build (stock on :8110):
    ~/venvs/unsloth-probe/bin/python data/build_skill_anchor.py --set run4 --extra-attempts 2 --max-len 16384
"""
from __future__ import annotations

import random

from skill_run2 import PDF_SCRIPT_RE, PDF_SKILLS, _gate, _gate_values


def _docs(banned: set[str]) -> list[tuple[str, list[str], list[tuple[str, str]]]]:
    r = random.Random("caveowl-run4-pdf")

    def pick(make):
        while True:
            v = make()
            if v not in banned:
                return v
    d = lambda: f"2026-{r.randint(10, 11)}-{r.randint(13, 28)}"  # noqa: E731
    ref = pick(lambda: f"BP{r.randint(1000, 9999)}")
    seat = pick(lambda: f"{r.randint(11, 39)}{r.choice('ACDF')}")
    gate_no = pick(lambda: f"B{r.randint(2, 19)}")
    serial = pick(lambda: f"WX-{r.randint(100000, 999999)}")
    w_end = d()
    vet = pick(lambda: f"{r.randint(400, 1900)}.{r.randint(10, 99)}")
    pet = r.choice(["Biscuit", "Juno", "Pepper", "Mango", "Tiger"])
    fine = pick(lambda: f"{r.randint(300, 1500)}.00")
    plate = pick(lambda: f"{r.choice(['CA', 'GP', 'ND'])} {r.randint(100, 999)}-{r.randint(100, 999)}")
    maths = pick(lambda: f"{r.randint(55, 98)}%")
    teacher = f"Ms {r.choice(['Naidoo', 'Fourie', 'Khumalo', 'Adams'])}"
    dn = pick(lambda: f"DN{r.randint(10000, 99999)}")
    cartons = pick(lambda: str(r.randint(12, 90)))
    ifsc = pick(lambda: f"{r.randint(100000, 999999)}")
    rate = pick(lambda: f"{r.randint(70, 140) / 10:.1f}%")
    quote = pick(lambda: f"{r.randint(2, 9)},{r.randint(100, 999)}.00")
    prem = pick(lambda: f"{r.randint(300, 1200)}.{r.randint(10, 99)}")
    pol = pick(lambda: f"POL-{r.randint(1000000, 9999999)}")
    gym = pick(lambda: f"{r.randint(299, 899)}.00")
    months = pick(lambda: str(r.choice([6, 24, 36])))
    outage = d()
    hours = pick(lambda: f"{r.randint(5, 9)}:00 to {r.randint(13, 17)}:00")
    table = pick(lambda: str(r.randint(4, 30)))
    tip = pick(lambda: f"{r.randint(20, 90)}.00")
    return [
        ("boarding_pass.pdf", ["BOARDING PASS", f"Booking ref: {ref}", "Flight: FA 214 CPT-JNB",
                               f"Seat: {seat}", f"Gate: {gate_no}", "Boarding: 06:40"],
         [("What seat am I in according to boarding_pass.pdf?", seat),
          ("Which gate does boarding_pass.pdf list?", gate_no),
          ("Read boarding_pass.pdf and give me the booking reference.", ref)]),
        ("warranty_card.pdf", ["WARRANTY CARD", "Product: cordless drill", f"Serial: {serial}",
                               f"Warranty ends: {w_end}"],
         [("What serial number is on warranty_card.pdf?", serial),
          ("When does the warranty in warranty_card.pdf end?", w_end)]),
        ("vet_invoice.pdf", ["ANIMAL CLINIC INVOICE", f"Patient: {pet}", "Consultation: R 450.00",
                             f"Total: R {vet}"],
         [("What's the total on vet_invoice.pdf?", vet),
          ("Which pet is vet_invoice.pdf for?", pet)]),
        ("traffic_fine.pdf", ["NOTICE OF INFRINGEMENT", f"Vehicle: {plate}", "Offence: parking in a loading zone",
                              f"Fine: R {fine}"],
         [("How much is the fine in traffic_fine.pdf?", fine),
          ("What registration number does traffic_fine.pdf name?", plate)]),
        ("school_report.pdf", ["TERM 3 REPORT", f"Class teacher: {teacher}", f"Mathematics: {maths}",
                               "English: 71%", "Days absent: 2"],
         [("What maths mark is in school_report.pdf?", maths),
          ("Who is the class teacher on school_report.pdf?", teacher)]),
        ("delivery_note.pdf", ["DELIVERY NOTE", f"Note no: {dn}", f"Cartons: {cartons}",
                               "Received by: stores"],
         [("How many cartons does delivery_note.pdf list?", cartons),
          ("What's the note number on delivery_note.pdf?", dn)]),
        ("bank_letter.pdf", ["HOME LOAN CONFIRMATION", f"Branch code: {ifsc}", f"Interest rate: {rate}",
                             "Term: 240 months"],
         [("What interest rate does bank_letter.pdf confirm?", rate),
          ("Extract bank_letter.pdf: what branch code is given?", ifsc)]),
        ("roof_quote.pdf", ["QUOTATION", "Job: roof waterproofing", f"Quote total: R {quote}",
                            "Valid for 30 days"],
         [("What does roof_quote.pdf quote in total?", quote)]),
        ("insurance_renewal.pdf", ["RENEWAL NOTICE", f"Policy: {pol}", f"New monthly premium: R {prem}"],
         [("What's the new premium in insurance_renewal.pdf?", prem),
          ("Which policy number is on insurance_renewal.pdf?", pol)]),
        ("gym_contract.pdf", ["MEMBERSHIP AGREEMENT", f"Monthly fee: R {gym}", f"Minimum term: {months} months"],
         [("What monthly fee is in gym_contract.pdf?", gym),
          ("How long is the minimum term in gym_contract.pdf?", months)]),
        ("outage_notice.pdf", ["PLANNED MAINTENANCE", f"Date: {outage}", f"Hours: {hours}",
                               "Area: Rondebosch East"],
         [("On what date is the outage in outage_notice.pdf?", outage),
          ("During which hours is power off per outage_notice.pdf?", hours)]),
        ("restaurant_bill.pdf", ["TAX INVOICE", f"Table: {table}", "Subtotal: R 640.00", f"Tip: R {tip}"],
         [("What table number is on restaurant_bill.pdf?", table),
          ("How much tip does restaurant_bill.pdf show?", tip)]),
    ]


def items() -> list[dict]:
    _, gate = _gate()
    out = []
    for fn, lines, qs in _docs(_gate_values(gate)):
        stem = fn.removesuffix(".pdf")
        for q, (prompt, ans) in enumerate(qs):
            out.append({"id": f"pdf4-{stem}-{q}", "family": "pdf", "prompt": prompt,
                        "fixture": {"pdfs": {fn: [lines]}},
                        "spec": {"skill": PDF_SKILLS, "script_re": PDF_SCRIPT_RE, "judge": "wide",
                                 "expect_norm": [ans]}})
    return out
