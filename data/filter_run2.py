"""Run-2 quality filter over the skill trajectories (operator-approved, 2026-10-13).

The judge checks that the procedure was followed and the answer is grounded; it does not check
that the answer is RIGHT or that nothing was made up. These rows passed it and were dropped
after reading them:

  - humanizer rewrites that invent facts the source never states (the humanizer skill itself
    forbids that). Read all 15 run-2 rows against their source paragraph; 5 add details.
  - coffee rows that keep querying SQLite after `report.py` (9 of 21): off-procedure, and one
    (coffee2-06) concludes the wrong retailer from its own table. Kept: rows whose last tool
    call is the report.py run.
  - run-1 rows found by the run-2 builder: pdf-lease-2 (its only SKILL.md read failed, so the
    pruned row never opens the skill) and pdf-lab-3 / pdf-lease-3 / pdf-agenda-2 (tool output
    lists other runs' bench scratch paths).

Writes data/out/skill_traj_run2_clean.jsonl and data/out/skill_traj_run1_clean.jsonl.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
INVENTS = {  # key -> what was invented
    "skill:hum-13-a1": "'last October', 'six hours' drive, kloof details",
    "skill:hum-22-a2": "'Saturday mornings'",
    "skill:hum-16-a2": "'one student at a time', 'once a week'",
    "skill:hum-25-a5": "'lack reliable grid access', a two-part plan",
    "skill:hum-01-a1": "a project that 'changed how a team made decisions'",
}
RUN1_DROP = {"skill:pdf-lease-2", "skill:pdf-lab-3", "skill:pdf-lease-3", "skill:pdf-agenda-2"}


def coffee_ends_at_report(row: dict) -> bool:
    calls = [json.dumps(tc["function"]["arguments"]) for m in row["messages"]
             if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    return bool(calls) and "report.py" in calls[-1]


def main() -> None:
    rows = [json.loads(l) for l in (OUT / "skill_traj_run2.jsonl").open()]
    keep = [r for r in rows if r["key"] not in INVENTS
            and (r["family"] != "coffee" or coffee_ends_at_report(r))]
    (OUT / "skill_traj_run2_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    run1 = [json.loads(l) for l in (OUT / "skill_traj.jsonl").open()]
    k1 = [r for r in run1 if r["key"] not in RUN1_DROP]
    (OUT / "skill_traj_run1_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in k1))
    from collections import Counter
    print("run2", len(rows), "->", len(keep), dict(Counter(r["family"] for r in keep)))
    print("run1", len(run1), "->", len(k1), "dropped", sorted(RUN1_DROP & {r["key"] for r in run1}))


if __name__ == "__main__":
    main()
