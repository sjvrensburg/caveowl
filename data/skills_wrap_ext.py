"""Give the run-2 synthtraces rows a <skills> block (run 3, 2026-10-13).

Why: run 2 trained 182 caveman-rewritten synthtraces rows whose system prompt had NO skills block
(data/build_synthtraces.py removed it). Its wide skill gate stayed at stock's 28/81 and pdf went
5/12 -> 0/12: the model answered PDF questions with `pdftotext`/`read` and never opened SKILL.md.
Suspected cause: 10% of the mix showed tool-first solving with no skill in sight. Production
always lists skills, so a row without them is off-distribution anyway.

What: each row gets pi's skills block (same header, same markup, alphabetical as pi lists them)
with 5, 6 or 8 skills, matching the skill trajectories' own counts, drawn ONLY from skills that
cannot apply to a question about a code repository (SAFE). The row then shows "skills listed,
none relevant, none opened", the negative case the skill rows' targets are the positive of.
Research/code skills (paper-code-audit, ml-training-recipe, docker, summarize, eli5, ...) are
left out because a repo question could plausibly call for them and these rows never read one.

The skill blocks are copied verbatim from the skill trajectories (run1/run2 clean), so the text is
what pi renders. Only the system message changes; the Haiku rewrite is untouched. Checked: removing
the inserted block gives back the row's original system prompt exactly.

Run:
    python3 data/skills_wrap_ext.py   # -> data/out/ext_traj_cave_skills.jsonl
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
SRC = OUT / "ext_traj_cave.jsonl"
DST = OUT / "ext_traj_cave_skills.jsonl"
SKILL_ROWS = [OUT / "skill_traj_run1_clean.jsonl", OUT / "skill_traj_run2_clean.jsonl"]
SAFE = ["academic-humanizer", "coffee-price-tracker", "deep-research-podcast", "humanizer",
        "ocr-and-documents", "paper-writing", "pdf-explore", "preference-mining", "preview",
        "sepia", "session-log", "system-health-check", "weather"]
SKILL = re.compile(r"  <skill>\n.*?</skill>\n", re.S)
BLOCK = re.compile(r"\n*<skills>.*?</skills>\n*", re.S)


def main() -> None:
    blocks, counts, template = {}, Counter(), None
    for f in SKILL_ROWS:
        for line in f.open():
            sp = json.loads(line)["messages"][0]["content"]
            found = SKILL.findall(sp)
            counts[len(found)] += 1
            for b in found:
                blocks[re.search(r"<name>(.*?)</name>", b).group(1)] = b
            template = template or sp
    assert set(SAFE) <= set(blocks), set(SAFE) - set(blocks)
    shell = BLOCK.search(template).group(0)            # header + <available_skills> wrapper
    inner = re.search(r"<available_skills>\n(.*?)</available_skills>", shell, re.S).group(1)
    ks, weights = zip(*sorted(counts.items()))
    rows, n_by = [], Counter()
    for line in SRC.open():
        r = json.loads(line)
        rng = random.Random(r["key"])
        k = rng.choices(ks, weights)[0]
        pick = sorted(rng.sample(SAFE, k))
        sk = shell.replace(inner, "".join(blocks[n] for n in pick))
        sp = r["messages"][0]["content"]
        assert "<skills>" not in sp
        # build_synthtraces removed the block with BLOCK -> "\n\n"; put it back at that seam,
        # which is the template's: just before the line that follows </skills> there.
        after = template[BLOCK.search(template).end():][:60]
        i = sp.index(after.split("\n")[0])
        new = sp[:i].rstrip("\n") + sk + sp[i:]
        assert BLOCK.sub("\n\n", new) == sp, r["key"]
        r["messages"][0]["content"] = new
        r["skills_listed"] = pick
        rows.append(r)
        n_by[k] += 1
    DST.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows, skills per row {dict(n_by)} (skill rows: {dict(counts)}) -> {DST}")


if __name__ == "__main__":
    main()
