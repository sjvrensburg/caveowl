"""The caveman rewrite prompts and the resume-key scheme, ported from cavegemma, MIT.

Copied verbatim from data/synthesize.py of https://github.com/JuliusBrussee/cavegemma (MIT,
Copyright (c) 2026 Julius Brussee): CATEGORY_NORMALIZE_SYSTEM, CAVEMAN_INSTRUCTIONS,
caveman_system(), _hash_key() and _existing_keys(). Only these pieces are kept; upstream's
`claude -p` / `codex exec` drivers are not, because data/synthesize_batch.py replaces them with
the Message Batches API. Keep the text byte-identical to upstream so pairs built here are
comparable with upstream's.

data/seeds/skill_md.md is the caveman skill's SKILL.md from
https://github.com/JuliusBrussee/caveman (MIT, licence in data/seeds/caveman_LICENSE); it is the
RULESET appended to the CAVEMAN system prompt.
"""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = ROOT / "data" / "seeds" / "skill_md.md"

CATEGORY_NORMALIZE_SYSTEM = {
    "debug": (
        "You are a senior engineer writing a debug-session walkthrough for a peer. "
        "Given a GitHub issue + the patch that fixed it, write a verbose multi-step debug narrative: "
        "1) reproduce, 2) hypothesize, 3) instrument/inspect, 4) fix, 5) verify. "
        "Quote relevant code from the patch in fenced blocks. 200-450 words. Markdown."
    ),
    "refactor": (
        "You are a senior engineer documenting a refactor for code review. "
        "Given a commit message + before/after code, write a verbose walkthrough: "
        "why this changed, what specifically moved, risk surface, how to verify. "
        "Quote before/after snippets in fenced blocks. 150-350 words. Markdown."
    ),
    "review": (
        "You are a senior engineer writing detailed code-review feedback. "
        "Given a diff hunk + a real review comment, expand into a full review note: "
        "what the issue is, why it matters, suggested fix with example code. "
        "150-300 words. Markdown."
    ),
    "dialogue": (  # rarely used — oasst2 already has source_normal
        "Rewrite this multi-turn technical dialogue verbosely, preserving every turn's intent "
        "and any code blocks byte-exact."
    ),
    "qa": (
        "You are a senior engineer answering a peer's question. Be helpful, complete, clear. "
        "Markdown with code blocks where appropriate. 80-300 words."
    ),
}

CAVEMAN_INSTRUCTIONS = (
    "You are a rewriting engine. Take the input answer and rewrite it in caveman-mode "
    "(level: full) per the rules below. CRITICAL invariants:\n"
    "  - All code blocks (between ``` fences) pass through BYTE-EXACT. Do NOT change "
    "identifiers, whitespace, or syntax inside code fences.\n"
    "  - All quoted error strings, API names, function names, CLI commands: exact.\n"
    "  - Drop articles (a/an/the), filler (just/really/basically/actually/simply), "
    "pleasantries, hedging. Fragments OK. Pattern: [thing] [action] [reason]. [next step].\n"
    "  - Preserve full technical meaning. Brain big, mouth small.\n"
    "  - If the input is a multi-turn dialogue (### user / ### assistant headings), "
    "rewrite ONLY the assistant turns in caveman; keep user turns and ### headings byte-exact.\n"
    "Return ONLY the rewritten answer. No preamble, no commentary, no markdown fence "
    "around the whole output.\n\n"
    "=== RULESET ===\n"
)


def caveman_system() -> str:
    skill = SKILL_MD.read_text(encoding="utf-8") if SKILL_MD.exists() else ""
    return CAVEMAN_INSTRUCTIONS + skill


def _hash_key(origin: str) -> str:
    return sha1(origin.encode("utf-8")).hexdigest()[:16]


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except Exception:
            continue
    return keys
