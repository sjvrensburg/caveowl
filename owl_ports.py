"""Bench port and live-port refusal, shared by the data builders and the gates.

Every script that talks to a model takes --base; its default is CAVEOWL_BENCH_URL, else
http://127.0.0.1:8110/v1 (the bench port scripts/serve_bench.sh uses; BENCH_PORT there).

refuse_live() stops a script from pointing at a server other people rely on (the gates run
real agent sessions; the builders hammer the server for hours). Which ports count as live is
per machine: CAVEOWL_LIVE_PORTS (comma or space separated), else ~/.config/caveowl/live-ports,
else none.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BENCH_URL = os.environ.get("CAVEOWL_BENCH_URL", "http://127.0.0.1:8110/v1")


def live_ports() -> set[str]:
    raw = os.environ.get("CAVEOWL_LIVE_PORTS")
    if raw is None:
        f = Path.home() / ".config" / "caveowl" / "live-ports"
        raw = f.read_text() if f.exists() else ""
    return {p for p in re.split(r"[\s,]+", raw) if p.isdigit()}


def refuse_live(base: str) -> None:
    m = re.search(r":(\d+)(/|$)", base)
    if m and m.group(1) in live_ports():
        sys.exit(f"refusing {base}: port {m.group(1)} is listed as live (CAVEOWL_LIVE_PORTS or "
                 "~/.config/caveowl/live-ports); serve the model on a bench port")
