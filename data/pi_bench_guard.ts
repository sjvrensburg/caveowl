// pi_bench_guard.ts: the only extension a bench pi run loads (pi -ne -e <this>).
//
// Bench runs (data/build_skill_anchor.py, eval/skill_gate.py) drive real pi with a real shell
// against a candidate model. They must not spend money, send anything, touch services or write
// outside the run's scratch directory. The production exec-guardrail.ts is NOT loaded here: it
// sends a Signal message on every block. This one is stricter, silent and logs nothing.
// A blocked command ends the run (terminate) and the judge rejects it as `guard_block`.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DENY: [string, RegExp][] = [
	["sudo", /\bsudo\b/],
	["sending a message", /\b(signal-send|signal-cli|sendmail|mail\s|voice_message|agentmail)\b/],
	["service control", /\bsystemctl\b(?!\s+--user\s+(status|list-timers|list-units|is-active|--failed|show)\b)(?!\s+(status|list-timers|list-units|is-active|--failed|show)\b)/],
	["process kill", /\b(kill|pkill|killall)\b/],
	["docker other than ps/images", /\bdocker\b(?!\s+(ps|images)\b)/],
	["coffee scan / address setup (writes the DB, drives Camofox)", /scan_specials|setup_address/],
	["marker / podcast / deep research", /marker_run|marker_single|make_podcast|launch\.sh|poll_and_notify|vibevoice/],
	["package install", /\b(pip|pip3|dnf|npm|pnpm|yarn)\s+(install|add|i)\b/],
	// Any uv or pip at all: `uv run` inside a project syncs that project's venv (it rebuilt
	// this repo's .venv once, 2026-10-12). Bench cwds are outside projects now; this is the
	// second fence.
	["uv / pip", /(^|[\s;&|(])(uv|uvx|pip3?|python3?\s+-m\s+pip)(\s|$)/],
	["a secrets env file", /\.env\b/],
	["a non-GET HTTP request", /\bcurl\b.*(\s-X\s*(POST|PUT|DELETE|PATCH)|\s(-d|--data\S*|-F|--form|-T|--upload-file)\s)/],
	["wget", /\bwget\b/],
	["git push / remote change", /\bgit\s+(push|remote|commit)\b/],
	["a key store", /\bsecret-tool\b|\bkeyring\b|\bgpg\b|\bpass\s+(show|ls)\b/],
	["a llama model port other than read-only health", /localhost:(80\d\d|81\d\d)(?!\/health)|127\.0\.0\.1:(80\d\d|81\d\d)(?!\/health)/],
	["rm outside the scratch dir", /\brm\b[^|;&\n]*\s(\/|~)/],
	["shutdown/reboot", /\b(shutdown|reboot|poweroff|halt)\b/],
	["crontab / at", /\b(crontab|at\s+now)\b/],
	// Added for eval/skill_gate_wide.py: eli5/alpha skills reach alphaXiv through feynman, and
	// preview's procedure ends in xdg-open (a window on the operator's desktop).
	["feynman (paper APIs)", /\bfeynman\b/],
	["opening a desktop window", /\b(xdg-open|gio\s+open)\b/],
];

// Machine-specific deny patterns (where THIS box keeps its API keys and per-service env files)
// live outside the repo: ~/.config/caveowl/bench-guard-extra.txt, one JavaScript regex per line,
// `#` for comments. Missing file = generic patterns only. Read once, at load.
const EXTRA = resolve(homedir(), ".config/caveowl/bench-guard-extra.txt");
if (existsSync(EXTRA)) {
	for (const line of readFileSync(EXTRA, "utf8").split("\n")) {
		const t = line.trim();
		if (t && !t.startsWith("#")) DENY.push(["a local deny pattern (bench-guard-extra.txt)", new RegExp(t)]);
	}
}

export default function benchGuard(pi: ExtensionAPI) {
	pi.on("tool_call", async (event, ctx) => {
		try {
			const input = (event.input as any) ?? {};
			const cwd = (ctx as any)?.cwd ?? process.cwd();
			if (event.toolName === "write" || event.toolName === "edit") {
				const p = resolve(cwd, String(input.path ?? "").replace(/^~(?=\/|$)/, homedir()));
				if (!(p === cwd || p.startsWith(cwd + "/"))) {
					return { block: true, terminate: true, reason: `bench-guard: ${event.toolName} outside the scratch dir is refused in a bench run.` };
				}
				return undefined;
			}
			if (event.toolName !== "bash") return undefined;
			const cmd = String(input.command ?? "");
			for (const [why, re] of DENY) {
				if (re.test(cmd)) {
					return { block: true, terminate: true, reason: `bench-guard: refused (${why}) in a bench run.` };
				}
			}
			return undefined;
		} catch {
			return { block: true, terminate: true, reason: "bench-guard: internal error, refusing." };
		}
	});
}
