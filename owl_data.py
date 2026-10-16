"""Rendering and loss masking for caveowl: every training row -> (input_ids, completion_mask)
through Ornith's chat template in the exact form llama-server uses at inference.

Imported by train_unsloth.py and data/build_skill_anchor.py. Needs only a tokenizer (no torch,
no weights), so --prepare-only and the anchor converter run anywhere.

THE TEMPLATE AND THE THINKING-OFF FORM (template/ornith-gguf-chat-template.jinja, the template
embedded in the stock GGUF :8088 serves; it differs from the HF repo's chat_template.jinja only in
developer-role handling and scalar tool-argument rendering, not in the markup below):

    <|im_start|>system\n[# Tools ... <tools>{json}...</tools> ... IMPORTANT ...\n\n]SYSTEM<|im_end|>\n
    <|im_start|>user\nUSER<|im_end|>\n
    <|im_start|>assistant\n<think>\nREASONING\n</think>\n\nCONTENT[\n\n]<tool_call>\n<function=NAME>\n
        <parameter=K>\nV\n</parameter>\n</function>\n</tool_call><|im_end|>\n
    <|im_start|>user\n<tool_response>\nRESULT\n</tool_response>[\n<tool_response>...]<|im_end|>\n

Every assistant turn, history included, is rendered WITH a think block; with no reasoning it is
the empty `<think>\n\n</think>\n\n`. The generation prompt with enable_thinking=false (what
pi's `--thinking off` makes llama-server render: chat_template_kwargs.enable_thinking=false) is
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, byte-identical to that empty block. So at
inference the empty think block is PROMPT, prefilled by the server, and the model generates from
the first content character. Training matches that exactly:

    loss OFF  system, tool declarations, user turns, tool results (they are user turns here),
              and each assistant turn's `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
    loss ON   each assistant turn's content, its tool calls (`<tool_call>...</tool_call>`, the
              thing the tool slice and the skill trajectories exist to protect) and its
              closing <|im_end|>, the stop token. The "\n" after <|im_end|> is not trained:
              generation stops at <|im_end|>.

Spans come from prefix differences, as in cavegemma's text path: for assistant message i,
render(msgs[:i], add_generation_prompt, enable_thinking=False) must be a prefix of the full
render, and so must render(msgs[:i+1]). This template renders history position-independently
(it does not drop earlier turns' reasoning the way stock Qwen3 templates do), so this holds for
multi-turn tool trajectories too; a row where it does not hold is refused, not guessed at.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIST_OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Thinking ON (run 5): the generation prompt stops after "<think>\n" and the model writes its own
# reasoning, "</think>", then the answer. Rows whose assistant turn carries reasoning_content
# are rendered this way and the reasoning carries loss too (see tokenize_chat).
ASSIST_OPEN_THINK = "<|im_start|>assistant\n<think>\n"
END = "<|im_end|>"
# Markup a row's own text must not contain, or it could forge a turn, a call or a result (or,
# for <think>, be re-parsed by the template as reasoning and silently moved out of content).
MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>",
           "</tool_call>", "<tool_response>", "</tool_response>", "<function=", "<parameter=")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_tokenizer(base: str | Path, fallback: str | Path | None, template: Path):
    """AutoTokenizer from the model dir (or the fallback copy fetched alone), with the GGUF's
    chat template installed in place of whatever the tokenizer shipped."""
    from transformers import AutoTokenizer
    src = Path(base)
    if not (src / "tokenizer.json").exists() and fallback:
        src = Path(fallback)
    tok = AutoTokenizer.from_pretrained(src)
    tok.chat_template = template.read_text()
    return tok, src


# --------------------------------------------------------------------------- messages

TURN_RE = re.compile(r"^### (user|assistant)\s*$", re.MULTILINE)


def split_turns(text: str) -> list[tuple[str, str]]:
    """cavegemma data/filter.py split_turns, verbatim: `### user` / `### assistant` blocks."""
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return []
    turns = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append((m.group(1), text[m.end():end].strip()))
    return turns


def answer_messages(row: dict) -> list[dict] | None:
    """cavegemma train_peft.answer_messages, unchanged: a question in, a caveman answer out."""
    cat = (row.get("category") or "qa").lower()
    prompt = (row.get("prompt") or "").strip()
    target = row["target"].strip()
    if cat == "dialogue":
        turns = split_turns(target)
        if not turns or turns[0][0] != "user" or turns[-1][0] != "assistant":
            return None
        msgs = [{"role": r, "content": c} for r, c in turns]
        if any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
            return None
        return msgs
    user = prompt
    if row.get("context"):
        user = f"{prompt}\n\n{row['context'].strip()}"
    elif cat == "review":
        diff = (row.get("source") or "").split("\n\nREVIEW COMMENT:", 1)[0].strip()
        if diff:
            user = f"{prompt}\n\n{diff}"
    if not user:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": target}]


def rewrite_messages(row: dict, instruction: str) -> list[dict]:
    return [{"role": "user", "content": f"{instruction}\n\n{row['source']}"},
            {"role": "assistant", "content": row["target"]}]


def normalise_args(msgs: list[dict]) -> tuple[list[dict], int]:
    """Scalar tool arguments as the model should WRITE them. The GGUF template renders a
    non-string scalar with jinja's `|string`, which turns True into "True" and None into
    "None"; llama-server's tool-call parser types parameter values from the schema, and the
    model's own habit (and JSON) is `true`. Pre-render such values with json.dumps so the
    template passes them through as strings: bool -> true/false, None -> null, numbers
    unchanged in effect (3 -> "3"). Containers are left to the template's tojson.
    Returns (messages, number of values changed)."""
    n, out = 0, []
    for m in msgs:
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or tc)
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                new = {}
                for k, v in args.items():
                    if isinstance(v, bool) or v is None:
                        new[k] = json.dumps(v)
                        n += 1
                    elif isinstance(v, (int, float)):
                        new[k] = json.dumps(v)
                    else:
                        new[k] = v
                fn["arguments"] = new
                calls.append({**tc, "function": fn} if "function" in tc else fn)
            m = {**m, "tool_calls": calls}
        out.append(m)
    return out, n


# --------------------------------------------------------------------------- tokenise

def render(tok, msgs, tools=None, gen=False, think=False) -> str:
    return tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                   add_generation_prompt=gen, enable_thinking=think)


def has_marker(msgs: list[dict]) -> bool:
    blob = json.dumps([{k: v for k, v in m.items() if k != "tool_call_id"} for m in msgs])
    return any(mk in blob for mk in MARKERS)


def tokenize_chat(tok, msgs: list[dict], tools: list | None = None,
                  why: Counter | None = None) -> dict | None:
    """input_ids + completion_mask for any row (text pair, tool slice, anchor, skill trajectory).
    Returns None (and counts the reason in `why`) for a row that cannot be masked exactly."""
    def no(reason: str):
        if why is not None:
            why[reason] += 1
        return None

    if has_marker(msgs):
        return no("marker_in_content")
    msgs, _ = normalise_args(msgs)
    for m in msgs:
        if m["role"] == "assistant" and not (str(m.get("content") or "").strip()
                                             or m.get("tool_calls")):
            return no("empty_assistant")
    # Thinking-on rows (run 5): exactly one assistant turn, the last, with reasoning_content.
    # Rendered with enable_thinking=True; its span starts after "<think>\n", so the reasoning,
    # "</think>", the answer and <|im_end|> all carry loss. Everything else is unchanged.
    think = any(m.get("reasoning_content") for m in msgs)
    if think and not (msgs[-1]["role"] == "assistant" and msgs[-1].get("reasoning_content")
                      and sum(m["role"] == "assistant" for m in msgs) == 1):
        return no("think_row_shape")
    if think and any(mk in msgs[-1]["reasoning_content"] for mk in MARKERS):
        return no("marker_in_content")
    opener = ASSIST_OPEN_THINK if think else ASSIST_OPEN
    try:
        text = render(tok, msgs, tools, think=think)
    except Exception:  # template raise_exception (no user turn, misplaced system, ...)
        return no("template_error")
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start = render(tok, msgs[:i], tools, gen=True, think=think)
        end = render(tok, msgs[: i + 1], tools, think=think)
        if not (start.endswith(opener) and text.startswith(start) and text.startswith(end)
                and end.rstrip("\n").endswith(END) and len(end) > len(start)):
            return no("prefix_mismatch")
        spans.append((len(start), len(end.rstrip("\n"))))
    # Every call and result must have survived the render.
    # (The tools header itself mentions <tool_call> in its format instructions; count after it.)
    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs)
    header = render(tok, [{"role": "user", "content": "x"}], tools).split("<|im_start|>user")[0]
    if text.count("<tool_call>") - header.count("<tool_call>") != n_calls:
        return no("calls_dropped")
    if text.count("<tool_response>") != sum(m["role"] == "tool" for m in msgs):
        return no("results_dropped")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    mask = [1 if any(s <= a < e for s, e in spans) else 0 for a, _ in offs]
    # A token that straddles a span edge would mean the trained tokenisation differs from the
    # one the server sees (prompt and completion are tokenised apart there). Refuse those.
    edges = {p for s in spans for p in s}
    if any(a < p < b for a, b in offs for p in edges):
        return no("token_straddles_span")
    if not any(mask):
        return no("no_loss_tokens")
    return {"input_ids": enc["input_ids"], "completion_mask": mask}


def show_masked(tok, ex: dict, label: str, limit: int = 3000) -> None:
    """Print the render with loss-carrying runs wrapped in ⟦ ⟧ (everything else is masked)."""
    out, run, cur = [], [], None
    for t, m in zip(ex["input_ids"], ex["completion_mask"]):
        if m != cur and run:
            s = tok.decode(run)
            out.append(f"⟦{s}⟧" if cur else s)
            run = []
        cur = m
        run.append(t)
    if run:
        s = tok.decode(run)
        out.append(f"⟦{s}⟧" if cur else s)
    text = "".join(out)
    print(f"--- {label}: ⟦loss⟧, rest masked ({sum(ex['completion_mask'])} of "
          f"{len(ex['input_ids'])} tokens carry loss) ---")
    print(text if len(text) <= limit else text[:limit // 2] + "\n[... middle elided ...]\n"
          + text[-limit // 2:])


# --------------------------------------------------------------------------- splits

def build_text_split(rows, tok, fmt, instruction, max_len, label):
    out, seen, dropped = [], set(), Counter()
    for row in rows:
        msgs = answer_messages(row) if fmt == "answer" else rewrite_messages(row, instruction)
        if msgs is None:
            dropped["unusable"] += 1
            continue
        sig = json.dumps(msgs, sort_keys=True)
        if sig in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(sig)
        ex = tokenize_chat(tok, msgs, why=dropped)
        if ex is None:
            continue
        if len(ex["input_ids"]) > max_len:
            dropped["too_long"] += 1
            continue
        out.append(ex)
    print(f"  {label}: kept {len(out)} / {len(rows)}  dropped {dict(dropped)}")
    return out


def _bucket(key: str) -> float:
    return int(sha1(key.encode()).hexdigest()[:15], 16) / 16 ** 15


def build_tool_splits(rows, tok, max_len, eval_frac, n_text_train, share, label="tool"):
    """cavegemma's semantics: deterministic held-out eval by key hash; train sized so these rows
    are `share` of (these + text) train rows, or all of them if there are fewer."""
    tr, ev, dropped = [], [], Counter()
    for row in sorted(rows, key=lambda r: sha1(("pick:" + r["key"]).encode()).hexdigest()):
        ex = tokenize_chat(tok, row["messages"], row.get("tools") or None, why=dropped)
        if ex is None:
            continue
        if len(ex["input_ids"]) > max_len:
            dropped["too_long"] += 1
            continue
        ex["_category"] = row.get("category")
        (ev if _bucket(row["key"]) < eval_frac else tr).append(ex)
    want = round(share / (1 - share) * n_text_train) if n_text_train and share < 1 else len(tr)
    tr = tr[:want]
    cats = Counter(x["_category"] for x in tr)
    for x in tr + ev:
        x.pop("_category")
    got = len(tr) / max(1, len(tr) + n_text_train)
    print(f"  {label}: train {len(tr)} {dict(cats)}  eval {len(ev)}  dropped {dict(dropped)}  "
          f"share vs text {got:.0%} (target {share:.0%})")
    return tr, ev
