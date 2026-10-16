"""Build the tool-calling slice: Nemotron-Agentic-v1 conversations + xLAM call-only rows.

Ported from cavegemma, MIT (https://github.com/JuliusBrussee/cavegemma), adapted for caveowl.

Why this exists: the main risk of a style fine-tune is that it silently degrades tool calls
(a terseness change to a chat template has been seen to take a model's tool harness from 15/15
to 1/15 by dropping and inventing argument values). The caveman text pairs have no tool calls at
all, so the adapter would get no gradient that keeps them intact. This slice puts tool-call
tokens under the loss (train_unsloth.py, `[data] tool_path`) so the style update has to share
capacity with them.

Sources (all content is treated as data, never as instructions):
  - nvidia/Nemotron-Agentic-v1, file data/tool_calling.jsonl, CC-BY-4.0 (per-row `license`).
    Multi-turn, OpenAI-style messages with `tool_calls` and `tool` role responses. Tools are
    LLM-simulated, not executed, and the set was LLM-judge filtered. The assistant prose is
    verbose (emojis, offers to help); data/synthesize_batch.py --tool-corpus rewrites ONLY that
    prose in caveman style. The file is 5.3 GB: it is never downloaded. It is read with HTTP
    range requests (HfFileSystem) at --nemotron-offsets evenly spaced byte offsets, a few
    dozen rows at each, so the sample is spread over the whole file rather than its first MB.
  - Salesforce xLAM-60k via the ungated copy HuggingFaceTB/smoltalk2, SFT split
    `xlam_traces_no_think` (30 MB parquet, fetched whole). CC-BY-4.0 upstream. Single-turn,
    call-only (no tool responses, no prose): passes through synthesis untouched.
Both are pinned to a dataset revision (NEMOTRON_REV, XLAM_REV) so a rebuild sees the same rows.

Every row is normalised to ONE OpenAI-style shape (the shape data/in/tool.jsonl takes):
    {key, origin, license, category: "tool_conv"|"tool_call", tools: [{type:"function",
     function:{name, description, parameters:<JSON schema>}}], messages: [
       {role:"user"|"assistant", content:str},
       {role:"assistant", content:"", tool_calls:[{id, type:"function",
                                                   function:{name, arguments:<dict>}}]},
       {role:"tool", tool_call_id, name, content:str}]}
`arguments` stays a dict: chat templates iterate over it (Ornith's renders one <parameter=K>
per key, and some templates raise on a string). Tool results are strings, because agents such
as pi send them as strings and templates render strings and mappings differently.
Nemotron's `reasoning_content` is dropped (caveowl trains this slice thinking-off) and its
empty system message too. An assistant message carrying both prose and calls is split into a
prose message followed by a call message. That keeps every prose segment in a message of its
own, so data/synthesize_batch.py can send prose and never a call, and it removes any doubt about
order: templates differ on whether a message's calls render before or after its content (one
renders calls first, which would put "Let me check" after the result it introduces). Ornith's
renders content first, so for caveowl the split changes nothing the model sees but the turn
boundary.

Filters, in order, each counted (a row is rejected for the first one it fails):
  bad_structure        roles out of order, consecutive user turns, call/response count
                       mismatch, no tool call, no prose after a tool result (tool_conv only)
  special_tokens       any content, call argument or tool declaration containing chat-template
                       control markup: the `<|` / `|>` delimiters of special tokens (Gemma's
                       `<|turn>`, ChatML's `<|im_start|>`, ...) and every marker in
                       owl_data.MARKERS (Ornith/Qwen `<think>`, `<tool_call>`,
                       `<tool_response>`, `<function=`, `<parameter=`). Text holding these could
                       forge a turn, a call or a result once rendered.
  args_not_dict        call arguments that do not parse to a JSON object
  unknown_tool         a call names a function not declared in `tools`
  unsupported_schema   a declared parameter type that has no JSON-schema mapping (xLAM's
                       `Callable[...]`)
  schema_invalid       required argument missing, unknown key, wrong type, value outside enum
  arg_untraceable      ARGUMENT TRACEABILITY: every string/number argument value (recursing
                       into lists/objects) must appear in a PRECEDING user turn or tool result,
                       case-insensitive, whitespace-collapsed, thousands separators removed,
                       numbers compared numerically. Exempt: booleans, null, empty strings,
                       values listed in the parameter's `enum`, and values equal to the
                       parameter's declared `default` (both come from the tool schema, not
                       from thin air). Values of 3 chars or fewer must match on word
                       boundaries. This is the "don't invent arguments" filter: xLAM's own
                       "released after 2010" -> 20110101 is dropped here.
  not_english          ASCII-letter share < 0.95 or common-English-word share < 0.12
  too_long             tokens of tools + messages > --max-tokens (default: 92% of
                       config.toml's [model] max_seq_length, i.e. 15073 of 16384; the 8% margin
                       leaves room for the chat-template markup this count does not include)
  duplicate            same first user turn (normalised) + same tool-name set
Tool results longer than --max-tool-chars are cut to that length plus "[…truncated]" (a tool
call is never touched), and traceability is checked against the truncated text, i.e. what
the model will actually see.

Token counting: tiktoken's cl100k_base if installed (what the released slice was measured
with), else the model's own tokenizer (owl_data.load_tokenizer on config.toml's [model] base /
tokenizer_fallback), else len(text) // 4. The run prints which one it used. The trainer drops
any row that does not fit max_seq_length with the real tokenizer anyway; this filter only stops
the corpus from being spent on rows that will be dropped later.

Reproducing the released slice: caveowl's tool rows were built with the defaults below except
--max-tokens 3768 (cl100k; 92% of a 4096-token config), which kept 300 Nemotron + 300 xLAM rows,
then rewritten with --fewshot 4 on Sonnet 5.5 and filtered at --max-compression 0.80 (see
data/filter_tool_pairs.py). The default here is caveowl's own 16384-token budget, so a rebuild
may keep longer Nemotron conversations than the release did.

Pipeline (run from the repo root; only step 2 costs money):
    python data/build_tool_corpus.py                                   # -> data/out/tool_corpus.jsonl
    python data/synthesize_batch.py --tool-corpus data/out/tool_corpus.jsonl --fewshot 4 --dry-run
    python data/synthesize_batch.py --tool-corpus data/out/tool_corpus.jsonl --fewshot 4 \\
        --confirm-spend                                                # -> data/out/tool_pairs.jsonl
    python data/filter_tool_pairs.py                                   # -> data/in/tool.jsonl
Small trial:
    python data/build_tool_corpus.py --nemotron 30 --xlam 30 --out /tmp/t.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tomllib
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from owl_data import MARKERS  # noqa: E402  (Ornith/Qwen template markup a row must not hold)

NEMOTRON_REPO = "nvidia/Nemotron-Agentic-v1"
NEMOTRON_REV = "650d590978ca35c8f1ecea2faf136e5fac421b62"  # pinned 2026-10-11
NEMOTRON_FILE = "data/tool_calling.jsonl"
XLAM_REPO = "HuggingFaceTB/smoltalk2"
XLAM_REV = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"  # pinned 2026-10-11
XLAM_FILE = "SFT/xlam_traces_no_think-00000-of-00001.parquet"

TRUNC_MARK = "[…truncated]"
# `<|` / `|>`: the delimiters of special tokens in Gemma- and ChatML-style templates (the
# cavegemma filter), plus every marker caveowl's trainer refuses (owl_data.MARKERS).
SPECIAL_RE = re.compile("|".join([r"<\|", r"\|>"] + [re.escape(m) for m in MARKERS]))
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
EN_WORDS = set("""the a an and or of to in on for with is are was were be been it this that
these those you your i we my me our can could would should will please what which who how
when where why do does did not no have has had from at by as if then than so but about
""".split())


def _hash_key(origin: str) -> str:
    # Same scheme as data/synthesize.py (sha1(origin)[:16]); restated here so this module
    # does not import the synthesis driver.
    return sha1(origin.encode("utf-8")).hexdigest()[:16]


class Reject(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


# --------------------------------------------------------------------------- schema

JSON_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def validate(value, schema: dict, path: str, top: bool = False) -> None:
    """Small JSON-schema subset (so jsonschema is not a dependency): type, enum, required, unknown
    keys (top level always; nested only where additionalProperties is false), array items."""
    if not isinstance(schema, dict):
        return
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        known = [x for x in types if x in JSON_TYPES]
        if known and not any(JSON_TYPES[x](value) for x in known):
            raise Reject("schema_invalid", f"{path}: {type(value).__name__} is not {t}")
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        raise Reject("schema_invalid", f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                raise Reject("schema_invalid", f"{path}: missing required {req!r}")
        if top or schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    raise Reject("schema_invalid", f"{path}: unknown key {k!r}")
        for k, v in value.items():
            if k in props:
                validate(v, props[k], f"{path}.{k}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            validate(v, schema["items"], f"{path}[{i}]")


# --------------------------------------------------------------------------- traceability

def norm_text(s: str) -> str:
    return THOUSANDS_RE.sub("", WS_RE.sub(" ", s.lower())).strip()


class Context:
    """Everything the model has seen from the user and the tools so far."""

    def __init__(self):
        self.text = ""
        self.nums: set[float] = set()

    def add(self, s: str) -> None:
        n = norm_text(s)
        self.text += " \n " + n
        self.nums.update(float(x) for x in NUM_RE.findall(n))

    def has_string(self, v: str) -> bool:
        n = norm_text(v)
        if not n:
            return True
        if len(n) <= 3:
            return re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", self.text) is not None
        return n in self.text

    def has_number(self, v: float) -> bool:
        return float(v) in self.nums


def untraced(value, schema: dict, ctx: Context) -> str | None:
    """First argument value that does not appear in the context, or None."""
    schema = schema if isinstance(schema, dict) else {}
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(schema.get("enum"), list) and value in schema["enum"]:
        return None
    if "default" in schema and value == schema["default"]:
        return None
    if isinstance(value, (int, float)):
        return None if ctx.has_number(value) else repr(value)
    if isinstance(value, str):
        if ctx.has_string(value):
            return None
        if NUM_RE.fullmatch(value.strip().replace(",", "")):
            return None if ctx.has_number(float(value.strip().replace(",", ""))) else repr(value)
        return repr(value)
    if isinstance(value, list):
        for v in value:
            bad = untraced(v, schema.get("items") or {}, ctx)
            if bad:
                return bad
        return None
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k, v in value.items():
            bad = untraced(v, props.get(k) or {}, ctx)
            if bad:
                return f"{k}={bad}"
        return None
    return repr(value)


# --------------------------------------------------------------------------- checks

def is_english(texts: list[str]) -> bool:
    joined = " ".join(texts)
    letters = [c for c in joined if c.isalpha()]
    if not letters:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return False
    words = re.findall(r"[a-z']+", joined.lower())
    if len(words) >= 8 and sum(w in EN_WORDS for w in words) / len(words) < 0.12:
        return False
    return True


def tok_counter(config: dict):
    """(count_fn, name): cl100k if tiktoken is installed, else the model's tokenizer, else
    chars/4. See the module docstring."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s, disallowed_special=()))), "tiktoken cl100k_base"
    except Exception:
        pass
    try:
        from owl_data import load_tokenizer
        m = config.get("model") or {}
        base = Path(m["base"]).expanduser()
        fallback = m.get("tokenizer_fallback")
        fallback = (ROOT / Path(fallback).expanduser()) if fallback else None
        tok, src = load_tokenizer(base, fallback, ROOT / m["chat_template"])
        return (lambda s: len(tok.encode(s, add_special_tokens=False))), f"model tokenizer ({src})"
    except Exception as e:  # no transformers, or no tokenizer on disk yet
        print(f"note: no tokenizer ({type(e).__name__}: {e}); counting len(text) // 4",
              file=sys.stderr)
        return (lambda s: max(1, len(s) // 4)), "len(text) // 4"


def check_row(row: dict, max_tokens: int, count_tokens) -> None:
    """Shared checks for both sources. Raises Reject."""
    msgs, tools = row["messages"], row["tools"]
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    for m in msgs:
        for s in [m.get("content") or ""] + [json.dumps(tc["function"]["arguments"])
                                             for tc in m.get("tool_calls") or []]:
            if SPECIAL_RE.search(s):
                raise Reject("special_tokens")
    if SPECIAL_RE.search(json.dumps(tools, ensure_ascii=False)):
        raise Reject("special_tokens")

    ctx = Context()
    for m in msgs:
        if m["role"] in ("user", "tool"):
            ctx.add(m["content"])
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if not isinstance(fn.get("arguments"), dict):
                raise Reject("args_not_dict")
            decl = by_name.get(fn["name"])
            if decl is None:
                raise Reject("unknown_tool", fn["name"])
            params = decl.get("parameters") or {"type": "object", "properties": {}}
            validate(fn["arguments"], params, fn["name"], top=True)
            props = params.get("properties") or {}
            for k, v in fn["arguments"].items():
                bad = untraced(v, props.get(k) or {}, ctx)
                if bad:
                    raise Reject("arg_untraceable", f"{fn['name']}.{k}={bad}")

    if not is_english([m["content"] for m in msgs if m["role"] in ("user", "assistant")]):
        raise Reject("not_english")
    total = count_tokens(json.dumps(tools, ensure_ascii=False))
    for m in msgs:
        total += count_tokens(m.get("content") or "") + 4
        for tc in m.get("tool_calls") or []:
            total += count_tokens(json.dumps(tc["function"], ensure_ascii=False)) + 4
    if total > max_tokens:
        raise Reject("too_long", str(total))


def truncate_tool(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + TRUNC_MARK


def tool_str(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


# --------------------------------------------------------------------------- Nemotron

def normalize_nemotron(raw: dict, max_tool_chars: int) -> dict:
    tools = []
    for t in raw.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn or not fn.get("name"):
            raise Reject("bad_structure", "tool without function.name")
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        tools.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description") or "",
            "parameters": params}})
    msgs: list[dict] = []
    pending: list[dict] = []  # calls still waiting for a tool response, in order
    n_calls = 0
    for i, m in enumerate(raw.get("messages") or []):
        role = m.get("role")
        if role == "system":
            if (m.get("content") or "").strip():
                raise Reject("bad_structure", "non-empty system prompt")  # none seen; be strict
            continue
        if role == "tool":
            if not pending:
                raise Reject("bad_structure", "tool response without a call")
            tc = pending.pop(0)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                         "content": truncate_tool(tool_str(m.get("content")), max_tool_chars)})
            continue
        if pending:
            raise Reject("bad_structure", "call without a tool response")
        if role == "user":
            if msgs and msgs[-1]["role"] == "user":
                raise Reject("bad_structure", "consecutive user turns")
            c = m.get("content")
            if not isinstance(c, str) or not c.strip():
                raise Reject("bad_structure", "empty user turn")
            msgs.append({"role": "user", "content": c})
        elif role == "assistant":
            content = m.get("content") or ""
            if not isinstance(content, str):
                raise Reject("bad_structure", "non-string assistant content")
            calls = m.get("tool_calls") or []
            if content.strip():
                msgs.append({"role": "assistant", "content": content})
            if calls:
                norm_calls = []
                for j, tc in enumerate(calls):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            raise Reject("args_not_dict")
                    if not isinstance(args, dict):
                        raise Reject("args_not_dict")
                    norm_calls.append({"id": tc.get("id") or f"call_{i}_{j}", "type": "function",
                                       "function": {"name": fn.get("name"), "arguments": args}})
                msgs.append({"role": "assistant", "content": "", "tool_calls": norm_calls})
                pending.extend(norm_calls)
                n_calls += len(norm_calls)
            if not content.strip() and not calls:
                raise Reject("bad_structure", "empty assistant turn")
        else:
            raise Reject("bad_structure", f"role {role!r}")
    if pending:
        raise Reject("bad_structure", "call without a tool response")
    # End on the last prose answer; trailing calls with no answer are cut.
    while msgs and not (msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")):
        msgs.pop()
    if not msgs or msgs[0]["role"] != "user" or not any(m.get("tool_calls") for m in msgs):
        raise Reject("bad_structure", "no tool call")
    seen_tool = False
    prose_after_tool = False
    for m in msgs:
        seen_tool |= m["role"] == "tool"
        prose_after_tool |= seen_tool and m["role"] == "assistant" and not m.get("tool_calls")
    if not prose_after_tool:
        raise Reject("bad_structure", "no prose after a tool result")
    return {"tools": tools, "messages": msgs}


def iter_nemotron(n_offsets: int, rows_per_offset: int, offset_done=lambda k: False):
    """Yield raw rows from evenly spaced byte offsets of the 5.3 GB file (range requests).
    Moves to the next offset as soon as offset_done(k) is true, so nothing past what is
    needed is fetched."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"datasets/{NEMOTRON_REPO}@{NEMOTRON_REV}/{NEMOTRON_FILE}"
    size = fs.info(path)["size"]
    with fs.open(path, "rb", block_size=1 << 20) as f:
        for k in range(n_offsets):
            off = int(size * k / n_offsets)
            f.seek(off)
            if off:
                f.readline()  # discard the partial line we landed in
            for _ in range(rows_per_offset):
                if offset_done(k):
                    break
                line = f.readline()
                if not line:
                    break
                try:
                    yield k, json.loads(line)
                except json.JSONDecodeError:
                    yield k, None


# --------------------------------------------------------------------------- xLAM

XLAM_SCALAR = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
XLAM_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def xlam_param_schema(spec: dict) -> tuple[dict, bool]:
    """xLAM's {description, type: "List[int], optional", default} -> (JSON schema, required)."""
    raw_t = str(spec.get("type") or "")
    parts = [p.strip() for p in raw_t.split(",")]
    base, optional = parts[0], any(p.startswith("optional") for p in parts[1:])
    out: dict = {}
    if spec.get("description"):
        out["description"] = spec["description"]
    if base in XLAM_SCALAR:
        out["type"] = XLAM_SCALAR[base]
    elif re.match(r"(List|list|Tuple|tuple|set|Set)\b", base):
        out["type"] = "array"
        inner = re.match(r"(?:List|Tuple|Set)\[(\w+)\]$", base)
        if inner and inner.group(1) in XLAM_SCALAR:
            out["items"] = {"type": XLAM_SCALAR[inner.group(1)]}
    elif re.match(r"(Dict|dict)\b", base):
        out["type"] = "object"
    else:
        raise Reject("unsupported_schema", raw_t)  # Callable[...], Union[...], Any
    if "default" in spec and spec["default"] not in ("", None):
        out["default"] = spec["default"]
    return out, not optional


def normalize_xlam(raw: dict) -> dict:
    xml = (raw.get("chat_template_kwargs") or {}).get("xml_tools") or []
    if len(xml) != 1:
        raise Reject("bad_structure", "xml_tools")
    body = xml[0].strip()
    if not (body.startswith("<tools>") and body.endswith("</tools>")):
        raise Reject("bad_structure", "xml_tools wrapper")
    tools = []
    for t in json.loads(body[len("<tools>"):-len("</tools>")]):
        fn = t["function"]
        props, req = {}, []
        for pname, spec in (fn.get("parameters") or {}).items():
            schema, required = xlam_param_schema(spec if isinstance(spec, dict) else {})
            props[pname] = schema
            if required:
                req.append(pname)
        params = {"type": "object", "properties": props}
        if req:
            params["required"] = req
        tools.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description") or "", "parameters": params}})
    msgs = raw.get("messages") or []
    if [m["role"] for m in msgs] != ["user", "assistant"]:
        raise Reject("bad_structure", "not user/assistant")
    a = msgs[1]["content"]
    if XLAM_CALL_RE.sub("", a).strip():
        raise Reject("bad_structure", "prose beside calls")
    calls = []
    for j, blob in enumerate(XLAM_CALL_RE.findall(a)):
        try:
            c = json.loads(blob)
        except json.JSONDecodeError:
            raise Reject("args_not_dict")
        if not isinstance(c, dict) or not isinstance(c.get("arguments"), dict):
            raise Reject("args_not_dict")
        calls.append({"id": f"call_{j}", "type": "function",
                      "function": {"name": c.get("name"), "arguments": c["arguments"]}})
    if not calls:
        raise Reject("bad_structure", "no tool call")
    return {"tools": tools, "messages": [
        {"role": "user", "content": msgs[0]["content"]},
        {"role": "assistant", "content": "", "tool_calls": calls}]}


def iter_xlam(seed: int = 42):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(XLAM_REPO, XLAM_FILE, repo_type="dataset", revision=XLAM_REV)
    rows = pq.read_table(p).to_pylist()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)  # spread the sample over the whole split
    for i in order:
        yield i, rows[i]


# --------------------------------------------------------------------------- driver

def dedupe_sig(row: dict) -> str:
    first = next(m["content"] for m in row["messages"] if m["role"] == "user")
    names = sorted(t["function"]["name"] for t in row["tools"])
    return sha1((norm_text(first) + "||" + ",".join(names)).encode()).hexdigest()


def run_source(label, it, normalize, quota, max_scan, max_tokens, count_tokens, seen, out,
               examples):
    counts: Counter[str] = Counter()
    scanned = 0
    for origin, lic, raw in it:
        if len([r for r in out if r["_src"] == label]) >= quota or scanned >= max_scan:
            break
        scanned += 1
        try:
            if raw is None:
                raise Reject("bad_structure", "bad json")
            row = normalize(raw)
            check_row(row, max_tokens, count_tokens)
            sig = dedupe_sig(row)
            if sig in seen:
                raise Reject("duplicate")
            seen.add(sig)
        except Reject as e:
            counts[e.reason] += 1
            if e.detail and len(examples[e.reason]) < int(os.environ.get("TOOL_CORPUS_EXAMPLES", "3")):
                examples[e.reason].append(f"{label}: {e.detail[:160]}")
            continue
        counts["kept"] += 1
        out.append({"_src": label, "key": _hash_key(origin), "origin": origin, "license": lic,
                    "category": "tool_conv" if label == "nemotron" else "tool_call", **row})
    return scanned, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "out" / "tool_corpus.jsonl")
    ap.add_argument("--nemotron", type=int, default=300, help="Nemotron rows to keep")
    ap.add_argument("--xlam", type=int, default=300, help="xLAM rows to keep")
    ap.add_argument("--nemotron-offsets", type=int, default=40,
                    help="evenly spaced byte offsets to read Nemotron rows from")
    ap.add_argument("--scan-factor", type=float, default=10.0,
                    help="stop a source after scanning quota x this many candidates")
    ap.add_argument("--max-tool-chars", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="token budget per row (default 92%% of --config's [model] "
                         "max_seq_length; the release used 3768)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.toml",
                    help="caveowl training config: max_seq_length and the tokenizer fallback")
    args = ap.parse_args()

    with args.config.open("rb") as f:
        config = tomllib.load(f)
    if not args.max_tokens:
        args.max_tokens = int(config["model"]["max_seq_length"] * 0.92)
    count_tokens, tok_name = tok_counter(config)
    seen: set[str] = set()
    out: list[dict] = []
    examples: defaultdict[str, list[str]] = defaultdict(list)
    report = {}

    if args.nemotron:
        max_scan = int(args.nemotron * args.scan_factor)
        per_off = math.ceil(max_scan / args.nemotron_offsets)
        # Cap kept rows per offset so one topical cluster of the file cannot fill the quota.
        per_off_keep = math.ceil(args.nemotron / args.nemotron_offsets)
        kept_at: Counter[int] = Counter()

        def nemotron_rows():
            for k, r in iter_nemotron(args.nemotron_offsets, per_off,
                                      lambda k: kept_at[k] >= per_off_keep):
                before = len(out)
                yield (f"{NEMOTRON_REPO}:{NEMOTRON_FILE}:{(r or {}).get('uuid')}",
                       (r or {}).get("license", "cc-by-4.0"), r)
                kept_at[k] += len(out) - before  # run_source appends before the next pull

        report["nemotron"] = run_source(
            "nemotron", nemotron_rows(), lambda r: normalize_nemotron(r, args.max_tool_chars),
            args.nemotron, max_scan, args.max_tokens, count_tokens, seen, out, examples)

    if args.xlam:
        rows = ((f"{XLAM_REPO}:{XLAM_FILE}:{i}", "cc-by-4.0", r) for i, r in iter_xlam())
        report["xlam"] = run_source(
            "xlam", rows, normalize_xlam, args.xlam, int(args.xlam * args.scan_factor),
            args.max_tokens, count_tokens, seen, out, examples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out:
            r = {k: v for k, v in r.items() if k != "_src"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"max_tokens={args.max_tokens} ({tok_name})  max_tool_chars={args.max_tool_chars}")
    for src, (scanned, counts) in report.items():
        print(f"\n[{src}] scanned {scanned}")
        for k, v in counts.most_common():
            print(f"  {k:22s} {v:5d}  ({v / max(1, scanned) * 100:.1f}%)")
    trunc = sum(1 for r in out for m in r["messages"]
                if m["role"] == "tool" and m["content"].endswith(TRUNC_MARK))
    print(f"\ntool results truncated: {trunc}")
    print("examples of rejections:")
    for reason, ex in examples.items():
        for e in ex:
            print(f"  {reason}: {e}")
    print(f"\nwrote {len(out)} rows -> {args.out}")
    sys.stdout.flush()
    os._exit(0)  # HfFileSystem / HF download threads can keep the interpreter alive at exit


if __name__ == "__main__":
    main()
