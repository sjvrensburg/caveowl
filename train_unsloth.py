"""caveowl: bf16 LoRA of Ornith-1.5-35B-A3B toward cavegemma's terse "caveman" style, with Unsloth.

Same data mix and hyperparameters as cavegemma v3 (../cavegemma-e4b, live on :8092 as E4B):
caveman text pairs + tool slice (~25%) + self-distilled MBPP code anchors (~20%) + self-distilled
pi skill trajectories (x2), 1 epoch, lr 1e-4, r16/a32. What is new here and why:

  - Unsloth, not plain peft. ../unsloth-probe (2026-10-12) showed Unsloth + bitsandbytes work on
    gfx1151 (ROCm 7.13 torch from ~/venvs/torch-rocm), ~2.9x plain peft on Qwen3-0.6B with the
    same loss curve, and on a tiny qwen35moe: loss matches transformers and CPU fp32, LoRA on the
    routed experts trains. The cavegemma README's "Unsloth does not run here" was never tested.
  - FastModel, loading the full Qwen3_5MoeForConditionalGeneration (the class config.json
    names), not a text-only CausalLM: the adapter's module paths then match the checkpoint 1:1,
    so the merged save keeps the stock layout (vision tower included) and llama.cpp converts it
    like the stock model. LoRA goes on the text decoder only (config.toml [lora] has the
    verified names); finetune_vision_layers=False and the leaf names keep the tower untouched,
    so the stock mmproj GGUF stays valid.
  - Masking for the Qwen-style template Ornith's GGUF serves, thinking OFF: owl_data.py.
  - transformers.Trainer with our own collator (labels = -100 outside the completion mask)
    instead of TRL's SFTTrainer: the mask is built here, explicitly, and no trainer-side
    re-tokenisation or completion_only heuristic can change it.
  - Over-length rows are DROPPED, not truncated (a cut code fence teaches the wrong thing).

Run (venv: ~/venvs/unsloth-probe, recipe in ../unsloth-probe/RECIPE.sh):
    ~/venvs/unsloth-probe/bin/python train_unsloth.py --config config.toml --prepare-only
    bash takedown_servers.sh       # bf16 LoRA is ~80-85 GiB: every resident server must be down
    ~/venvs/unsloth-probe/bin/python train_unsloth.py --config config_smoke.toml
    ~/venvs/unsloth-probe/bin/python train_unsloth.py --config config.toml
    bash restore_servers.sh
Resumes from the newest checkpoint-* in output_dir (--no-resume to start over).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import owl_data as D  # noqa: E402


def rpath(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (ROOT / q).resolve()


def stats(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label:8s} n=0")
        return
    n = sorted(len(x["input_ids"]) for x in rows)
    loss = sum(sum(x["completion_mask"]) for x in rows)
    print(f"  {label:8s} n={len(n):5d}  tokens median={n[len(n) // 2]:5d} p90={n[int(len(n) * .9)]:5d} "
          f"max={n[-1]:5d} total={sum(n):,}  loss-bearing={loss:,} ({loss / max(1, sum(n)):.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true",
                    help="render + tokenise + mask every row, print masked samples and token "
                         "stats; tokenizer and template only, loads no weights")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    mc, lc, dc, tc = cfg["model"], cfg["lora"], cfg["data"], cfg["train"]
    mc["base"] = str(rpath(mc["base"]))  # configs write ~/models/...; expand once here
    max_len = mc["max_seq_length"]
    tok, tok_src = D.load_tokenizer(mc["base"], rpath(mc.get("tokenizer_fallback", "")) if
                                    mc.get("tokenizer_fallback") else None,
                                    rpath(mc["chat_template"]))
    print(f">>> tokenizer from {tok_src}, chat template {mc['chat_template']}, "
          f"max_seq_length={max_len}")
    print(f"    thinking-off assistant prefix (masked): {D.ASSIST_OPEN!r}")

    # ---- text pairs (reused from cavegemma)
    train = D.build_text_split(D.read_jsonl(rpath(dc["train_path"])), tok, dc["format"],
                               dc["instruction"], max_len, "text train")
    ev = D.build_text_split(D.read_jsonl(rpath(dc["eval_path"])), tok, dc["format"],
                            dc["instruction"], max_len, "text eval")
    n_text = len(train)
    parts = {"text": list(train)}

    # ---- tool slice (reused from cavegemma; rendered through Ornith's template)
    tool_train, tool_ev = [], []
    tool_path = rpath(dc["tool_path"])
    if dc.get("tool_share", 0) > 0:
        if not tool_path.exists():
            sys.exit(f"tool_share > 0 but {tool_path} is missing")
        tool_rows = D.read_jsonl(tool_path)
        tool_train, tool_ev = D.build_tool_splits(tool_rows, tok, max_len,
                                                  dc.get("tool_eval_frac", 0.10), n_text,
                                                  dc["tool_share"])
        _, n_bool = D.normalise_args([m for r in tool_rows for m in r["messages"]])
        print(f"    tool args rendered json-style (bool/None -> true/false/null): {n_bool}")
        parts["tool"] = tool_train

    # ---- MBPP anchors (self-distilled from stock Ornith)
    missing = []
    anchor_path = rpath(dc["anchor_path"])
    if dc.get("anchor_share", 0) > 0:
        if anchor_path.exists():
            anchor_train, _ = D.build_tool_splits(D.read_jsonl(anchor_path), tok, max_len, 0.0,
                                                  n_text, dc["anchor_share"], label="anchor")
            parts["anchor"] = anchor_train
        else:
            missing.append(f"{anchor_path} (run data/build_anchor.py)")

    # ---- pi skill trajectories (self-distilled from stock Ornith), repeated
    # skill_path: one file or a list (run 2 adds data/out/skill_traj_run2.jsonl beside run 1's).
    sp = dc["skill_path"]
    skill_paths = [rpath(x) for x in (sp if isinstance(sp, list) else [sp])]
    skill_path = skill_paths[0]
    skill_rows = []
    if dc.get("skill_share", 0) > 0:
        if all(x.exists() for x in skill_paths):
            skill_rows = [r for x in skill_paths for r in D.read_jsonl(x)]
            skill_train, _ = D.build_tool_splits(skill_rows, tok, max_len, 0.0, n_text,
                                                 dc["skill_share"], label="skill")
            fams = Counter(r.get("family") for r in skill_rows)
            rep = int(dc.get("skill_repeat", 1))
            print(f"    skill families {dict(fams)}; repeated x{rep} -> {len(skill_train) * rep}")
            parts["skill"] = skill_train * rep
        else:
            missing.append(f"{[str(x) for x in skill_paths if not x.exists()]} "
                           "(run data/build_skill_anchor.py)")

    # ---- external Qwen3.6 pi sessions (data/build_synthtraces.py), caveman-rewritten answers
    # (cavegemma's synthesize_batch.py --tool-corpus, Haiku). General multi-step tool loops on
    # the student's base distribution; no skill reads (see build_synthtraces.py).
    if dc.get("ext_share", 0) > 0:
        ext_path = rpath(dc["ext_path"])
        if ext_path.exists():
            ext_train, _ = D.build_tool_splits(D.read_jsonl(ext_path), tok, max_len, 0.0, n_text,
                                               dc["ext_share"], label="ext")
            parts["ext"] = ext_train
        else:
            missing.append(f"{ext_path} (run data/build_synthtraces.py + the Haiku rewrite)")

    # ---- bounded tool calls (data/build_bound_anchor.py, run 4): stock's own calls that carry
    # a user-stated bound/flag, at the served sampler. Runs 1-3 missed sampler-check 27-29/30.
    if dc.get("bound_share", 0) > 0:
        bound_path = rpath(dc["bound_path"])
        if bound_path.exists():
            bound_train, _ = D.build_tool_splits(D.read_jsonl(bound_path), tok, max_len, 0.0, n_text,
                                                 dc["bound_share"], label="bound")
            parts["bound"] = bound_train
        else:
            missing.append(f"{bound_path} (run data/build_bound_anchor.py)")

    # ---- thinking-on rows (data/build_think_anchor.py, run 5): stock's reasoning, closed, then the
    # caveman answer. Run 4 left its answer inside an unclosed think block on 26/32 thinking-on
    # prompts (eval/think_on.py).
    if dc.get("think_share", 0) > 0:
        think_path = rpath(dc["think_path"])
        if think_path.exists():
            think_train, _ = D.build_tool_splits(D.read_jsonl(think_path), tok, max_len, 0.0, n_text,
                                                 dc["think_share"], label="think")
            parts["think"] = think_train
        else:
            missing.append(f"{think_path} (run data/build_think_anchor.py)")

    if missing and not args.prepare_only:
        sys.exit("missing training data: " + "; ".join(missing))
    for m in missing:
        print(f"  (prepare-only: {m} not built yet, skipped)")

    # ---- masked samples: one text, one tool, one skill row
    if args.prepare_only:
        rng = random.Random(0)
        if train:
            D.show_masked(tok, rng.choice(train), "text pair")
        tool_rows = D.read_jsonl(tool_path) if tool_path.exists() else []
        conv = sorted((r for r in tool_rows if r.get("category") == "tool_conv"),
                      key=lambda r: (sum(m["role"] == "tool" for m in r["messages"]) < 2,
                                     len(json.dumps(r))))
        for r in conv[:1] + [r for r in tool_rows if r.get("category") == "tool_call"][:1]:
            ex = D.tokenize_chat(tok, r["messages"], r.get("tools"))
            if ex:
                D.show_masked(tok, ex, f"tool {r.get('category')} {r.get('origin')}")
        if skill_rows:
            r = min(skill_rows, key=lambda r: len(json.dumps(r)))
            ex = D.tokenize_chat(tok, r["messages"], r.get("tools"))
            if ex:
                D.show_masked(tok, ex, f"skill {r.get('family')} {r['key']}", limit=6000)

    train = [x for rows in parts.values() for x in rows]
    if tc_pre := cfg["train"].get("smoke_longest"):
        # Memory smoke only: train on the N longest rows, so the peak is measured at the longest
        # sequence the real run will see (the Trainer's random order would usually miss it).
        train = sorted(train, key=lambda x: -len(x["input_ids"]))[:int(tc_pre)]
        print(f">>> smoke_longest: {len(train)} longest rows, "
              f"{len(train[0]['input_ids'])}..{len(train[-1]['input_ids'])} tokens")
    print(">>> token stats (train unless noted)")
    for k, rows in parts.items():
        stats(k, rows)
    stats("eval-txt", ev)
    stats("eval-tool", tool_ev)
    stats("ALL", train)
    steps = -(-len(train) // (tc["per_device_train_batch_size"] * tc["gradient_accumulation_steps"]))
    print(f"    {len(train)} train rows -> {steps} optimiser steps/epoch at "
          f"{tc['per_device_train_batch_size']}x{tc['gradient_accumulation_steps']}")
    if args.prepare_only:
        return
    if not train:
        sys.exit("empty training set")

    # ------------------------------------------------------------------ model
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", mc.get("moe_backend", "grouped_mm"))
    import unsloth  # noqa: F401  (patches transformers; must come before it)
    from unsloth import FastModel
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    if not torch.cuda.is_available():
        sys.exit("torch.cuda.is_available() is False: wrong venv? (~/venvs/unsloth-probe)")
    if not (Path(mc["base"]) / "model.safetensors.index.json").exists():
        sys.exit(f"no weights at {mc['base']} yet")
    print(f">>> device {torch.cuda.get_device_name(0)}; loading {mc['base']} "
          f"(bf16, load_in_4bit={mc['load_in_4bit']}, MoE backend "
          f"{os.environ['UNSLOTH_MOE_BACKEND']})")
    # pread, not mmap: with mmap the load holds a CPU copy of the whole checkpoint next to the
    # GPU one and the first smoke was OOM-killed (owl_load.py has the measurements). device_map
    # is pinned anyway, and a split model is refused below.
    from owl_load import use_pread
    use_pread()
    model, _processor = FastModel.from_pretrained(
        model_name=mc["base"], max_seq_length=max_len, dtype=getattr(torch, mc["dtype"]),
        load_in_4bit=mc["load_in_4bit"], full_finetuning=False, device_map={"": 0})
    off = sorted({str(p.device) for p in model.parameters()} - {"cuda:0"})
    if off:
        sys.exit(f"parameters on {off} after load: refusing to train with a split model")
    print(f"    loaded {type(model).__name__}; GTT allocated "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    model = FastModel.get_peft_model(
        model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"],
        target_modules=lc["target_modules"],
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing=tc["use_gradient_checkpointing"],
        random_state=tc["seed"], max_seq_length=max_len)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_exp = sum(k for n, k in trainable if ".experts." in n)
    n_vis = sum(k for n, k in trainable if "visual" in n)
    print(f"    trainable {sum(k for _, k in trainable) / 1e6:.1f}M params "
          f"(routed-expert LoRA {n_exp / 1e6:.1f}M, vision {n_vis})")
    if n_vis:
        sys.exit("LoRA reached the vision tower; refusing (the stock mmproj must keep matching)")
    if n_exp == 0:
        print("!!! no LoRA on the routed experts: Unsloth's MoE path did not attach. Training "
              "continues on attention/DeltaNet/shared-expert only; say so in the gate notes.")

    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        pad = tok.pad_token_id
        ids = [x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in batch]
        lab = [[t if m else -100 for t, m in zip(x["input_ids"], x["completion_mask"])]
               + [-100] * (n - len(x["input_ids"])) for x in batch]
        att = [[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

    targs = TrainingArguments(
        output_dir=str(rpath(tc["output_dir"])),
        num_train_epochs=tc["num_train_epochs"],
        max_steps=tc.get("max_steps", -1),
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        # transformers 5.x: warmup_steps takes a float in [0, 1) as a ratio (no warmup_ratio).
        warmup_steps=float(tc["warmup_ratio"]),
        weight_decay=tc["weight_decay"],
        optim=tc["optim"],
        bf16=tc["bf16"],
        gradient_checkpointing=False,  # Unsloth's, set in get_peft_model
        logging_steps=tc["logging_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc.get("save_steps", 500),
        save_total_limit=tc["save_total_limit"],
        eval_strategy=tc["eval_strategy"] if (ev or tool_ev) else "no",
        report_to=tc["report_to"],
        seed=tc["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    eval_ds = None
    if targs.eval_strategy != "no":
        eval_ds = ({"text": Dataset.from_list(ev), "tool": Dataset.from_list(tool_ev)}
                   if ev and tool_ev else Dataset.from_list(ev or tool_ev))
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(train),
                      eval_dataset=eval_ds, data_collator=collate)

    out_dir = Path(targs.output_dir)
    resume = (not args.no_resume and out_dir.exists()
              and any(p.name.startswith("checkpoint-") for p in out_dir.iterdir()))
    if resume:
        print(f"    resuming from the newest checkpoint under {out_dir}")
    print(">>> training")
    trainer.train(resume_from_checkpoint=resume or None)
    print(f"    peak GTT allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
    if tc["save_strategy"] == "no":
        print(">>> smoke run: nothing saved")
        return
    print(f">>> saving adapter -> {out_dir}")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    if eval_ds is not None:
        metrics = trainer.evaluate()
        (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
    print(">>> done. Next: bash merge_and_gguf.sh (servers still down: it needs the bf16 base)")


if __name__ == "__main__":
    main()
