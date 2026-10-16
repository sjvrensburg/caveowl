"""Merge the caveowl LoRA into Ornith-1.5-35B-A3B (bf16) and check the result tensor by tensor.

Called by merge_and_gguf.sh; not run by the setup. Needs the full bf16 base (~67 GiB) in memory,
so the resident model servers must be down (takedown_servers.sh), as for training.

  --method unsloth (default)  FastModel loads base + adapter, save_pretrained_merged(merged_16bit)
  --method peft               transformers + PeftModel.merge_and_unload on the CPU: the fallback if
                              the Unsloth save leaves the routed experts unmerged (check 3 below)

After the save, three checks, each fatal:
  1. Tensor names: merged index == base index. The base also carries the MTP head (`mtp.*`,
     785 tensors) that transformers' Qwen3_5MoeForConditionalGeneration does not load, so a
     transformers-side save drops it; it is copied byte-exact from the base shards into
     model-mtp.safetensors. Without it convert_hf_to_gguf.py writes no nextn layer and the GGUF's
     block_count differs from the stock GGUF's (41), which would confound the A/B.
  2. Untouched tensors are byte-identical to the base: the vision tower (the stock mmproj must
     keep matching), the router, in_proj_a/b, norms, embeddings, lm_head.
  3. Targeted tensors CHANGED: attention q/k/v/o, DeltaNet in_proj_qkv/in_proj_z/out_proj, the
     shared expert, and (if the adapter has them) the routed experts' gate_up_proj / down_proj.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CHECK_SAME = ["model.visual.blocks.0.attn.qkv.weight", "model.visual.merger.linear_fc2.weight",
              "model.language_model.layers.0.mlp.gate.weight",
              "model.language_model.layers.0.linear_attn.in_proj_a.weight",
              "model.language_model.layers.0.linear_attn.A_log",
              "model.language_model.embed_tokens.weight", "lm_head.weight",
              "model.language_model.layers.3.self_attn.q_norm.weight"]
CHECK_CHANGED = ["model.language_model.layers.3.self_attn.q_proj.weight",
                 "model.language_model.layers.3.self_attn.o_proj.weight",
                 "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                 "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                 "model.language_model.layers.0.linear_attn.out_proj.weight",
                 "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"]
CHECK_EXPERTS = ["model.language_model.layers.0.mlp.experts.gate_up_proj",
                 "model.language_model.layers.0.mlp.experts.down_proj"]


def index(d: Path) -> dict[str, str]:
    p = d / "model.safetensors.index.json"
    if p.exists():
        return json.loads(p.read_text())["weight_map"]
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def tensor(d: Path, idx: dict[str, str], name: str):
    from safetensors import safe_open
    with safe_open(str(d / idx[name]), "pt") as f:
        return f.get_tensor(name)


def merge_unsloth(base: Path, adapter: Path, out: Path, tok) -> None:
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
    import unsloth  # noqa: F401
    from unsloth import FastModel
    import torch
    from owl_load import use_pread
    use_pread()  # else the load holds a second, CPU copy of the checkpoint (owl_load.py)
    model, _ = FastModel.from_pretrained(model_name=str(adapter), max_seq_length=4096,
                                         dtype=torch.bfloat16, load_in_4bit=False)
    model.save_pretrained_merged(str(out), tok, save_method="merged_16bit")


def merge_peft(base: Path, adapter: Path, out: Path, tok) -> None:
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5MoeForConditionalGeneration
    from owl_load import use_pread
    use_pread()  # mmap would hold the checkpoint twice in CPU RAM (owl_load.py)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16,
                                                               device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out), max_shard_size="5GB")
    tok.save_pretrained(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=["unsloth", "peft"], default="unsloth")
    ap.add_argument("--check-only", action="store_true", help="skip the merge, re-run checks")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import owl_data as D
    tok, _ = D.load_tokenizer(args.base, None, Path(__file__).resolve().parent /
                              "template" / "ornith-gguf-chat-template.jinja")
    if not args.check_only:
        if args.out.exists():
            sys.exit(f"{args.out} exists; move it away first")
        (merge_unsloth if args.method == "unsloth" else merge_peft)(args.base, args.adapter,
                                                                    args.out, tok)
        # The vendor's non-weight files the converter and llama-server may read.
        for f in ("chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
            if (args.base / f).exists() and not (args.out / f).exists():
                shutil.copy(args.base / f, args.out / f)
        # The GGUF must embed the template the stock GGUF embeds (the GGUF one, not the HF one).
        (args.out / "chat_template.jinja").write_text(tok.chat_template)

    bidx, midx = index(args.base), index(args.out)
    missing = sorted(set(bidx) - set(midx))
    extra = sorted(set(midx) - set(bidx))
    if extra:
        sys.exit(f"merged checkpoint has tensors the base lacks: {extra[:10]}")
    if missing and not all(k.startswith("mtp.") for k in missing):
        sys.exit(f"merged checkpoint lacks non-MTP tensors: {[k for k in missing if not k.startswith('mtp.')][:10]}")
    if missing:
        from safetensors import safe_open
        from safetensors.torch import save_file
        by_file = defaultdict(list)
        for k in missing:
            by_file[bidx[k]].append(k)
        tensors = {}
        for fn, keys in by_file.items():
            with safe_open(str(args.base / fn), "pt") as f:
                for k in keys:
                    tensors[k] = f.get_tensor(k)
        save_file(tensors, str(args.out / "model-mtp.safetensors"), metadata={"format": "pt"})
        ip = args.out / "model.safetensors.index.json"
        if not ip.exists():  # single-file save: make it an indexed one
            ip.write_text(json.dumps({"metadata": {}, "weight_map": midx}))
        data = json.loads(ip.read_text())
        data["weight_map"].update({k: "model-mtp.safetensors" for k in missing})
        ip.write_text(json.dumps(data, indent=2))
        print(f"copied {len(missing)} MTP tensors byte-exact from the base")
        midx = index(args.out)

    import torch
    bad = []
    for n in CHECK_SAME:  # MTP tensors are copied byte-exact above; re-comparing them checked nothing
        if not torch.equal(tensor(args.base, bidx, n), tensor(args.out, midx, n)):
            bad.append(f"changed but should not: {n}")
    adapter_cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    has_experts = bool(adapter_cfg.get("target_parameters"))
    for n in CHECK_CHANGED + (CHECK_EXPERTS if has_experts else []):
        a, b = tensor(args.base, bidx, n), tensor(args.out, midx, n)
        if a.shape != b.shape:
            bad.append(f"shape changed: {n} {tuple(a.shape)} -> {tuple(b.shape)}")
        elif torch.equal(a, b):
            bad.append(f"UNCHANGED but targeted: {n}")
        else:
            rel = ((b.float() - a.float()).norm() / a.float().norm()).item()
            print(f"  merged delta {n}: rel {rel:.2e}")
    print(f"adapter target_parameters (routed experts): {adapter_cfg.get('target_parameters')}")
    if bad:
        sys.exit("merge check FAILED:\n  " + "\n  ".join(bad) +
                 ("\n  (experts unmerged? retry with --method peft)" if any("experts" in x for x in bad) else ""))
    print("merge checks passed")


if __name__ == "__main__":
    main()
