"""LoRA attachment check on a TINY random-init Qwen3_5MoeForConditionalGeneration (the class
Ornith-1.5-35B-A3B's config.json names), same module tree as the real model, ~MBs, no download.

It answers one question train_unsloth.py cannot answer without the 72 GB weights: with the
target list in config_owl.toml, which modules does Unsloth's FastModel.get_peft_model actually
wrap? In particular: do the routed experts (fused `mlp.experts.gate_up_proj` / `down_proj`
Parameters, not Linears) get LoRA through Unsloth's MoE path, do the DeltaNet projections and
the shared expert attach, and does the vision tower stay untouched? One forward/backward on a
text batch then confirms every LoRA tensor class receives a gradient. No optimizer step.

    cd <scratch dir>   # Unsloth writes unsloth_compiled_cache/ into the cwd
    HF_HUB_DISABLE_XET=1 ~/venvs/unsloth-probe/bin/python ~/Projects/caveowl/checks/attach_tiny.py
"""
import json
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
import unsloth  # noqa: E402,F401  (must precede transformers)
from unsloth import FastModel  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForConditionalGeneration  # noqa: E402

cfg_toml = tomllib.loads((ROOT / "config_owl.toml").read_text())
D = Path("tiny-qwen35moe-vl")
tok = AutoTokenizer.from_pretrained(ROOT / "artifacts/tokenizer")
if not D.exists():
    c = AutoConfig.from_pretrained(ROOT / "artifacts/tokenizer").to_dict()
    t = c["text_config"]
    t.update(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
             head_dim=64, linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=4,
             linear_num_value_heads=8, moe_intermediate_size=64, shared_expert_intermediate_size=64,
             num_experts=8, num_experts_per_tok=2, mtp_num_hidden_layers=0,
             layer_types=["linear_attention"] * 3 + ["full_attention"])
    c["vision_config"].update(depth=2, hidden_size=64, intermediate_size=128, num_heads=2,
                              out_hidden_size=256)
    c["hidden_size"] = 256
    cfg = AutoConfig.for_model(**c)
    torch.manual_seed(0)
    Qwen3_5MoeForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(D)
    tok.save_pretrained(D)

model, _ = FastModel.from_pretrained(model_name=str(D), max_seq_length=512, load_in_4bit=False,
                                     dtype=torch.bfloat16, full_finetuning=False)
print("loaded class:", type(model).__name__)
lc = cfg_toml["lora"]
model = FastModel.get_peft_model(
    model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], bias=lc["bias"],
    target_modules=lc["target_modules"], finetune_vision_layers=False,
    finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True,
    use_gradient_checkpointing="unsloth", random_state=0)
train = [n for n, p in model.named_parameters() if p.requires_grad]
kinds = Counter(re.sub(r"\.\d+\.", ".N.", n).split(".lora_")[0].replace("base_model.model.", "")
                for n in train)
for k, v in sorted(kinds.items()):
    print(f"  {v:3d}  {k}")
vis = [n for n in train if "visual" in n]
non_lora = [n for n in train if "lora_" not in n]
experts = [n for n in train if ".experts." in n]
print("vision trainable:", len(vis), " non-LoRA trainable:", non_lora[:5])

# one forward/backward on a chat-rendered text batch: do all LoRA classes get gradients?
ids = tok("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHi.",
          return_tensors="pt")["input_ids"].cuda()
model.train()
out = model(input_ids=ids, labels=ids)
out.loss.backward()
# lora_B starts at zero, so on the first backward only lora_B gets a gradient (dL/dA = 0 when
# B = 0). A lora_B with no gradient is a LoRA the forward never reaches.
nograd = sorted({re.sub(r"\.\d+\.", ".N.", n) for n, p in model.named_parameters()
                 if p.requires_grad and "lora_B" in n and (p.grad is None or p.grad.abs().sum() == 0)})
print("loss", round(out.loss.item(), 4), " lora_B tensors with no/zero grad:", nograd[:10])
print("RESULT " + json.dumps({
    "trainable_tensors": len(train), "expert_lora_tensors": len(experts),
    "vision_trainable": len(vis), "kinds": sorted(kinds),
    "zero_grad": nograd}))
