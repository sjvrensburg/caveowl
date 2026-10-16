"""Make transformers read safetensors shards with the "pread" backend instead of "mmap".

Why (2026-10-12): the first smoke was killed by the GLOBAL OOM killer 110 s into loading the bf16
checkpoint: 46 GB of CPU anon RSS next to 66 GiB of GTT, and on this box both come out of the
same 128 GB. Measured on a 5.2 GiB random qwen35moe in the same venv: with mmap, anon RSS climbs
to the whole checkpoint size during the load and is only released when the shard handles close,
and transformers (5.17) keeps every shard open until the load ends. It is plain transformers, not
Unsloth; not the device map (all-GPU either way); not glibc's thresholds; not async loading.
With pread the peak is ~2.4 GiB and flat, and the loaded weights are bit-identical.

Call use_pread() once, before any from_pretrained. Used by train_unsloth.py and merge_owl.py.
"""


def use_pread() -> None:
    import transformers.modeling_utils as mu
    if getattr(mu.safe_open, "_owl_pread", False):
        return
    orig = mu.safe_open

    def safe_open(file, framework="pt", device="cpu", backend=None, **kw):
        return orig(file, framework=framework, device=device,
                    backend="pread" if backend in (None, "mmap") else backend, **kw)

    safe_open._owl_pread = True
    mu.safe_open = safe_open
