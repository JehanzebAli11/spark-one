"""Predict eval variants with a Hugging Face model (base Gemma or base + LoRA adapter),
using the prefix-cached backend. Output rows match spark_one.tiny.predict.

python -m spark_one.hf_predict --model google/gemma-4-E2B-it --adapter runs/gemma/final_adapter \
    --bank data/generated/bank --split test --out runs/gemma/test_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .eval.variants import expand
from .prompt import question_text
from .schema import load_items, read_jsonl
from .steps import item_steps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--primitives", nargs="*")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-prefix-cache", action="store_true")
    a = ap.parse_args(argv)
    import torch

    from .hf import HFBackend

    if a.threads:
        torch.set_num_threads(a.threads)
    dtype = {"auto": None, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    root = Path(a.bank)
    items = [it for it in load_items(root / "items.jsonl") if it.split == a.split]
    if a.primitives:
        items = [it for it in items if it.primitive in a.primitives]
    if a.limit:
        items = random.Random(0).sample(items, min(a.limit, len(items)))
    var = expand(items, perms=not a.no_variants, negations=not a.no_variants)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    be = HFBackend(a.model, a.adapter, dtype=dtype, use_prefix_cache=not a.no_prefix_cache)
    with open(out, "a") as f:
        for i, (vid, variant, order, it) in enumerate(var):
            if vid in done:
                continue
            t0 = time.time()
            img = (root / it.image).read_bytes() if it.image else None
            pre, _ = be.prefix(it.state, img)
            q, sp = question_text(it.primitive, it.instructions, it.criteria)
            lps = be.answer_steps(pre, q, sp, [st.labels for st in item_steps(it)])
            f.write(json.dumps({"id": vid, "base_id": vid.split("~")[0], "variant": variant, "order": order,
                                "logp": [[round(x, 5) for x in lp] for lp in lps],
                                "latency_ms": round((time.time() - t0) * 1000, 2), "source": "logprobs"}) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(var)}", flush=True)
    print(f"wrote predictions to {out}")


if __name__ == "__main__":
    main()
