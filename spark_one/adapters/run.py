"""Run an eval-only adapter over eval variants with a hard cost cap; resumable.

python -m spark_one.adapters.run --adapter openrouter --model openai/gpt-4o-mini \
    --bank data/generated/bank --split test --limit 300 --max-usd 2 --out runs/eval/gpt4omini.jsonl
python -m spark_one.adapters.run --adapter jev --model jev-latest --max-usd 5 ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..eval.variants import expand
from ..schema import load_items, read_jsonl
from .cost import BudgetExceeded, CostTracker


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["jev", "openrouter"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--primitives", nargs="*")
    a = ap.parse_args(argv)
    root = Path(a.bank)
    items = [it for it in load_items(root / "items.jsonl") if it.split == a.split]
    if a.primitives:
        items = [it for it in items if it.primitive in a.primitives]
    items = random.Random(0).sample(items, min(a.limit, len(items)))
    var = expand(items, perms=not a.no_variants, negations=not a.no_variants)
    budget = CostTracker(a.max_usd)
    if a.adapter == "jev":
        from .jev import JevAdapter

        ad = JevAdapter(budget, a.model)
    else:
        from .openrouter import OpenRouterAdapter

        ad = OpenRouterAdapter(a.model, budget)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    n_err = 0
    with open(out, "a") as f:
        for vid, variant, order, it in var:
            if vid in done:
                continue
            try:
                row = ad.predict(it, root)
            except BudgetExceeded as e:
                print(f"STOP: {e}")
                break
            except Exception as e:
                n_err += 1
                print(f"error on {vid}: {str(e)[:200]}")
                if n_err > 20:
                    break
                continue
            row.update({"id": vid, "base_id": vid.split("~")[0], "variant": variant, "order": order,
                        "eval_only": True, "model": a.model})
            f.write(json.dumps(row) + "\n")
            f.flush()
    s = budget.summary()
    s["errors"] = n_err
    print(json.dumps(s))
    (out.with_suffix(".cost.json")).write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
