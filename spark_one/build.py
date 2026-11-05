"""CLI: build the procedural question bank.

python -m spark_one.build --out data/generated/bank --n 1500 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .bank import build_task_items, write_bank
from .generators import ILLUSIONS, REGISTRY


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/bank")
    ap.add_argument("--n", type=int, default=1500, help="scenes per generator per language")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="subset of generator names")
    ap.add_argument("--illusions", action="store_true", help="build the illusion eval set instead")
    ap.add_argument("--sample", type=int, default=0, help="also write N random items to sample.jsonl")
    a = ap.parse_args(argv)
    reg = ILLUSIONS if a.illusions else REGISTRY
    out = Path(a.out)
    items, tasks = [], {}
    for name, (fn, version, langs) in reg.items():
        if a.only and name not in a.only:
            continue
        its = build_task_items(name, fn, version, langs, a.n, a.seed, out / "images")
        if a.illusions:  # illusion set is eval-only
            for it in its:
                it.split = "test"
        items += its
        tasks[name] = {"version": version, "langs": list(langs), "n_items": len(its)}
        print(f"{name}: {len(its)} items", flush=True)
    m = write_bank(items, out, a.seed, a.n, tasks)
    print(json.dumps({k: m[k] for k in ("n_items", "by_split", "by_primitive", "by_lang", "by_modality", "none_fits")}))
    if a.sample:
        rng = random.Random(a.seed)
        from .schema import write_jsonl

        write_jsonl(out / "sample.jsonl", (it.to_dict() for it in rng.sample(items, min(a.sample, len(items)))))


if __name__ == "__main__":
    main()
