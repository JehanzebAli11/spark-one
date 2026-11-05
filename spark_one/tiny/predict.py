"""Predict with a tiny checkpoint over eval variants; writes one JSON row per variant with
per-step log-probabilities over the step labels. Box steps decode greedily (no teacher forcing).

python -m spark_one.tiny.predict --ckpt runs/tiny/student_kd/final.pt --bank data/generated/bank \
    --split test --out runs/tiny/student_kd/test_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..eval.variants import expand
from ..schema import load_items
from ..steps import Step, item_steps
from .encode import encode_item, image_array
from .model import TinyConfig, TinySystemOne
from .tokenizer import PAD


def load_model(path: str) -> TinySystemOne:
    ck = torch.load(path, weights_only=False)
    m = TinySystemOne(TinyConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def _forward(model, items, steps_list, k, root):
    c = model.c
    encs = [encode_item(it, c.n_patches, st) for it, st in zip(items, steps_list)]
    T = max(len(e.ids) for e in encs)
    ids = torch.full((len(items), T), PAD, dtype=torch.long)
    images = torch.zeros(len(items), 3, c.img_size, c.img_size)
    for b, (it, e) in enumerate(zip(items, encs)):
        ids[b, : len(e.ids)] = torch.tensor(e.ids)
        arr = image_array(it, root, c.img_size)
        if arr is not None:
            images[b] = torch.from_numpy(np.array(arr)).permute(2, 0, 1).float() / 255.0
    logits, _ = model(ids, images)
    out = []
    for b, e in enumerate(encs):
        sub = logits[b, e.step_pos[k], e.step_labels[k]]
        out.append(torch.log_softmax(sub, -1).tolist())
    return out


def predict(model, items, root: Path, batch_size: int = 64):
    """Returns list of (logp_per_step, seconds_per_item)."""
    results = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: (items[i].primitive == "box", len(items[i].state)))
    for s in range(0, len(order), batch_size):
        idx = order[s: s + batch_size]
        its = [items[i] for i in idx]
        t0 = time.time()
        steps = [item_steps(it) for it in its]
        n_steps = max(len(st) for st in steps)
        logps = [[] for _ in its]
        for k in range(n_steps):
            live = [j for j, st in enumerate(steps) if k < len(st)]
            res = _forward(model, [its[j] for j in live], [steps[j] for j in live], k, root)
            for j, lp in zip(live, res):
                logps[j].append(lp)
                if k + 1 < len(steps[j]):  # greedy: feed the predicted bin to the next step
                    st = steps[j][k]
                    steps[j][k] = Step(st.name, st.labels, int(np.argmax(lp)))
        dt = (time.time() - t0) / len(its)
        for j, i in enumerate(idx):
            results[i] = (logps[j], dt)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    root = Path(a.bank)
    items = [it for it in load_items(root / "items.jsonl") if it.split == a.split]
    if a.limit:
        import random

        items = random.Random(0).sample(items, min(a.limit, len(items)))
    var = expand(items, perms=not a.no_variants, negations=not a.no_variants)
    model = load_model(a.ckpt)
    res = predict(model, [v[3] for v in var], root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for (vid, variant, order, it), (logp, dt) in zip(var, res):
            base_id = vid.split("~")[0]
            f.write(json.dumps({"id": vid, "base_id": base_id, "variant": variant, "order": order,
                                "logp": [[round(x, 5) for x in lp] for lp in logp],
                                "latency_ms": round(dt * 1000, 3), "source": "logprobs"}) + "\n")
    print(f"wrote {len(var)} predictions to {a.out}")


if __name__ == "__main__":
    main()
