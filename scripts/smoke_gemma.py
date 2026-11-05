"""Smoke-test the Gemma 4 path on real weights (CPU or GPU): first-token behaviour at the
answer position, label coverage, prefix-cache equivalence, and Box steps.

python scripts/smoke_gemma.py --model google/gemma-4-E2B-it --bank data/generated/bank --n 6
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spark_one.hf import HFBackend, encode_example, forward_example, load
from spark_one.prompt import question_text
from spark_one.schema import load_items
from spark_one.steps import item_steps

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-4-E2B-it")
ap.add_argument("--bank", default="data/generated/bank")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(a.threads)
t = time.time()
dtype = torch.bfloat16
model, proc, tok = load(a.model, dtype=dtype)
print(f"loaded in {time.time() - t:.0f}s on {next(model.parameters()).device}", flush=True)
root = Path(a.bank)
items = [it for it in load_items(root / "items.jsonl") if it.split == "test"]
rng = random.Random(0)
pick = []
for prim, mod in (("choice", "text"), ("noul", "image"), ("choice", "image"), ("score", "text"), ("box", "image"), ("noul", "text")):
    c = [it for it in items if it.primitive == prim and it.modality == mod and not it.none_fits]
    pick.append(rng.choice(c))
pick = pick[: a.n]
be = HFBackend(a.model, model=model, processor=proc, tok=tok)
results = []
for it in pick:
    t = time.time()
    ex = encode_example(it, root, proc, tok, be.device)
    with torch.no_grad():
        logits = forward_example(model, ex, be.device).float()
    top = torch.topk(torch.softmax(logits[0], -1), 8)
    top_tokens = [(tok.decode([i]), round(p, 4)) for p, i in zip(top.values.tolist(), top.indices.tolist())]
    st = ex.steps[0]
    label_mass = float(torch.softmax(logits[0], -1)[ex.step_label_ids[0]].sum())
    full_s = time.time() - t
    img = (root / it.image).read_bytes() if it.image else None
    t = time.time()
    pre, hit = be.prefix(it.state, img)
    q, sp = question_text(it.primitive, it.instructions, it.criteria)
    lps = be.answer_steps(pre, q, sp, [s.labels for s in item_steps(it)])
    cached_s = time.time() - t
    t = time.time()
    lps2 = be.answer_steps(pre, q, sp, [s.labels for s in item_steps(it)])
    reuse_s = time.time() - t
    lp_full = torch.log_softmax(logits[0, ex.step_label_ids[0]], -1).tolist()
    r = {"id": it.id, "primitive": it.primitive, "modality": it.modality, "answer": it.answer,
         "target_label": st.labels[st.target] if st.target is not None else None,
         "top_tokens_at_answer": top_tokens, "label_mass_first_step": round(label_mass, 4),
         "pred_label": st.labels[int(torch.tensor(lps[0]).argmax())],
         "max_abs_diff_cached_vs_full_step0": round(max(abs(x - y) for x, y in zip(lps[0], lp_full)), 4),
         "seconds_full_forward": round(full_s, 1), "seconds_prefix_plus_answer": round(cached_s, 1),
         "seconds_answer_with_cached_prefix": round(reuse_s, 1), "n_prefix_tokens": pre[2]}
    if it.primitive == "box":
        import numpy as np

        from spark_one.steps import box_iou, decode_box
        ps = [np.exp(lp) / np.exp(lp).sum() for lp in lps]
        r["pred_box"] = decode_box(ps)
        r["iou"] = round(box_iou(r["pred_box"], it.answer), 3)
    print(json.dumps(r), flush=True)
    results.append(r)
if a.out:
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n")
