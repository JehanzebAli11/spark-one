"""Train the tiny System One model on CPU: KL to teacher + log loss on ground truth,
logits at answer tokens only, checkpoints every N steps, resume via LAB_RESUME_FROM.

python -m spark_one.tiny.train --config configs/tiny_student.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..schema import load_items, read_jsonl
from ..train.losses import answer_loss, step_logits
from .data import collate
from .model import TinyConfig, TinySystemOne


def load_teacher(path: str | None) -> dict:
    if not path:
        return {}
    out = {}
    for r in read_jsonl(path):
        if r.get("ok", True):
            out[r["id"]] = [s["probs"] for s in r["steps"]]
    return out


def save_checkpoint(run_dir: Path, step: int, model, opt, rng_state, cfg: dict, keep: int = 3):
    ck = run_dir / f"checkpoint-{step}"
    tmp = run_dir / f".tmp-checkpoint-{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save(opt.state_dict(), tmp / "optim.pt")
    torch.save(rng_state, tmp / "rng.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2))
    if ck.exists():
        shutil.rmtree(ck)
    os.replace(tmp, ck)  # atomic: a half-written checkpoint is never picked for resume
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in cks[:-keep]:
        shutil.rmtree(old)
    return ck


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from", default=os.environ.get("LAB_RESUME_FROM"))
    ap.add_argument("--max_steps", type=int, help="override (e.g. to simulate a crash)")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    torch.set_num_threads(cfg.get("threads", os.cpu_count() or 4))
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    bank = Path(cfg["bank"])
    items = [it for it in load_items(bank / "items.jsonl") if it.split == "train"]
    if cfg.get("tasks"):
        items = [it for it in items if it.provenance["generator"] in cfg["tasks"]]
    if cfg.get("max_train_items"):
        items = random.Random(0).sample(items, min(cfg["max_train_items"], len(items)))
    teacher = load_teacher(cfg.get("teacher"))
    mc = TinyConfig(**cfg["model"])
    torch.manual_seed(cfg.get("seed", 0))
    model = TinySystemOne(mc)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    rng = random.Random(cfg.get("seed", 0))
    step = 0
    if a.resume_from:
        ck = Path(a.resume_from)
        model.load_state_dict(torch.load(ck / "model.pt"))
        opt.load_state_dict(torch.load(ck / "optim.pt"))
        st = torch.load(ck / "rng.pt", weights_only=False)
        rng.setstate(st["python"])
        torch.set_rng_state(st["torch"])
        step = json.loads((ck / "state.json").read_text())["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    total = cfg["steps"]
    stop_at = min(total, a.max_steps) if a.max_steps else total
    print(f"train items {len(items)}, teacher labels {len(teacher)}, params {model.n_params():,}", flush=True)
    log = open(run_dir / "train_log.jsonl", "a")
    model.train()
    t0 = time.time()
    while step < stop_at:
        batch = [items[rng.randrange(len(items))] for _ in range(cfg["batch_size"])]
        ids, images, sb = collate(batch, bank, mc, teacher)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, cfg["lr"], cfg.get("warmup", 100))
        logits, _ = model(ids, images)
        loss, parts = answer_loss(step_logits(logits, sb), sb, cfg.get("alpha", 0.0), cfg.get("kd_temp", 1.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % cfg.get("log_every", 50) == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), **{k: round(v, 4) for k, v in parts.items()},
                   "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % cfg.get("ckpt_every", 500) == 0 or step == total:
            save_checkpoint(run_dir, step, model, opt,
                            {"python": rng.getstate(), "torch": torch.get_rng_state()}, cfg)
    if step >= total:
        torch.save({"config": mc.to_dict(), "state_dict": model.state_dict()}, run_dir / "final.pt")
        print(f"done: {run_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
