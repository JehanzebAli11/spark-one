"""LoRA training of the Gemma 4 E2B student on the Spark.

loss = (1 - alpha) * log loss on ground truth + alpha * KL(teacher || student), on the logits
of the label tokens at the answer positions only; Box = 4 binned coordinate steps.
Checkpoints go to run_dir/checkpoint-N/ (adapter + optimizer + RNG); the spark-lab worker
passes the newest one as LAB_RESUME_FROM and training continues from that step.

python -m spark_one.train.lora --config configs/gemma_e2b_lora.json
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

import torch

from ..schema import load_items
from ..tiny.train import load_teacher
from .losses import StepBatch, answer_loss


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def example_batch(ex, item, teacher, device) -> StepBatch:
    S = len(ex.step_pos)
    L = max(len(l) for l in ex.step_label_ids)
    lab = torch.zeros(S, L, dtype=torch.long)
    mask = torch.zeros(S, L, dtype=torch.bool)
    soft = torch.zeros(S, L)
    tea = torch.zeros(S, L)
    has = torch.zeros(S, dtype=torch.bool)
    tgt = torch.full((S,), -1, dtype=torch.long)
    tp = teacher.get(item.id)
    for s, st in enumerate(ex.steps):
        n = len(st.labels)
        lab[s, :n] = torch.tensor(ex.step_label_ids[s])
        mask[s, :n] = True
        if st.soft is not None:
            soft[s, :n] = torch.tensor(st.soft)
        if st.target is not None:
            tgt[s] = st.target
        if tp and s < len(tp) and len(tp[s]) == n:
            tea[s, :n] = torch.tensor(tp[s])
            has[s] = True
    return StepBatch(row=torch.zeros(S, dtype=torch.long), pos=torch.arange(S), label_ids=lab.to(device),
                     label_mask=mask.to(device), target=tgt.to(device), soft=soft.to(device),
                     teacher=tea.to(device), has_teacher=has.to(device), primitive=[item.primitive] * S)


def attach_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model

    names = [n for n, _ in model.named_modules()]
    target = cfg.get("target_regex", r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    import re

    if not any(re.match(target, n) for n in names):  # text-only models have no language_model prefix
        target = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    lc = LoraConfig(r=cfg.get("r", 16), lora_alpha=cfg.get("alpha", 32), lora_dropout=cfg.get("dropout", 0.05),
                    target_modules=target, bias="none")
    return get_peft_model(model, lc)


def save_ckpt(run_dir: Path, step: int, model, opt, rng, cfg, keep: int = 3):
    tmp, ck = run_dir / f".tmp-checkpoint-{step}", run_dir / f"checkpoint-{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save(opt.state_dict(), tmp / "optim.pt")
    torch.save({"python": rng.getstate(), "torch": torch.get_rng_state()}, tmp / "rng.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2))
    if ck.exists():
        shutil.rmtree(ck)
    os.replace(tmp, ck)
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in cks[:-keep]:
        shutil.rmtree(old)


def main(argv=None, model_loader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from", default=os.environ.get("LAB_RESUME_FROM"))
    ap.add_argument("--max_steps", type=int)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    from ..hf import encode_example, forward_example, load

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model, processor, tok = (model_loader or load)(cfg["base_model"])
    device = next(model.parameters()).device
    model = attach_lora(model, cfg.get("lora", {}))
    if cfg.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    bank = Path(cfg["bank"])
    items = [it for it in load_items(bank / "items.jsonl") if it.split == "train"]
    if cfg.get("modalities"):
        items = [it for it in items if it.modality in cfg["modalities"]]
    if cfg.get("max_train_items"):
        items = random.Random(0).sample(items, min(cfg["max_train_items"], len(items)))
    teacher = load_teacher(cfg.get("teacher"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    rng = random.Random(cfg.get("seed", 0))
    step = 0
    if a.resume_from:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ck = Path(a.resume_from)
        set_peft_model_state_dict(model, load_file(str(ck / "adapter" / "adapter_model.safetensors")))
        opt.load_state_dict(torch.load(ck / "optim.pt"))
        st = torch.load(ck / "rng.pt", weights_only=False)
        rng.setstate(st["python"])
        torch.set_rng_state(st["torch"])
        step = json.loads((ck / "state.json").read_text())["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    total, accum = cfg["steps"], cfg.get("grad_accum", 16)
    stop_at = min(total, a.max_steps) if a.max_steps else total
    print(f"train items {len(items)}, teacher labels {len(teacher)}, trainable params "
          f"{sum(p.numel() for p in params):,}", flush=True)
    log = open(run_dir / "train_log.jsonl", "a")
    model.train()
    t0 = time.time()
    while step < stop_at:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, cfg["lr"], cfg.get("warmup", 50))
        tot, parts_acc = 0.0, {"ce": 0.0, "kl": 0.0}
        for _ in range(accum):
            it = items[rng.randrange(len(items))]
            ex = encode_example(it, bank, processor, tok, device)
            logits = forward_example(model, ex, device).float()
            sb = example_batch(ex, it, teacher, device)
            sub = logits.gather(1, sb.label_ids).masked_fill(~sb.label_mask, float("-inf"))
            loss, parts = answer_loss(sub, sb, cfg.get("alpha", 0.5))
            (loss / accum).backward()
            tot += loss.item() / accum
            for k in parts_acc:
                parts_acc[k] += parts[k] / accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.get("log_every", 10) == 0 or step == 1:
            rec = {"step": step, "loss": round(tot, 4), **{k: round(v, 4) for k, v in parts_acc.items()},
                   "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % cfg.get("ckpt_every", 200) == 0 or step == total:
            save_ckpt(run_dir, step, model, opt, rng, cfg)
    if step >= total:
        model.save_pretrained(run_dir / "final_adapter")
        print(f"done: {run_dir / 'final_adapter'}", flush=True)


if __name__ == "__main__":
    main()
