"""Answer-step loss shared by the tiny model and the Gemma LoRA trainer.

loss = (1 - alpha) * CE(ground truth) + alpha * KL(teacher || student)
computed only over the logits of the step's label tokens at the answer positions.
Steps without teacher probabilities use the ground-truth term alone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StepBatch:
    """S answer steps flattened over a batch. L = max labels in any step."""
    row: torch.Tensor  # [S] batch row
    pos: torch.Tensor  # [S] position whose logits predict the label
    label_ids: torch.Tensor  # [S, L] token ids (pad 0)
    label_mask: torch.Tensor  # [S, L] bool
    target: torch.Tensor  # [S] index of correct label, -1 when using soft target
    soft: torch.Tensor  # [S, L] code-derived soft target (used when target == -1)
    teacher: torch.Tensor  # [S, L] teacher probs (zeros when absent)
    has_teacher: torch.Tensor  # [S] bool
    primitive: list[str]


def step_logits(logits: torch.Tensor, sb: StepBatch) -> torch.Tensor:
    """logits: [B, T, V] -> label-subset logits [S, L] with padding at -inf."""
    at = logits[sb.row, sb.pos]  # [S, V]
    sub = at.gather(1, sb.label_ids)
    return sub.masked_fill(~sb.label_mask, float("-inf"))


def answer_loss(sub: torch.Tensor, sb: StepBatch, alpha: float, kd_temp: float = 1.0):
    logp = F.log_softmax(sub, dim=-1)
    safe_logp = logp.masked_fill(~sb.label_mask, 0.0)
    hard = sb.target >= 0
    tgt = torch.where(hard, sb.target, torch.zeros_like(sb.target))
    ce_hard = -safe_logp.gather(1, tgt[:, None]).squeeze(1)
    ce_soft = -(sb.soft * safe_logp).sum(1)
    ce = torch.where(hard, ce_hard, ce_soft)
    if kd_temp != 1.0:
        logp_t = F.log_softmax(sub / kd_temp, dim=-1).masked_fill(~sb.label_mask, 0.0)
    else:
        logp_t = safe_logp
    t = sb.teacher.clamp_min(1e-12)
    kl = (sb.teacher * (t.log() - logp_t)).masked_fill(~sb.label_mask, 0.0).sum(1) * kd_temp ** 2
    w = torch.where(sb.has_teacher, torch.full_like(ce, alpha), torch.zeros_like(ce))
    per = (1 - w) * ce + w * kl
    return per.mean(), {"ce": ce.mean().item(),
                        "kl": (kl[sb.has_teacher].mean().item() if sb.has_teacher.any() else 0.0)}
