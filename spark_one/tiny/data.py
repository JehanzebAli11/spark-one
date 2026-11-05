"""Batching for the tiny model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..schema import Item
from ..steps import item_steps
from ..train.losses import StepBatch
from .encode import encode_item, image_array
from .tokenizer import PAD


def collate(items: list[Item], root: Path, cfg, teacher: dict | None = None):
    encs = [encode_item(it, cfg.n_patches) for it in items]
    B, T = len(items), max(len(e.ids) for e in encs)
    ids = torch.full((B, T), PAD, dtype=torch.long)
    images = torch.zeros(B, 3, cfg.img_size, cfg.img_size)
    rows, pos, labs, tgt, soft, teach, has_t, prim = [], [], [], [], [], [], [], []
    for b, (it, e) in enumerate(zip(items, encs)):
        ids[b, : len(e.ids)] = torch.tensor(e.ids)
        arr = image_array(it, root, cfg.img_size)
        if arr is not None:
            images[b] = torch.from_numpy(np.array(arr)).permute(2, 0, 1).float() / 255.0
        tsteps = (teacher or {}).get(it.id)
        for s in range(len(e.step_pos)):
            rows.append(b)
            pos.append(e.step_pos[s])
            labs.append(e.step_labels[s])
            tgt.append(-1 if e.step_target[s] is None else e.step_target[s])
            soft.append(e.step_soft[s] or [0.0] * len(e.step_labels[s]))
            tp = tsteps[s] if tsteps and s < len(tsteps) else None
            ok = tp is not None and len(tp) == len(e.step_labels[s])
            teach.append(tp if ok else [0.0] * len(e.step_labels[s]))
            has_t.append(ok)
            prim.append(it.primitive)
    L = max(len(l) for l in labs)
    pad = lambda rows_, v: [r + [v] * (L - len(r)) for r in rows_]
    sb = StepBatch(
        row=torch.tensor(rows), pos=torch.tensor(pos),
        label_ids=torch.tensor(pad(labs, 0)),
        label_mask=torch.tensor([[True] * len(l) + [False] * (L - len(l)) for l in labs]),
        target=torch.tensor(tgt), soft=torch.tensor(pad(soft, 0.0)),
        teacher=torch.tensor(pad(teach, 0.0)), has_teacher=torch.tensor(has_t), primitive=prim)
    return ids, images, sb
