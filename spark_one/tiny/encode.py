"""Turn Items into token sequences and answer-step positions for the tiny model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from ..prompt import question_text, state_prefix
from ..schema import Item
from ..steps import item_steps
from .tokenizer import ANS, BOS, IMG, encode_text, label_id

MAX_TEXT = 640


@dataclass
class Encoded:
    ids: list[int]
    step_pos: list[int]  # position whose output predicts each step label
    step_labels: list[list[int]]  # candidate label token ids per step
    step_target: list[int | None]
    step_soft: list[list[float] | None]
    prefix_len: int  # tokens in the cacheable prefix (BOS + image + state)


def encode_parts(state: str, primitive: str, instructions: str, criteria, n_patches: int,
                 has_image: bool):
    st = encode_text(state_prefix(state))
    q_text, step_prefixes = question_text(primitive, instructions, criteria)
    q = encode_text(q_text)
    if len(st) + len(q) > MAX_TEXT:  # keep the question whole, cut the state
        st = st[: max(16, MAX_TEXT - len(q) - 4)] + encode_text("...\n")
    pre = [BOS] + ([IMG] * n_patches if has_image else []) + st
    return pre, q + [ANS], step_prefixes


def encode_item(item: Item, n_patches: int, label_steps=None) -> Encoded:
    steps = label_steps or item_steps(item)
    pre, q, sp = encode_parts(item.state, item.primitive, item.instructions, item.criteria,
                              n_patches, item.image is not None)
    ids = pre + q
    pos, labs, tg, soft = [], [], [], []
    for i, st in enumerate(steps):
        ids += encode_text(sp[i]) if sp[i] else []
        pos.append(len(ids) - 1)
        labs.append([label_id(l) for l in st.labels])
        tg.append(st.target)
        soft.append(st.soft)
        if i < len(steps) - 1:  # teacher forcing: feed the true label for the next step
            ids.append(label_id(st.labels[st.target]))
    return Encoded(ids, pos, labs, tg, soft, len(pre))


@lru_cache(maxsize=200_000)
def load_image(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def image_array(item: Item, root: Path, size: int) -> np.ndarray | None:
    if not item.image:
        return None
    return load_image(str(root / item.image), size)
