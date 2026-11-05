"""Reduce every primitive to a sequence of categorical "answer steps".

choice: 1 step over option letters A.. (none-fits items target the uniform distribution)
noul:   1 step over ("Yes", "No")
score:  1 step over level digits "0".."n-1"
box:    4 steps (top, left, bottom, right), each over K coordinate bins, teacher-forced in order

Training reads the logits of the label tokens at each step position only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .schema import BOX_KEYS, Item

LETTERS = "ABCDEFGHIJ"
NOUL_LABELS = ("Yes", "No")
BOX_BINS = 32
# One single-character label per bin, so every backend can map a bin to one token.
BIN_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef")
BOX_STEP_PREFIXES = ("top=", " left=", " bottom=", " right=")


@dataclass
class Step:
    name: str
    labels: tuple[str, ...]
    target: int | None  # index of the correct label (None: no single correct label)
    soft: list[float] | None = None  # code-derived target distribution; trains instead of target when set


def bin_index(v: float, k: int = BOX_BINS) -> int:
    return max(0, min(k - 1, int(math.floor(v * k - 1e-9)) if v > 0 else 0))


def bin_center(i: int, k: int = BOX_BINS) -> float:
    return (i + 0.5) / k


def two_hot(v: float, k: int = BOX_BINS) -> list[float]:
    """Exact soft target: mass split between the two nearest bin centres so that the
    expected coordinate sum_i p_i * bin_center(i) equals v (inside the representable range)."""
    x = min(max(v * k - 0.5, 0.0), k - 1.0)
    i0 = int(math.floor(x))
    w = x - i0
    out = [0.0] * k
    out[i0] += 1 - w
    if w > 0:
        out[i0 + 1] += w
    return out


def choice_labels(n: int) -> tuple[str, ...]:
    return tuple(LETTERS[:n])


def item_steps(item: Item, k: int = BOX_BINS) -> list[Step]:
    p = item.primitive
    if p == "choice":
        keys = list(item.criteria)
        labels = choice_labels(len(keys))
        if item.none_fits:
            return [Step("choice", labels, None, [1.0 / len(keys)] * len(keys))]
        return [Step("choice", labels, keys.index(item.answer))]
    if p == "noul":
        return [Step("noul", NOUL_LABELS, 0 if item.answer else 1)]
    if p == "score":
        return [Step("score", tuple(str(i) for i in range(len(item.criteria))), int(item.answer))]
    if p == "box":
        if k != BOX_BINS:
            raise ValueError("BIN_LABELS are defined for 32 bins")
        steps = []
        for key in BOX_KEYS:
            soft = two_hot(item.answer[key], k)
            steps.append(Step(f"box_{key}", BIN_LABELS, max(range(k), key=lambda i: soft[i]), soft))
        return steps
    raise ValueError(p)


def decode_box(step_probs: Sequence[Sequence[float]], k: int = BOX_BINS, mode: str = "mean") -> dict:
    out = {}
    for key, probs in zip(BOX_KEYS, step_probs):
        if mode == "argmax":
            out[key] = bin_center(max(range(len(probs)), key=lambda i: probs[i]), k)
        else:
            out[key] = sum(p * bin_center(i, k) for i, p in enumerate(probs))
    if out["bottom"] < out["top"]:
        out["top"], out["bottom"] = out["bottom"], out["top"]
    if out["right"] < out["left"]:
        out["left"], out["right"] = out["right"], out["left"]
    return out


def box_iou(a: dict, b: dict) -> float:
    ih = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    iw = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    inter = ih * iw
    area = lambda x: max(0.0, x["bottom"] - x["top"]) * max(0.0, x["right"] - x["left"])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0
