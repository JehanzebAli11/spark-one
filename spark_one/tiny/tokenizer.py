"""Byte-level tokenizer with one token per answer label and image placeholders."""

from __future__ import annotations

from ..steps import BIN_LABELS, LETTERS, NOUL_LABELS

PAD, BOS, ANS, IMG = 256, 257, 258, 259
N_SPECIAL = 4
LABELS = tuple(sorted(set(LETTERS) | set(NOUL_LABELS) | {str(i) for i in range(10)} | set(BIN_LABELS)))
LABEL_ID = {lab: 256 + N_SPECIAL + i for i, lab in enumerate(LABELS)}
ID_LABEL = {v: k for k, v in LABEL_ID.items()}
VOCAB = 256 + N_SPECIAL + len(LABELS)
ANSWER_IDS = tuple(LABEL_ID[l] for l in LABELS)  # the answer vocabulary


def encode_text(s: str) -> list[int]:
    return list(s.encode("utf-8"))


def label_id(label: str) -> int:
    return LABEL_ID[label]
