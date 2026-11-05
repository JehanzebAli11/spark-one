"""Item schema shared by every generator, the teacher job, training and evaluation.

An Item is one question with an exact (code-derived) label and full provenance.

criteria by primitive:
  choice: dict option_key -> description (order is the presented order)
  score:  list of level descriptions (level number = index)
  noul:   optional {"true": ..., "false": ...}
  box:    None
answer by primitive:
  choice: option key, or None when no option fits (none_fits=True)
  score:  int level
  noul:   bool
  box:    {"top","left","bottom","right"} fractions of image height/width
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

PRIMITIVES = ("choice", "noul", "score", "box")
SPLITS = ("train", "val", "test")
BOX_KEYS = ("top", "left", "bottom", "right")


@dataclass
class Item:
    id: str
    task: str
    primitive: str
    instructions: str
    state: str
    criteria: Any
    answer: Any
    lang: str = "en"  # "en" or "ur-roman"
    image: str | None = None  # path relative to the bank root
    none_fits: bool = False
    split: str = "train"
    provenance: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def modality(self) -> str:
        return "image" if self.image else "text"

    def validate(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"{self.id}: bad primitive {self.primitive}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: bad split {self.split}")
        p, a, c = self.primitive, self.answer, self.criteria
        if p == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= 10:
                raise ValueError(f"{self.id}: choice needs 2..10 options")
            if self.none_fits:
                if a is not None:
                    raise ValueError(f"{self.id}: none_fits choice must have answer None")
            elif a not in c:
                raise ValueError(f"{self.id}: answer {a!r} not in options")
        elif p == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 10:
                raise ValueError(f"{self.id}: score needs 2..10 levels")
            if not isinstance(a, int) or not 0 <= a < len(c):
                raise ValueError(f"{self.id}: score answer out of range")
        elif p == "noul":
            if not isinstance(a, bool):
                raise ValueError(f"{self.id}: noul answer must be bool")
        elif p == "box":
            if self.image is None:
                raise ValueError(f"{self.id}: box needs an image")
            if not isinstance(a, dict) or set(a) != set(BOX_KEYS):
                raise ValueError(f"{self.id}: box answer needs {BOX_KEYS}")
            if not (0 <= a["top"] < a["bottom"] <= 1 and 0 <= a["left"] < a["right"] <= 1):
                raise ValueError(f"{self.id}: degenerate box {a}")
        for k in ("generator", "version", "seed", "label_source"):
            if k not in self.provenance:
                raise ValueError(f"{self.id}: provenance missing {k}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**d)


def stable_hash(*parts: Any) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]
