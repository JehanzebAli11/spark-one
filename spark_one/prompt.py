"""One prompt format for the teacher, the student, the server and the eval adapters.

A prompt is split into a cacheable prefix (image + state) and a per-question suffix, so
several questions about one state share the prefix (prefix caching).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Item
from .steps import BIN_LABELS, BOX_BINS, BOX_STEP_PREFIXES, LETTERS

SYSTEM = (
    "You answer one question about the given state. Reply with only the requested label."
)


@dataclass
class Prompt:
    prefix: str  # image (if any) is placed before this text
    question: str
    step_prefixes: tuple[str, ...]  # assistant text emitted before each answer label

    @property
    def text(self) -> str:
        return self.prefix + self.question


def state_prefix(state: str) -> str:
    return f"State:\n{state}\n\n"


def question_text(primitive: str, instructions: str, criteria) -> tuple[str, tuple[str, ...]]:
    if primitive == "choice":
        lines = [f"Question: {instructions}", "Options:"]
        for letter, (key, desc) in zip(LETTERS, criteria.items()):
            lines.append(f"{letter}. {key}" + (f": {desc}" if desc else ""))
        lines.append("Reply with only the letter of the best option.")
        return "\n".join(lines) + "\nAnswer:", ("",)
    if primitive == "noul":
        lines = [f"Question: {instructions}"]
        if criteria:
            lines.append(f"Yes means: {criteria.get('true')}")
            lines.append(f"No means: {criteria.get('false')}")
        lines.append("Reply with only Yes or No.")
        return "\n".join(lines) + "\nAnswer:", ("",)
    if primitive == "score":
        lines = [f"Question: {instructions}", "Levels:"]
        for i, desc in enumerate(criteria):
            lines.append(f"{i}. {desc}")
        lines.append("Reply with only the level number.")
        return "\n".join(lines) + "\nAnswer:", ("",)
    if primitive == "box":
        lines = [
            f"Question: Locate {instructions}",
            f"Each image axis is cut into {BOX_BINS} equal bins labelled "
            f"{BIN_LABELS[0]}..{BIN_LABELS[-1]} from the top (rows) and from the left (columns).",
            "Reply as: top=<bin> left=<bin> bottom=<bin> right=<bin>",
        ]
        return "\n".join(lines) + "\nAnswer:", BOX_STEP_PREFIXES
    raise ValueError(primitive)


def build_prompt(item: Item) -> Prompt:
    q, sp = question_text(item.primitive, item.instructions, item.criteria)
    return Prompt(state_prefix(item.state), q, sp)
