"""Build OpenAI-compatible chat requests for an Item and read option-token logprobs back.
Shared by the vLLM teacher job and the OpenRouter eval adapter."""

from __future__ import annotations

import base64
import math
from functools import lru_cache
from pathlib import Path

from ..prompt import SYSTEM, build_prompt
from ..schema import Item
from ..steps import Step, item_steps


@lru_cache(maxsize=4096)
def image_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def build_messages(item: Item, root: Path, steps: list[Step] | None = None, step_index: int = 0):
    """Messages for one answer step. Steps after the first are teacher-forced: the assistant
    turn is pre-filled with the earlier ground-truth labels (continue_final_message)."""
    p = build_prompt(item)
    steps = steps or item_steps(item)
    content = []
    if item.image:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(str(root / item.image))}})
    content.append({"type": "text", "text": p.text})
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    prefix = ""
    for i in range(step_index):
        prefix += p.step_prefixes[i] + steps[i].labels[steps[i].target]
    prefix += p.step_prefixes[step_index]
    prefix = prefix.lstrip()
    if prefix:
        msgs.append({"role": "assistant", "content": prefix})
    return msgs, bool(prefix)


def _norm(tok: str) -> str:
    return tok.replace("▁", " ").replace("Ġ", " ").strip().strip("*").strip()


def parse_top_logprobs(top: list[dict], labels: tuple[str, ...], case_insensitive: bool = False):
    """Map first-token top_logprobs onto the step labels.

    Variants of a label (" A", "A", "**A") are pooled. Labels missing from the top-k get
    min(smallest top-k prob, residual mass / n_missing). Returns (probs, coverage) where
    coverage is the raw probability mass the model put on the labels; low coverage means
    the model did not answer in the requested format."""
    key = (lambda s: s.lower()) if case_insensitive else (lambda s: s)
    want = {key(l): i for i, l in enumerate(labels)}
    mass = [0.0] * len(labels)
    seen = [False] * len(labels)
    min_p, total = 1.0, 0.0
    for t in top:
        p = math.exp(t["logprob"])
        total += p
        min_p = min(min_p, p)
        k = key(_norm(t["token"]))
        if k in want:
            mass[want[k]] += p
            seen[want[k]] = True
    coverage = sum(mass)
    missing = [i for i, s in enumerate(seen) if not s]
    if missing:
        fill = min(min_p, max(0.0, 1.0 - total) / len(missing)) if top else 1.0 / len(labels)
        for i in missing:
            mass[i] = max(fill, 1e-9)
    z = sum(mass)
    return [m / z for m in mass], coverage


def chat_body(model: str, messages, continue_final: bool, top_logprobs: int = 20) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": top_logprobs}
    if continue_final:  # vLLM extensions
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    return body


def first_token_top(resp: dict) -> list[dict]:
    lp = resp["choices"][0].get("logprobs") or {}
    content = lp.get("content") or []
    return content[0].get("top_logprobs", []) if content else []
