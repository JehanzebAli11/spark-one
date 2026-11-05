"""Eval-only adapter for OpenRouter chat models. Reads first-token logprobs when the model
provides them (source=logprobs); otherwise parses the text answer into a near one-hot
distribution (source=text, flagged in reports). Box: asks for the 4 bins, step by step, as text.
Costs come from OpenRouter's usage accounting and are capped by CostTracker.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx

from ..schema import Item
from ..steps import item_steps
from ..teacher.openai_chat import build_messages, first_token_top, parse_top_logprobs
from .cost import CostTracker

URL = "https://openrouter.ai/api/v1/chat/completions"
TEXT_EPS = 0.02  # probability mass spread over the other labels for text-only answers


def text_to_probs(text: str, labels) -> list[float] | None:
    m = re.search(r"[A-Za-z0-9]+", text.strip().strip("*"))
    if not m:
        return None
    tok = m.group(0)
    idx = next((i for i, l in enumerate(labels) if l == tok or (len(l) > 1 and l.lower() == tok.lower())), None)
    if idx is None:
        return None
    n = len(labels)
    return [1 - TEXT_EPS if i == idx else TEXT_EPS / (n - 1) for i in range(n)]


class OpenRouterAdapter:
    def __init__(self, model: str, budget: CostTracker, client: httpx.Client | None = None,
                 use_logprobs: bool = True, url: str = URL):
        self.model, self.budget, self.url, self.use_logprobs = model, budget, url, use_logprobs
        key = os.environ.get("OPENROUTER_API_KEY", "proxy-injected")
        self.client = client or httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"})

    def _call(self, messages, continue_final: bool) -> dict:
        self.budget.check()
        if continue_final:  # prefill is passed as a trailing assistant message
            pass
        body = {"model": self.model, "messages": messages, "max_tokens": 4, "temperature": 0,
                "usage": {"include": True}}
        if self.use_logprobs:
            body.update({"logprobs": True, "top_logprobs": 20})
        r = self.client.post(self.url, json=body)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(str(data["error"])[:300])
        cost = (data.get("usage") or {}).get("cost")
        self.budget.add(float(cost) if cost is not None else 0.0)
        return data

    def predict(self, item: Item, root: Path) -> dict:
        steps = item_steps(item)
        logp_rows, sources = [], set()
        t0 = time.time()
        import math

        for k, st in enumerate(steps):
            msgs, cont = build_messages(item, root, steps, k)
            if k > 0:  # feed our own greedy previous bins, not the ground truth
                prev = ""
                from ..prompt import build_prompt

                sp = build_prompt(item).step_prefixes
                for j in range(k):
                    prev += sp[j] + st_labels_argmax(steps[j].labels, logp_rows[j])
                prev += sp[k]
                msgs[-1] = {"role": "assistant", "content": prev.lstrip()}
            data = self._call(msgs, cont)
            top = first_token_top(data)
            probs = None
            if top:
                probs, cov = parse_top_logprobs(top, st.labels, case_insensitive=item.primitive == "noul")
                if cov < 0.5:
                    probs = None
                else:
                    sources.add("logprobs")
            if probs is None:
                text = data["choices"][0]["message"].get("content") or ""
                probs = text_to_probs(text, st.labels)
                sources.add("text")
                if probs is None:
                    probs = [1 / len(st.labels)] * len(st.labels)
                    sources.add("unparsed")
            logp_rows.append([math.log(max(p, 1e-12)) for p in probs])
        return {"logp": logp_rows, "latency_ms": round((time.time() - t0) * 1000, 1),
                "source": "+".join(sorted(sources))}


def st_labels_argmax(labels, logp):
    return labels[max(range(len(labels)), key=lambda i: logp[i])]
