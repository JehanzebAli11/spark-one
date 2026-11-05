"""Eval-only adapter for TypeSafe's Jev via POST https://api.typesafe.ai/v1/systemone.

Maps an Item to a System One question, sends the image as an asset, and maps the answer
back to a prediction row (probabilities in the item's option order; Box as coordinates).
Cost = input tokens x price (set JEV_USD_PER_MTOK_IN; output tokens are free per the docs).
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx

from ..schema import Item
from .cost import CostTracker

URL = "https://api.typesafe.ai/v1/systemone"


def to_request(item: Item, root: Path, model: str = "jev-latest") -> tuple[dict, dict]:
    q: dict = {"type": item.primitive, "instructions": item.instructions}
    if item.primitive == "choice":
        q["criteria"] = {k: v for k, v in item.criteria.items()}
    elif item.primitive == "score":
        q["criteria"] = list(item.criteria)
    elif item.primitive == "noul" and item.criteria:
        q["criteria"] = item.criteria
    body = {"model": model, "state": item.state, "questions": {"q": q}}
    headers = {}
    if item.image:
        data = base64.b64encode((root / item.image).read_bytes()).decode()
        body["assets"] = {"image": {"type": "image", "data_base64": data, "format": "png"}}
        headers["TypeSafe-Danger-Zone-Images"] = "true"
    return body, headers


def from_answer(item: Item, ans: dict) -> dict:
    if item.primitive == "choice":
        return {"probs": [[float(ans["probabilities"].get(k, 0.0)) for k in item.criteria]]}
    if item.primitive == "score":
        return {"probs": [[float(ans["probabilities"].get(str(i), 0.0)) for i in range(len(item.criteria))]]}
    if item.primitive == "noul":
        p = float(ans["noul"])
        return {"probs": [[p, 1 - p]]}
    b = ans["box"]
    return {"box": {k: float(b[k]) for k in ("top", "left", "bottom", "right")}}


class JevAdapter:
    def __init__(self, budget: CostTracker, model: str = "jev-latest", api_key: str | None = None,
                 usd_per_mtok_in: float | None = None, client: httpx.Client | None = None, url: str = URL):
        self.model, self.budget, self.url = model, budget, url
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.price = usd_per_mtok_in if usd_per_mtok_in is not None else float(os.environ.get("JEV_USD_PER_MTOK_IN", "1.0"))
        self.client = client or httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    def predict(self, item: Item, root: Path) -> dict:
        self.budget.check(upcoming_estimate=self.price * 2000 / 1e6)
        body, headers = to_request(item, root, self.model)
        t0 = time.time()
        r = self.client.post(self.url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        self.budget.add(usage.get("input_tokens", 0) * self.price / 1e6)
        row = from_answer(item, data["answers"]["q"])
        row.update({"latency_ms": round((time.time() - t0) * 1000, 1), "source": f"jev:{data.get('model')}"})
        return row
