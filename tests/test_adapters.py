import json
import math

import httpx
import pytest

from spark_one.adapters.cost import BudgetExceeded, CostTracker
from spark_one.adapters.jev import JevAdapter
from spark_one.adapters.openrouter import OpenRouterAdapter, text_to_probs
from spark_one.schema import load_items


def _items(bank):
    its = load_items(bank / "items.jsonl")
    pick = {}
    for it in its:
        if not it.none_fits:
            pick.setdefault(it.primitive, it)
    return pick


def test_jev_adapter_maps_answers_and_caps_cost(small_bank):
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append((body, dict(req.headers)))
        q = body["questions"]["q"]
        if q["type"] == "choice":
            keys = list(q["criteria"])
            ans = {"type": "choice", "choice": keys[0], "probabilities": {k: 1 / len(keys) for k in keys}, "confidence": 0}
        elif q["type"] == "noul":
            ans = {"type": "noul", "noul": 0.8}
        elif q["type"] == "score":
            ans = {"type": "score", "score": 1.0, "probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}}
        else:
            ans = {"type": "box", "box": {"top": .1, "bottom": .5, "left": .2, "right": .6}}
        return httpx.Response(200, json={"model": "jev-x", "answers": {"q": ans}, "usage": {"input_tokens": 100000, "output_tokens": 1}})

    budget = CostTracker(0.25)
    ad = JevAdapter(budget, client=httpx.Client(transport=httpx.MockTransport(handler)), usd_per_mtok_in=1.0)
    its = _items(small_bank)
    assert ad.predict(its["noul"], small_bank)["probs"] == [[0.8, pytest.approx(0.2)]]
    assert "box" in ad.predict(its["box"], small_bank)
    assert seen[-1][1].get("typesafe-danger-zone-images") == "true"
    with pytest.raises(BudgetExceeded):
        for _ in range(5):
            ad.predict(its["choice"], small_bank)
    assert budget.spent <= 0.25 + 1e-9


def test_openrouter_logprobs_and_text_fallback(small_bank):
    def handler(req):
        body = json.loads(req.content)
        if body["model"] == "with-logprobs":
            top = [{"token": "B", "logprob": math.log(0.7)}, {"token": "A", "logprob": math.log(0.2)}]
            return httpx.Response(200, json={"choices": [{"message": {"content": "B"}, "logprobs": {"content": [{"token": "B", "logprob": 0, "top_logprobs": top}]}}],
                                             "usage": {"cost": 0.001}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yes"}}], "usage": {"cost": 0.002}})

    its = _items(small_bank)
    b = CostTracker(1.0)
    ad = OpenRouterAdapter("with-logprobs", b, client=httpx.Client(transport=httpx.MockTransport(handler)))
    row = ad.predict(its["choice"], small_bank)
    assert row["source"] == "logprobs" and max(range(len(row["logp"][0])), key=lambda i: row["logp"][0][i]) == 1
    ad2 = OpenRouterAdapter("text-only", b, client=httpx.Client(transport=httpx.MockTransport(handler)))
    row = ad2.predict(its["noul"], small_bank)
    assert row["source"] == "text" and math.exp(row["logp"][0][0]) > 0.9
    assert b.spent == pytest.approx(0.003)
    assert text_to_probs("**C**", ("A", "B", "C"))[2] > 0.9
    assert text_to_probs("nonsense", ("A", "B")) is None
