import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from spark_one.confidence import choice_confidence
from spark_one.serve.app import create_app
from spark_one.serve.backends import TinyBackend


@pytest.fixture()
def client(tiny_ckpt):
    be = TinyBackend(tiny_ckpt, threads=1)
    return TestClient(create_app(be, {"choice": {"temperature": 1.5}})), be


def _png():
    b = io.BytesIO()
    Image.new("RGB", (40, 30), "red").save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def test_systemone_all_primitives(client):
    c, be = client
    req = {"model": "spark-1", "state": {"chat": ["hi", "hello"]},
           "assets": {"img": {"type": "image", "data_base64": _png(), "format": "png"}},
           "questions": {
               "tone": {"type": "choice", "instructions": "Tone?", "criteria": {"calm": None, "angry": "upset", "excited": None}},
               "q": {"type": "noul", "instructions": "Is it friendly?"},
               "s": {"type": "score", "instructions": "How friendly?", "criteria": ["not", "a bit", "very"]},
               "b": {"type": "box", "instructions": "the red area"}}}
    r = c.post("/v1/systemone", json=req)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    p = a["tone"]["probabilities"]
    assert set(p) == {"calm", "angry", "excited"} and sum(p.values()) == pytest.approx(1)
    assert a["tone"]["choice"] == max(p, key=p.get)
    assert a["tone"]["confidence"] == pytest.approx(choice_confidence(list(p.values())))
    n = 3
    assert a["tone"]["confidence"] == pytest.approx((n * max(p.values()) - 1) / (n - 1))
    assert 0 <= a["q"]["noul"] <= 1
    assert 0 <= a["s"]["score"] <= 2 and set(a["s"]["legend"]) == {"0", "1", "2"}
    box = a["b"]["box"]
    assert 0 <= box["top"] <= box["bottom"] <= 1 and 0 <= box["left"] <= box["right"] <= 1
    assert r.json()["spark"]["prefix_cache"] == "miss"
    r2 = c.post("/v1/systemone", json=req)
    assert r2.json()["spark"]["prefix_cache"] == "hit"
    assert r2.json()["answers"]["tone"]["probabilities"] == pytest.approx(p)


def test_prefix_cache_matches_uncached(client):
    c, be = client
    pre_c, _ = be.prefix("some state", None)
    pre_u, _ = be.prefix("some state", None, use_cache=False)
    from spark_one.prompt import question_text

    q, sp = question_text("choice", "Pick", {"x": None, "y": None})
    a = be.answer_steps(pre_c, q, sp, [("A", "B")])
    b = be.answer_steps(pre_u, q, sp, [("A", "B")])
    assert a[0] == pytest.approx(b[0], abs=1e-5)


def test_errors(client):
    c, _ = client
    assert c.post("/v1/systemone", json={"state": "x", "questions": {"b": {"type": "box", "instructions": "x"}}}).status_code == 400
    assert c.post("/v1/systemone", json={"state": "x", "questions": {"c": {"type": "choice", "instructions": "x", "criteria": {"a": None}}}}).status_code == 400
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
