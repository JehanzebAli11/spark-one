"""System One HTTP API: POST /v1/systemone (request/response shape modelled on TypeSafe's
public System One API), with prefix caching of the shared state across questions.

SPARK1_BACKEND=tiny SPARK1_CKPT=runs/tiny/student_kd/final.pt SPARK1_TEMPS=.../temps.json \
    uvicorn spark_one.serve.app:app --port 8100
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..calibrate import apply_temperature
from ..confidence import box_confidence, choice_confidence, score_confidence
from ..prompt import question_text
from ..steps import BIN_LABELS, LETTERS, NOUL_LABELS, decode_box


class Asset(BaseModel):
    type: Literal["image"] = "image"
    data_base64: str
    format: Literal["png", "jpg", "jpeg"] = "png"


class Question(BaseModel):
    type: Literal["choice", "noul", "score", "box"]
    instructions: str | dict | list
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneRequest(BaseModel):
    model: str = "spark-1"
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1)
    assets: dict[str, Asset] | None = None


def _text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)


def _probs(lp, temp):
    if temp != 1.0:
        lp = apply_temperature(lp, temp)
    p = np.exp(np.asarray(lp, dtype=np.float64))
    return (p / p.sum()).tolist()


def create_app(backend, temps: dict | None = None) -> FastAPI:
    app = FastAPI(title="Spark-1 System One")
    temps = temps or {}
    T = lambda prim: temps.get(prim, {}).get("temperature", 1.0)

    @app.get("/healthz")
    def health():
        return {"ok": True, "model": backend.name, "prefix_cache": {"hits": backend.cache.hits,
                                                                    "misses": backend.cache.misses}}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": backend.name}]}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        t0 = time.time()
        state = _text(req.state)
        images = [base64.b64decode(a.data_base64) for a in (req.assets or {}).values() if a.type == "image"]
        if len(images) > 1:
            raise HTTPException(400, "this model accepts at most one image asset")
        img = images[0] if images else None
        pre, hit = backend.prefix(state, img)
        answers, out_tokens = {}, 0
        for name, q in req.questions.items():
            crit = q.criteria
            instr = _text(q.instructions)
            if q.type == "choice":
                if isinstance(crit, list):
                    crit = {c: None for c in crit}
                if not crit or not 2 <= len(crit) <= len(LETTERS):
                    raise HTTPException(400, f"{name}: choice needs 2..{len(LETTERS)} options")
                labels = [tuple(LETTERS[: len(crit)])]
            elif q.type == "score":
                if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                    raise HTTPException(400, f"{name}: score criteria must be a list of 2..10 levels")
                labels = [tuple(str(i) for i in range(len(crit)))]
            elif q.type == "noul":
                if isinstance(crit, list):
                    raise HTTPException(400, f"{name}: noul criteria must be {{'true':..,'false':..}}")
                labels = [NOUL_LABELS]
            else:
                if img is None:
                    raise HTTPException(400, f"{name}: box needs an image asset")
                crit = None
                labels = [BIN_LABELS] * 4
            qtext, sp = question_text(q.type, instr, crit)
            lps = backend.answer_steps(pre, qtext, sp, labels)
            out_tokens += len(lps)
            if q.type == "choice":
                p = _probs(lps[0], T("choice"))
                keys = list(crit)
                answers[name] = {"type": "choice", "choice": keys[int(np.argmax(p))],
                                 "probabilities": dict(zip(keys, p)), "confidence": choice_confidence(p)}
            elif q.type == "score":
                p = _probs(lps[0], T("score"))
                answers[name] = {"type": "score", "score": float(sum(i * x for i, x in enumerate(p))),
                                 "legend": {str(i): d for i, d in enumerate(crit)},
                                 "probabilities": {str(i): x for i, x in enumerate(p)},
                                 "confidence": score_confidence(p)}
            elif q.type == "noul":
                p = _probs(lps[0], T("noul"))
                answers[name] = {"type": "noul", "noul": p[0]}
            else:
                ps = [_probs(lp, T("box")) for lp in lps]
                answers[name] = {"type": "box", "box": decode_box(ps), "confidence": box_confidence(ps)}
        return {"model": backend.name, "answers": answers,
                "usage": {"input_tokens": pre[1] if isinstance(pre, tuple) else None, "output_tokens": out_tokens},
                "spark": {"prefix_cache": "hit" if hit else "miss", "latency_ms": round((time.time() - t0) * 1000, 2)}}

    return app


def _from_env():
    kind = os.environ.get("SPARK1_BACKEND", "tiny")
    if kind == "tiny":
        from .backends import TinyBackend

        backend = TinyBackend(os.environ.get("SPARK1_CKPT", "runs/tiny/student_kd/final.pt"))
    elif kind == "hf":
        from ..hf import HFBackend

        backend = HFBackend(os.environ["SPARK1_BASE_MODEL"], os.environ.get("SPARK1_ADAPTER"))
    else:
        raise SystemExit(f"unknown SPARK1_BACKEND {kind}")
    tp = os.environ.get("SPARK1_TEMPS")
    temps = json.loads(open(tp).read()) if tp and os.path.exists(tp) else {}
    return create_app(backend, temps)


def __getattr__(name):  # lazy so importing the module does not load a model
    if name == "app":
        return _from_env()
    raise AttributeError(name)
