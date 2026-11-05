"""A fake vLLM OpenAI-compatible server for tests and the CPU loop proof.

`make_app(scorer)` serves POST /v1/chat/completions. A scorer maps
(text, has_image, image_bytes, assistant_prefix) -> list of (token, logprob) for the first
generated token. Two scorers ship:
  CannedScorer      fixed distributions (unit tests: token variants, junk, failures)
  TinyModelScorer   a trained tiny model checkpoint (end-to-end distillation proof)
"""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Callable

from fastapi import FastAPI, HTTPException, Request

Scorer = Callable[[str, bool, bytes | None, str], list[tuple[str, float]]]


def _split_messages(messages):
    text, img, prefix = "", None, ""
    for m in messages:
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, str):
                text += c
            else:
                for part in c:
                    if part["type"] == "text":
                        text += part["text"]
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        img = base64.b64decode(url.split(",", 1)[1])
        elif m["role"] == "assistant":
            prefix = m["content"]
    return text, img, prefix


def make_app(scorer: Scorer, model_name: str = "fake-teacher", fail_every: int = 0) -> FastAPI:
    app = FastAPI()
    state = {"n": 0}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state["n"] += 1
        if fail_every and state["n"] % fail_every == 0:
            raise HTTPException(503, "simulated overload")
        if body.get("messages", [{}])[-1].get("role") == "assistant" and not body.get("continue_final_message"):
            raise HTTPException(400, "assistant prefill requires continue_final_message")
        text, img, prefix = _split_messages(body["messages"])
        top = sorted(scorer(text, img is not None, img, prefix), key=lambda t: -t[1])
        k = min(int(body.get("top_logprobs", 0) or 0), 20)
        if body.get("top_logprobs", 0) > 20:
            raise HTTPException(400, "top_logprobs > max_logprobs (20)")
        tok, lp = top[0]
        return {
            "id": f"cmpl-{state['n']}", "object": "chat.completion", "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": tok},
                         "finish_reason": "length",
                         "logprobs": {"content": [{"token": tok, "logprob": lp,
                                                   "top_logprobs": [{"token": t, "logprob": l} for t, l in top[:k]]}]}}],
            "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 1},
        }

    return app


class CannedScorer:
    def __init__(self, table: dict[str, dict[str, float]] | None = None, default=None):
        self.table = table or {}
        self.default = default or {"A": 0.7, " B": 0.2, "Hello": 0.1}

    def __call__(self, text, has_image, img, prefix):
        dist = self.default
        for k, v in self.table.items():
            if k in text or (prefix and k in prefix):
                dist = v
        return [(t, math.log(p)) for t, p in dist.items()]


class TinyModelScorer:
    """Serves a trained tiny model: first-token distribution over its answer vocabulary."""

    def __init__(self, ckpt_path: str, threads: int = 2):
        import torch

        from ..tiny.model import TinyConfig, TinySystemOne

        torch.set_num_threads(threads)
        ck = torch.load(ckpt_path, weights_only=False)
        self.model = TinySystemOne(TinyConfig(**ck["config"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.torch = torch

    def __call__(self, text, has_image, img, prefix):
        import numpy as np
        from PIL import Image

        from ..tiny.tokenizer import ANS, ANSWER_IDS, BOS, ID_LABEL, IMG, encode_text, label_id

        c = self.model.c
        ids = [BOS] + ([IMG] * c.n_patches if has_image else []) + encode_text(text) + [ANS]
        for seg, lab in re.findall(r"(\s?\w+=)(\S)?", prefix):  # teacher-forced Box steps
            ids += encode_text(seg) + ([label_id(lab)] if lab else [])
        t = self.torch
        images = None
        if has_image:
            im = Image.open(io.BytesIO(img)).convert("RGB").resize((c.img_size, c.img_size), Image.BILINEAR)
            images = t.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 255.0
        with t.no_grad():
            logits, _ = self.model(t.tensor([ids]), images)
        sub = logits[0, -1, list(ANSWER_IDS)]
        logp = t.log_softmax(sub, -1).tolist()
        return [(ID_LABEL[i], lp) for i, lp in zip(ANSWER_IDS, logp)]


def main(argv=None):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    uvicorn.run(make_app(TinyModelScorer(a.ckpt, a.threads), "tiny-teacher"), host="127.0.0.1", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()
