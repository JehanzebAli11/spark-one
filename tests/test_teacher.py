import math
import socket
import threading
import time
from argparse import Namespace

import pytest
import uvicorn

from spark_one.schema import load_items, read_jsonl
from spark_one.teacher.fake_vllm import CannedScorer, make_app
from spark_one.teacher.openai_chat import build_messages, parse_top_logprobs
from spark_one.teacher.score import run


def _serve(app):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}/v1"


def test_parse_top_logprobs_pools_variants_and_fills_missing():
    top = [{"token": "A", "logprob": math.log(0.5)}, {"token": " A", "logprob": math.log(0.1)},
           {"token": "**B", "logprob": math.log(0.2)}, {"token": "Hello", "logprob": math.log(0.1)}]
    probs, cov = parse_top_logprobs(top, ("A", "B", "C"))
    assert cov == pytest.approx(0.8)
    assert probs[0] > probs[1] > probs[2] > 0
    assert sum(probs) == pytest.approx(1)
    p2, _ = parse_top_logprobs([{"token": "yes", "logprob": math.log(0.9)}], ("Yes", "No"), case_insensitive=True)
    assert p2[0] > 0.89


def test_box_steps_are_teacher_forced(small_bank):
    it = next(i for i in load_items(small_bank / "items.jsonl") if i.primitive == "box")
    msgs, cont = build_messages(it, small_bank, step_index=2)
    assert cont and msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"].startswith("top=") and msgs[-1]["content"].endswith(" bottom=")
    assert msgs[1]["content"][0]["type"] == "image_url"


def test_score_job_against_fake_server(small_bank, tmp_path):
    scorer = CannedScorer(default={"A": 0.6, " B": 0.25, "Yes": 0.05, "1": 0.05, "zz": 0.05})
    server, url = _serve(make_app(scorer, fail_every=7))  # every 7th request returns 503
    try:
        out = tmp_path / "t.jsonl"
        args = Namespace(bank=str(small_bank), split="train", out=str(out), base_url=url, model="fake",
                         concurrency=4, timeout=10, limit=40, primitives=None, min_coverage=0.5, top_logprobs=20)
        stats = __import__("asyncio").run(run(args))
        assert stats["failed"] == 0 and stats["scored"] == 40
        rows = list(read_jsonl(out))
        items = {i.id: i for i in load_items(small_bank / "items.jsonl")}
        for r in rows:
            it = items[r["id"]]
            for s in r["steps"]:
                assert sum(s["probs"]) == pytest.approx(1, abs=1e-4)
            if it.primitive == "choice":
                assert r["steps"][0]["probs"][0] > r["steps"][0]["probs"][1] > r["steps"][0]["probs"][-1]
                assert r["ok"]
            if it.primitive == "box":
                assert len(r["steps"]) == 4 and r["steps"][0]["probs"][0] > 0.6  # "A" is also bin 0
        # resume: nothing left to do
        stats2 = __import__("asyncio").run(run(args))
        assert stats2["scored"] == 0
    finally:
        server.should_exit = True
