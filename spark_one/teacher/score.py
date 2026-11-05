"""Teacher-score job: read option-token logprobs from a vLLM OpenAI-compatible server
(Gemma 4 31B on the Spark) and write soft labels.

python -m spark_one.teacher.score --bank data/generated/bank --split train \
    --out runs/teacher/train.jsonl --base-url http://localhost:8000/v1 --model google/gemma-4-31B-it

Resumable: ids already in --out are skipped. Rows with low label coverage are kept with
ok=false and ignored by training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

from ..schema import load_items, read_jsonl
from ..steps import item_steps
from .openai_chat import build_messages, chat_body, first_token_top, parse_top_logprobs


async def post_with_retry(client: httpx.AsyncClient, url: str, body: dict, tries: int = 5) -> dict:
    delay = 1.0
    for attempt in range(tries):
        try:
            r = await client.post(url, json=body)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (httpx.TransportError, httpx.TimeoutException) as e:
            err = repr(e)
        if attempt == tries - 1:
            raise RuntimeError(err)
        await asyncio.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


async def score_item(client, url, model, item, root, min_coverage, top_logprobs):
    steps = item_steps(item)
    out_steps, ok = [], True
    t0 = time.time()
    for i, st in enumerate(steps):
        msgs, cont = build_messages(item, root, steps, i)
        resp = await post_with_retry(client, url, chat_body(model, msgs, cont, top_logprobs))
        top = first_token_top(resp)
        probs, cov = parse_top_logprobs(top, st.labels, case_insensitive=item.primitive == "noul")
        ok = ok and cov >= min_coverage
        out_steps.append({"labels": list(st.labels), "probs": [round(p, 6) for p in probs],
                          "coverage": round(cov, 4)})
    return {"id": item.id, "ok": ok, "steps": out_steps, "teacher": model,
            "latency_ms": round((time.time() - t0) * 1000, 1)}


async def run(a) -> dict:
    root = Path(a.bank)
    items = [it for it in load_items(root / "items.jsonl") if a.split in (None, "all") or it.split == a.split]
    if a.primitives:
        items = [it for it in items if it.primitive in a.primitives]
    if a.limit:
        items = random.Random(0).sample(items, min(a.limit, len(items)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [it for it in items if it.id not in done]
    print(f"{len(items)} items, {len(done)} already scored, {len(todo)} to go", flush=True)
    url = a.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"}
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"scored": 0, "ok": 0, "failed": 0}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=a.timeout, headers=headers) as client:
        with open(out, "a", encoding="utf-8") as f:
            async def one(it):
                async with sem:
                    try:
                        rec = await score_item(client, url, a.model, it, root, a.min_coverage, a.top_logprobs)
                    except Exception as e:  # keep going; the row is retried on the next run
                        stats["failed"] += 1
                        print(f"FAILED {it.id}: {e}", flush=True)
                        return
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    stats["scored"] += 1
                    stats["ok"] += rec["ok"]
                    if stats["scored"] % 500 == 0:
                        rate = stats["scored"] / (time.time() - t0)
                        print(f"{stats['scored']}/{len(todo)} ({rate:.1f}/s)", flush=True)
            await asyncio.gather(*(one(it) for it in todo))
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats), flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "google/gemma-4-31B-it"))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--primitives", nargs="*", help="e.g. choice noul score (Box needs top_logprobs>=32)")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--top-logprobs", type=int, default=20)
    a = ap.parse_args(argv)
    stats = asyncio.run(run(a))
    if stats["failed"]:
        raise SystemExit(f"{stats['failed']} items failed; re-run to retry them")


if __name__ == "__main__":
    main()
