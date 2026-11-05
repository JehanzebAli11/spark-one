"""Latency benchmark of POST /v1/systemone: 4 questions sharing one state, prefix cache on/off.

python scripts/bench_server.py --backend tiny --ckpt runs/tiny/student_kd/final.pt --out runs/tiny/serve_bench.md
python scripts/bench_server.py --backend hf --model google/gemma-4-E2B-it --adapter runs/.../final_adapter --out ...
"""

import argparse
import base64
import random
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from spark_one.schema import load_items
from spark_one.serve.app import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
ap.add_argument("--ckpt")
ap.add_argument("--model")
ap.add_argument("--adapter")
ap.add_argument("--bank", default="data/generated/bank")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.backend == "tiny":
    from spark_one.serve.backends import TinyBackend

    be = TinyBackend(a.ckpt, threads=4)
else:
    from spark_one.hf import HFBackend

    be = HFBackend(a.model, a.adapter)
client = TestClient(create_app(be))
root = Path(a.bank)
items = [it for it in load_items(root / "items.jsonl") if it.split == "test" and it.image]
by_img = {}
for it in items:
    by_img.setdefault(it.image, []).append(it)
scenes = [v for v in by_img.values() if len(v) >= 3]
random.Random(0).shuffle(scenes)
scenes = scenes[: a.n]


def request(group):
    qs = {}
    for i, it in enumerate(group[:4]):
        q = {"type": it.primitive, "instructions": it.instructions}
        if it.criteria is not None:
            q["criteria"] = it.criteria
        qs[f"q{i}"] = q
    data = base64.b64encode((root / group[0].image).read_bytes()).decode()
    return {"state": group[0].state, "questions": qs, "assets": {"img": {"type": "image", "data_base64": data}}}


rows = {}
for mode in ("cold (no prefix cache)", "warm (prefix cached)"):
    lat = []
    for g in scenes:
        body = request(g)
        if mode.startswith("cold"):
            be.cache.d.clear()
        else:
            client.post("/v1/systemone", json=body)  # fill the cache
        t = time.perf_counter()
        r = client.post("/v1/systemone", json=body)
        lat.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200, r.text
    rows[mode] = lat
lines = [f"# Server latency: {be.name}", "", f"{len(scenes)} requests, each 3-4 questions about one image + state.", "",
         "| mode | mean ms | p50 ms | p95 ms |", "|---|---|---|---|"]
for mode, lat in rows.items():
    lat.sort()
    lines.append(f"| {mode} | {statistics.mean(lat):.1f} | {lat[len(lat) // 2]:.1f} | {lat[int(len(lat) * 0.95) - 1]:.1f} |")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
