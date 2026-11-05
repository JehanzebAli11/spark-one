from pathlib import Path

import pytest

from spark_one.bank import build_task_items, write_bank
from spark_one.generators import REGISTRY


@pytest.fixture(scope="session")
def small_bank(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bank")
    items, tasks = [], {}
    for name, (fn, ver, langs) in REGISTRY.items():
        its = build_task_items(name, fn, ver, langs, 6, 123, out / "images")
        items += its
        tasks[name] = {"version": ver}
    write_bank(items, out, 123, 6, tasks)
    return out


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory) -> str:
    import torch

    from spark_one.tiny.model import TinyConfig, TinySystemOne

    torch.manual_seed(0)
    c = TinyConfig(d_model=32, n_layers=2, n_heads=2)
    m = TinySystemOne(c)
    p = tmp_path_factory.mktemp("ck") / "final.pt"
    torch.save({"config": c.to_dict(), "state_dict": m.state_dict()}, p)
    return str(p)
