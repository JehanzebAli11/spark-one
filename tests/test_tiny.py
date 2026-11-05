import json

import numpy as np
import pytest
import torch

from spark_one.schema import load_items
from spark_one.tiny import train as tiny_train
from spark_one.tiny.encode import encode_item
from spark_one.tiny.model import TinyConfig, TinySystemOne
from spark_one.tiny.tokenizer import ANS
from spark_one.train.losses import StepBatch, answer_loss


def _cfg(tmp_path, bank, steps, run="run"):
    c = {"bank": str(bank), "run_dir": str(tmp_path / run), "steps": steps, "batch_size": 4, "lr": 1e-3,
         "warmup": 2, "log_every": 1, "ckpt_every": 2, "threads": 1, "seed": 0, "alpha": 0.0,
         "model": {"d_model": 32, "n_layers": 2, "n_heads": 2}}
    p = tmp_path / f"{run}.json"
    p.write_text(json.dumps(c))
    return p


def test_answer_positions_and_teacher_forcing(small_bank):
    it = next(i for i in load_items(small_bank / "items.jsonl") if i.primitive == "box")
    e = encode_item(it, 64)
    assert len(e.step_pos) == 4
    assert e.ids[e.step_pos[0] - 4: e.step_pos[0] + 1][:1] == [ANS] or ANS in e.ids
    # the token after each non-final step position is the bytes of the next step prefix or the label
    for k in range(3):
        assert e.ids[e.step_pos[k] + 1] == e.step_labels[k][e.step_target[k]]


def test_loss_terms():
    S, L = 3, 4
    logits = torch.randn(S, L)
    sb = StepBatch(row=torch.zeros(S, dtype=torch.long), pos=torch.zeros(S, dtype=torch.long),
                   label_ids=torch.zeros(S, L, dtype=torch.long), label_mask=torch.ones(S, L, dtype=torch.bool),
                   target=torch.tensor([0, 1, 2]), soft=torch.zeros(S, L),
                   teacher=torch.softmax(logits, -1), has_teacher=torch.ones(S, dtype=torch.bool), primitive=["c"] * 3)
    ce, _ = answer_loss(logits, sb, alpha=0.0)
    ref = torch.nn.functional.cross_entropy(logits, sb.target)
    assert ce.item() == pytest.approx(ref.item(), abs=1e-5)
    kl, parts = answer_loss(logits, sb, alpha=1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)  # student == teacher
    mixed, _ = answer_loss(logits, sb, alpha=0.5)
    assert mixed.item() == pytest.approx(0.5 * ref.item(), abs=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = TinySystemOne(TinyConfig(d_model=32, n_layers=2, n_heads=2)).eval()
    ids = torch.randint(0, 256, (1, 30))
    with torch.no_grad():
        full, _ = m(ids)
        _, past = m(ids[:, :18])
        part, _ = m(ids[:, 18:], past=past, offset=18)
    assert torch.allclose(full[:, 18:], part, atol=1e-5)


def test_train_checkpoint_and_resume_is_exact(small_bank, tmp_path):
    cfg = _cfg(tmp_path, small_bank, 4, "a")
    tiny_train.main(["--config", str(cfg)])
    a = torch.load(tmp_path / "a" / "final.pt")["state_dict"]
    cfg_b = _cfg(tmp_path, small_bank, 4, "b")
    tiny_train.main(["--config", str(cfg_b), "--max_steps", "2"])  # "crash" after step 2
    assert (tmp_path / "b" / "checkpoint-2").exists() and not (tmp_path / "b" / "final.pt").exists()
    tiny_train.main(["--config", str(cfg_b), "--resume_from", str(tmp_path / "b" / "checkpoint-2")])
    b = torch.load(tmp_path / "b" / "final.pt")["state_dict"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_predict_and_harness(small_bank, tiny_ckpt, tmp_path):
    from spark_one.calibrate import fit
    from spark_one.eval.harness import evaluate
    from spark_one.eval.variants import expand
    from spark_one.tiny.predict import load_model, predict

    items = [i for i in load_items(small_bank / "items.jsonl") if i.split != "train"][:60]
    var = expand(items)
    res = predict(load_model(tiny_ckpt), [v[3] for v in var], small_bank, batch_size=16)
    preds = [{"id": vid, "base_id": vid.split("~")[0], "variant": v, "order": o, "logp": lp, "latency_ms": dt * 1000}
             for (vid, v, o, _), (lp, dt) in zip(var, res)]
    byid = {i.id: i for i in items}
    temps = fit(byid, preds)
    rep, _ = evaluate(byid, preds, temps)
    assert 0 <= rep["overall_core"]["accuracy"] <= 1
    assert rep["option_order_consistency"]["n"] > 0
    # a perfect oracle gets accuracy 1 and full consistency
    from spark_one.steps import item_steps

    oracle = []
    for vid, v, o, it in var:
        lps = []
        for st in item_steps(it):
            p = st.soft if st.soft is not None else [1.0 if j == st.target else 1e-6 for j in range(len(st.labels))]
            lps.append(list(np.log(np.asarray(p) / sum(p))))
        oracle.append({"id": vid, "base_id": vid.split("~")[0], "variant": v, "order": o, "logp": lps})
    rep, _ = evaluate(byid, oracle)
    assert rep["overall_core"]["accuracy"] == 1.0
    assert rep["option_order_consistency"]["argmax_agreement"] == 1.0
    if rep["self_consistency"]["n"]:
        assert rep["self_consistency"]["logical_agreement"] == 1.0
    if rep["by_primitive"]["box"]["n"]:
        assert rep["by_primitive"]["box"]["mean_iou"] > 0.8


def test_temperature_scaling_fixes_overconfidence():
    from spark_one.calibrate import fit_temperature, nll_at

    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(2000):
        y = int(rng.integers(4))
        z = rng.normal(0, 1, 4)
        z[y] += 1.0
        pairs.append((z * 5.0, y))  # logits blown up by 5x -> overconfident
    t = fit_temperature(pairs)
    assert t > 2.5
    assert nll_at(pairs, t) < nll_at(pairs, 1.0)
