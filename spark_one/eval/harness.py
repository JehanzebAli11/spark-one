"""Eval harness: accuracy, ECE, Brier, log loss, reliability diagrams, accuracy vs coverage,
option-order consistency, self-consistency (negation twins), latency and Box IoU, with
English / Roman Urdu / text / image slices reported separately.

python -m spark_one.eval.harness --bank data/generated/bank --preds preds.jsonl \
    --temps temps.json --out runs/x/eval --name "tiny student"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..calibrate import apply_temperature
from ..schema import load_items, read_jsonl
from ..steps import decode_box, item_steps
from . import metrics as M

CORE = ("choice", "noul", "score")


def _probs(rec, k, temp):
    if "logp" in rec and rec["logp"] is not None:
        lp = rec["logp"][k]
        if temp != 1.0:
            lp = apply_temperature(lp, temp)
        p = np.exp(np.asarray(lp, dtype=np.float64))
        return p / p.sum()
    return np.asarray(rec["probs"][k], dtype=np.float64)


def score_records(items, preds, temps):
    """One evaluation row per base prediction."""
    rows = []
    for r in preds:
        if r.get("variant", "base") != "base":
            continue
        it = items.get(r["base_id"])
        if it is None:
            continue
        t = temps.get(it.primitive, {}).get("temperature", 1.0) if temps else 1.0
        row = {"id": it.id, "primitive": it.primitive, "lang": it.lang, "modality": it.modality,
               "task": it.task, "generator": it.provenance["generator"], "none_fits": it.none_fits,
               "latency_ms": r.get("latency_ms")}
        steps = item_steps(it)
        if it.primitive == "box":
            if r.get("box"):
                pred_box = r["box"]
                row["bin_conf"] = None
            else:
                ps = [_probs(r, k, t) for k in range(4)]
                pred_box = decode_box(ps)
                row["step_conf"] = [float(p.max()) for p in ps]
                row["step_correct"] = [bool(int(np.argmax(p)) == st.target) for p, st in zip(ps, steps)]
            from ..steps import box_iou

            row["iou"] = box_iou(pred_box, it.answer)
        else:
            p = _probs(r, 0, t)
            st = steps[0]
            row["probs"] = p
            row["conf"] = float(p.max())
            if st.target is not None:
                y = np.zeros_like(p)
                y[st.target] = 1
                row["target"] = st.target
                row["correct"] = bool(int(np.argmax(p)) == st.target)
                row["nll"] = float(-np.log(max(p[st.target], 1e-12)))
                row["brier"] = float(np.sum((p - y) ** 2))
                if it.primitive == "score":
                    row["abs_err"] = float(abs(sum(i * q for i, q in enumerate(p)) - st.target))
        rows.append(row)
    return rows


def summarize(rows):
    hard = [r for r in rows if "correct" in r]
    if not hard:
        return {"n": 0}
    conf = np.array([r["conf"] for r in hard])
    cor = np.array([r["correct"] for r in hard], dtype=float)
    out = {"n": len(hard), "accuracy": float(cor.mean()), "ece": M.ece(conf, cor),
           "brier": float(np.mean([r["brier"] for r in hard])),
           "log_loss": float(np.mean([r["nll"] for r in hard])),
           "mean_confidence": float(conf.mean()), "aurc": M.aurc(conf, cor)}
    cov = M.coverage_curve(conf, cor)
    out["acc_at_50pct_coverage"] = cov[9]["accuracy"] if len(cov) >= 10 else None
    lat = [r["latency_ms"] for r in hard if r.get("latency_ms") is not None]
    if lat:
        out["latency_ms_mean"] = float(np.mean(lat))
    return out


def box_summary(rows):
    b = [r for r in rows if r["primitive"] == "box"]
    if not b:
        return {"n": 0}
    ious = np.array([r["iou"] for r in b])
    out = {"n": len(b), "mean_iou": float(ious.mean()), "iou_at_0.5": float((ious >= 0.5).mean()),
           "iou_at_0.75": float((ious >= 0.75).mean())}
    sc = [c for r in b if r.get("step_conf") for c in r["step_conf"]]
    if sc:
        sk = [c for r in b if r.get("step_correct") for c in r["step_correct"]]
        out["bin_step_accuracy"] = float(np.mean(sk))
        out["bin_step_ece"] = M.ece(np.array(sc), np.array(sk, dtype=float))
    return out


def order_consistency(items, preds, temps):
    base = {r["base_id"]: r for r in preds if r.get("variant") == "base"}
    agree, tv = [], []
    for r in preds:
        if not str(r.get("variant", "")).startswith("perm") or r["base_id"] not in base:
            continue
        t = temps.get("choice", {}).get("temperature", 1.0) if temps else 1.0
        pb = _probs(base[r["base_id"]], 0, t)
        pp = _probs(r, 0, t)
        mapped = np.zeros_like(pp)
        for i, j in enumerate(r["order"]):
            mapped[j] = pp[i]
        agree.append(int(np.argmax(mapped)) == int(np.argmax(pb)))
        tv.append(0.5 * float(np.abs(mapped - pb).sum()))
    if not agree:
        return {"n": 0}
    return {"n": len(agree), "argmax_agreement": float(np.mean(agree)), "mean_tv_distance": float(np.mean(tv))}


def self_consistency(items, preds, temps):
    base = {r["base_id"]: r for r in preds if r.get("variant") == "base"}
    gaps, agree = [], []
    t = temps.get("noul", {}).get("temperature", 1.0) if temps else 1.0
    for r in preds:
        if r.get("variant") != "neg" or r["base_id"] not in base:
            continue
        p_yes = _probs(base[r["base_id"]], 0, t)[0]
        p_yes_neg = _probs(r, 0, t)[0]
        gaps.append(abs(p_yes - (1 - p_yes_neg)))
        agree.append((p_yes > 0.5) != (p_yes_neg > 0.5))
    if not gaps:
        return {"n": 0}
    return {"n": len(gaps), "mean_gap": float(np.mean(gaps)), "logical_agreement": float(np.mean(agree))}


def evaluate(items, preds, temps=None):
    rows = score_records(items, preds, temps or {})
    core = [r for r in rows if r["primitive"] in CORE]
    rep = {"overall_core": summarize(core), "by_primitive": {}, "slices": {}, "by_task": {}}
    for p in CORE:
        rep["by_primitive"][p] = summarize([r for r in rows if r["primitive"] == p])
    rep["by_primitive"]["box"] = box_summary(rows)
    sc = [r for r in rows if r["primitive"] == "score" and "abs_err" in r]
    if sc:
        rep["by_primitive"]["score"]["mae_expected_level"] = float(np.mean([r["abs_err"] for r in sc]))
    for name, f in (("en", lambda r: r["lang"] == "en"), ("ur-roman", lambda r: r["lang"] == "ur-roman"),
                    ("text", lambda r: r["modality"] == "text"), ("image", lambda r: r["modality"] == "image")):
        sub = [r for r in rows if f(r)]
        rep["slices"][name] = {"core": summarize([r for r in sub if r["primitive"] in CORE]),
                               "box": box_summary(sub)}
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        rep["by_task"][task] = box_summary(rs) if rs[0]["primitive"] == "box" else summarize(rs)
    nf = [r for r in rows if r["none_fits"]]
    if nf:
        rep["none_fits_choice"] = {"n": len(nf), "mean_top1": float(np.mean([r["conf"] for r in nf])),
                                   "note": "no option is correct; lower top-1 is better"}
    rep["option_order_consistency"] = order_consistency(items, preds, temps or {})
    rep["self_consistency"] = self_consistency(items, preds, temps or {})
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if lat:
        rep["latency_ms"] = {"mean": float(np.mean(lat)), "p50": float(np.percentile(lat, 50)),
                             "p95": float(np.percentile(lat, 95)), "n": len(lat)}
    return rep, rows


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(name, rep_raw, rep_cal, temps):
    L = [f"# Eval: {name}", ""]
    if temps:
        L.append("Temperatures (fitted on val): " + ", ".join(f"{k} {v['temperature']:.3f}" for k, v in temps.items()))
        L.append("")
    L += ["## Core primitives (choice, noul, score)", "",
          "| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |",
          "|---|---|---|---|---|---|---|---|"]
    for p in list(CORE) + ["overall"]:
        a = rep_raw["overall_core"] if p == "overall" else rep_raw["by_primitive"][p]
        b = rep_cal["overall_core"] if p == "overall" else rep_cal["by_primitive"][p]
        if not a.get("n"):
            continue
        L.append(f"| {p} | {a['n']} | {_fmt(a['accuracy'])} | {_fmt(a['ece'])} | {_fmt(b['ece'])} | "
                 f"{_fmt(b['brier'])} | {_fmt(b['log_loss'])} | {_fmt(b.get('acc_at_50pct_coverage'))} |")
    bx = rep_cal["by_primitive"]["box"]
    if bx.get("n"):
        L += ["", "## Box", "", f"n {bx['n']}, mean IoU {_fmt(bx['mean_iou'])}, IoU>=0.5 {_fmt(bx['iou_at_0.5'])}, "
              f"IoU>=0.75 {_fmt(bx['iou_at_0.75'])}, bin-step accuracy {_fmt(bx.get('bin_step_accuracy'))}, "
              f"bin-step ECE {_fmt(bx.get('bin_step_ece'))}"]
    L += ["", "## Slices (calibrated)", "", "| slice | n core | accuracy | ECE | Brier | box mean IoU |", "|---|---|---|---|---|---|"]
    for s, v in rep_cal["slices"].items():
        c, b = v["core"], v["box"]
        L.append(f"| {s} | {c.get('n', 0)} | {_fmt(c.get('accuracy'))} | {_fmt(c.get('ece'))} | {_fmt(c.get('brier'))} | {_fmt(b.get('mean_iou'))} |")
    oc, sc = rep_cal["option_order_consistency"], rep_cal["self_consistency"]
    L += ["", "## Consistency", "",
          f"- Option-order consistency (choice, 2 rotations per item): argmax agreement {_fmt(oc.get('argmax_agreement'))}, "
          f"mean TV distance {_fmt(oc.get('mean_tv_distance'))} (n {oc.get('n', 0)})",
          f"- Self-consistency (noul vs negated twin): logical agreement {_fmt(sc.get('logical_agreement'))}, "
          f"mean |p - (1 - p_neg)| {_fmt(sc.get('mean_gap'))} (n {sc.get('n', 0)})"]
    if rep_cal.get("none_fits_choice"):
        nf = rep_cal["none_fits_choice"]
        L.append(f"- None-fits choice items: mean top-1 {_fmt(nf['mean_top1'])} (n {nf['n']}; lower is better)")
    if rep_cal.get("latency_ms"):
        lt = rep_cal["latency_ms"]
        L.append(f"- Latency per item (batched CPU): mean {_fmt(lt['mean'], 2)} ms, p50 {_fmt(lt['p50'], 2)}, p95 {_fmt(lt['p95'], 2)}")
    L += ["", "## By task (calibrated)", "", "| task | n | accuracy / mean IoU | ECE |", "|---|---|---|---|"]
    for t, v in rep_cal["by_task"].items():
        if not v.get("n"):
            continue
        L.append(f"| {t} | {v['n']} | {_fmt(v.get('accuracy', v.get('mean_iou')))} | {_fmt(v.get('ece', v.get('bin_step_ece')))} |")
    return "\n".join(L) + "\n"


def figures(name, rows_raw, rows_cal, out: Path):
    from . import plots

    def panel(rows, prim):
        h = [r for r in rows if r["primitive"] == prim and "correct" in r]
        if not h:
            return None
        c = np.array([r["conf"] for r in h]); k = np.array([r["correct"] for r in h], dtype=float)
        return (prim, M.reliability_bins(c, k), M.ece(c, k))

    for tag, rows in (("raw", rows_raw), ("calibrated", rows_cal)):
        panels = [p for p in (panel(rows, x) for x in CORE) if p]
        b = [r for r in rows if r.get("step_conf")]
        if b:
            c = np.array([x for r in b for x in r["step_conf"]]); k = np.array([x for r in b for x in r["step_correct"]], dtype=float)
            panels.append(("box bins", M.reliability_bins(c, k), M.ece(c, k)))
        if panels:
            plots.reliability(panels, out / f"reliability_{tag}.png", f"{name}: reliability ({tag})")
    curves = {}
    for prim in CORE:
        h = [r for r in rows_cal if r["primitive"] == prim and "correct" in r]
        if h:
            curves[prim] = M.coverage_curve(np.array([r["conf"] for r in h]), np.array([r["correct"] for r in h], dtype=float))
    plots.coverage(curves, out / "accuracy_vs_coverage.png", f"{name}: accuracy vs coverage")
    srows = []
    for s, f in (("en", lambda r: r["lang"] == "en"), ("ur-roman", lambda r: r["lang"] == "ur-roman"),
                 ("text", lambda r: r["modality"] == "text"), ("image", lambda r: r["modality"] == "image")):
        sm = summarize([r for r in rows_cal if f(r) and r["primitive"] in CORE])
        if sm.get("n"):
            srows.append((s, sm["accuracy"], sm["ece"]))
    if srows:
        plots.slice_bars(srows, out / "slices.png", f"{name}: slices (choice+noul+score, calibrated)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--temps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    items = {it.id: it for it in load_items(Path(a.bank) / "items.jsonl")}
    preds = list(read_jsonl(a.preds))
    temps = json.loads(Path(a.temps).read_text()) if a.temps else {}
    rep_raw, rows_raw = evaluate(items, preds, None)
    rep_cal, rows_cal = evaluate(items, preds, temps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"name": a.name, "temperatures": temps, "raw": rep_raw,
                                                 "calibrated": rep_cal}, indent=2) + "\n")
    (out / "report.md").write_text(markdown(a.name, rep_raw, rep_cal, temps))
    if not a.no_figures:
        figures(a.name, rows_raw, rows_cal, out)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
