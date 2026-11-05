"""Temperature scaling: one temperature per primitive, fitted on validation predictions by
minimising log loss. Applied to stored log-probabilities, so no model call is needed.

python -m spark_one.calibrate --bank B --preds val_preds.jsonl --out temps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import load_items, read_jsonl
from .steps import item_steps


def collect(items_by_id, preds):
    """-> {primitive: [(logp_vector, target)]} over base predictions; target is an index or,
    for Box steps, the exact two-hot distribution."""
    out: dict[str, list] = {}
    for r in preds:
        if r.get("variant", "base") != "base" or "logp" not in r:
            continue
        it = items_by_id.get(r["base_id"])
        if it is None:
            continue
        for st, lp in zip(item_steps(it), r["logp"]):
            if it.primitive == "choice" and st.target is None:
                continue  # none-fits items carry no calibration signal for the choice head
            y = np.asarray(st.soft) if st.soft is not None else st.target
            out.setdefault(it.primitive, []).append((np.asarray(lp, dtype=np.float64), y))
    return out


def nll_at(pairs, t):
    s = 0.0
    for lp, y in pairs:
        z = lp / t
        z = z - z.max()
        ls = z - np.log(np.exp(z).sum())
        s += -(float(np.dot(y, ls)) if isinstance(y, np.ndarray) else ls[y])
    return s / len(pairs)


def fit_temperature(pairs) -> float:
    if not pairs:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 121))
    vals = [nll_at(pairs, t) for t in grid]
    best = grid[int(np.argmin(vals))]
    lo, hi = best / 1.07, best * 1.07  # golden-section refine
    for _ in range(40):
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if nll_at(pairs, m1) < nll_at(pairs, m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def fit(items_by_id, preds) -> dict:
    groups = collect(items_by_id, preds)
    res = {}
    for prim, pairs in groups.items():
        t = fit_temperature(pairs)
        res[prim] = {"temperature": t, "n": len(pairs), "nll_before": nll_at(pairs, 1.0),
                     "nll_after": nll_at(pairs, t)}
    return res


def apply_temperature(logp: list[float], t: float) -> list[float]:
    z = np.asarray(logp, dtype=np.float64) / t
    z = z - z.max()
    return (z - np.log(np.exp(z).sum())).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = {it.id: it for it in load_items(Path(a.bank) / "items.jsonl")}
    res = fit(items, list(read_jsonl(a.preds)))
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
