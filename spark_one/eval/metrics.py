"""Calibration and selective-prediction metrics (pure numpy)."""

from __future__ import annotations

import numpy as np


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on top-1 confidence."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability_bins(conf, correct, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(correct[m].mean()) if m.any() else None})
    return out


def brier(probs: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - y_k)^2 (y may be soft)."""
    if not probs:
        return float("nan")
    return float(np.mean([np.sum((p - t) ** 2) for p, t in zip(probs, targets)]))


def log_loss(probs: list[np.ndarray], target_idx: list[int], eps: float = 1e-12) -> float:
    if not probs:
        return float("nan")
    return float(np.mean([-np.log(max(p[t], eps)) for p, t in zip(probs, target_idx)]))


def coverage_curve(conf: np.ndarray, correct: np.ndarray, points: int = 20):
    """Accuracy of the most confident fraction c of answers, for c in (0, 1]."""
    if len(conf) == 0:
        return []
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    out = []
    for c in np.linspace(1 / points, 1, points):
        k = max(1, int(round(c * len(conf))))
        out.append({"coverage": float(c), "accuracy": float(cum[k - 1]),
                    "threshold": float(conf[order][k - 1])})
    return out


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    return float(risk.mean())


def softmax_logp(logp: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logp / temp
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
