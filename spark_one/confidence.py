"""Confidence statistics of an answer's own probabilities (not P(correct)).

choice: (n * p_top - 1) / (n - 1)          == (p_top - 1/n) / (1 - 1/n)
score:  max(0, 1 - sum_i p_i |i - m| / MAD_unif),  MAD_unif = mean_i |i - (n-1)/2|
noul:   no confidence (|2p - 1| is offered as an optional comparable number)
box:    Spark-1 convention: mean of the choice confidence of the 4 coordinate steps.
"""

from __future__ import annotations

from typing import Sequence


def choice_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1.0) / (n - 1.0)))


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    m = max(range(n), key=lambda i: probs[i])
    mad = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - mad / mad_unif)


def noul_strength(p: float) -> float:
    return abs(2 * p - 1)


def box_confidence(step_probs: Sequence[Sequence[float]]) -> float:
    return sum(choice_confidence(p) for p in step_probs) / len(step_probs)
