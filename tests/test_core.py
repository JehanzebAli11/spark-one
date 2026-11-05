import math

import numpy as np
import pytest

from spark_one.confidence import box_confidence, choice_confidence, score_confidence
from spark_one.eval import metrics as M
from spark_one.steps import BIN_LABELS, bin_center, bin_index, box_iou, decode_box


def test_choice_confidence_matches_formula():
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    for n in (2, 3, 5, 10):
        p = [0.7] + [0.3 / (n - 1)] * (n - 1)
        assert choice_confidence(p) == pytest.approx((n * 0.7 - 1) / (n - 1))
        assert choice_confidence([1 / n] * n) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0]) == 1.0


def test_score_confidence_doc_example():
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.355, abs=0.002)
    assert score_confidence([0, 1, 0, 0]) == 1.0
    assert score_confidence([0.5, 0, 0.5]) < score_confidence([0.5, 0.5, 0])


def test_bins_and_iou():
    assert len(BIN_LABELS) == 32
    assert bin_index(0.0) == 0 and bin_index(1.0) == 31 and bin_index(0.5) == 15
    assert abs(bin_center(bin_index(0.37)) - 0.37) <= 1 / 64
    one_hot = lambda i: [1.0 if j == i else 0.0 for j in range(32)]
    b = decode_box([one_hot(4), one_hot(8), one_hot(20), one_hot(24)])
    assert b["top"] < b["bottom"] and b["left"] < b["right"]
    assert box_iou(b, b) == pytest.approx(1.0)
    assert box_iou({"top": 0, "left": 0, "bottom": .5, "right": .5}, {"top": .5, "left": .5, "bottom": 1, "right": 1}) == 0
    assert 0 <= box_confidence([one_hot(1)] * 4) <= 1


def test_metrics():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 20000)
    correct = (rng.uniform(size=20000) < conf).astype(float)
    assert M.ece(conf, correct) < 0.02
    assert M.ece(np.ones(100), np.zeros(100)) == pytest.approx(1.0)
    cov = M.coverage_curve(conf, correct)
    assert cov[0]["accuracy"] >= cov[-1]["accuracy"]
    assert M.brier([np.array([1.0, 0.0])], [np.array([1.0, 0.0])]) == 0
    assert M.log_loss([np.array([0.5, 0.5])], [0]) == pytest.approx(math.log(2))
