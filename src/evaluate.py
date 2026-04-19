"""Đánh giá mô hình: F1-Score và các metric liên quan."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import f1_score


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Literal["micro", "macro", "weighted", "binary"] = "weighted",
    labels: np.ndarray | None = None,
) -> float:
    """
    Tính F1-Score.

    - weighted: phù hợp đa lớp, có trọng số theo support.
    - macro: trung bình không trọng số giữa các lớp.
    """
    return float(
        f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    )


def compute_f1_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """F1 cho từng lớp (không average)."""
    return f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )
