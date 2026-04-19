"""Huấn luyện mô hình phân loại phân khúc giá (SVM, Random Forest)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm_pipeline(
    *,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    random_state: int | None = 42,
) -> Pipeline:
    """Pipeline chuẩn hóa + SVM."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest(
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int | None = 42,
    class_weight: str | dict | None = "balanced",
) -> RandomForestClassifier:
    """Random Forest cho bài toán phân loại."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
    )


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit mô hình scikit-learn trên tập huấn luyện."""
    model.fit(X_train, y_train)
    return model


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Dự đoán nhãn."""
    return model.predict(X)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Xác suất lớp (nếu mô hình hỗ trợ)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError("Mô hình không có predict_proba; dùng SVM với probability=True hoặc RF.")
