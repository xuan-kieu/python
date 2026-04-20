"""Đánh giá mô hình: F1-Score và các metric liên quan."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    f1_score,
    roc_curve,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import label_binarize
from sklearn.base import clone



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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Vẽ confusion matrix cho 4 lớp giá."""
    display_labels = labels or ["Low", "Mid", "High", "Very High"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=display_labels,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """
    Vẽ ROC đa lớp (one-vs-rest) cho 4 lớp trên cùng một biểu đồ.
    In ra macro-AUC (trung bình AUC theo lớp).
    """
    classes = np.array([0, 1, 2, 3])
    y_test_bin = label_binarize(y_test, classes=classes)

    # Áp dụng one-vs-rest tường minh cho bài toán đa lớp.
    ovr_model = OneVsRestClassifier(clone(model))
    ovr_model.fit(X_test, y_test)
    y_score = ovr_model.predict_proba(X_test)

    if y_score.shape[1] != len(classes):
        raise ValueError("Số cột điểm dự đoán không khớp số lớp trong y_test.")

    class_names = ["Low", "Mid", "High", "Very High"]
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_auc_scores: list[float] = []

    for i, class_id in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        class_auc = auc(fpr, tpr)
        roc_auc_scores.append(class_auc)
        label_name = class_names[i] if i < len(class_names) else str(class_id)
        ax.plot(fpr, tpr, label=f"{label_name} (AUC={class_auc:.3f})")

    macro_auc = float(np.mean(roc_auc_scores))
    print(f"Macro-AUC (OvR): {macro_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Vẽ learning curve: train size vs F1 weighted (train/cv)."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, marker="o", label="Train F1 (weighted)")
    ax.plot(train_sizes, val_mean, marker="o", label="Cross-val F1 (weighted)")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("F1 Score")
    ax.set_title("Learning Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curve(
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: list[float] | list[int],
    save_path: str | Path | None = None,
) -> None:
    """
    Vẽ validation curve cho SVM:
    - param_name='C', param_range=[0.01, 0.1, 1, 10, 100, 1000]
    """
    from sklearn.svm import SVC

    if param_name != "C":
        raise ValueError("param_name không hỗ trợ. Dùng 'C' cho SVM.")

    estimator = SVC(kernel="rbf", gamma="scale", probability=True)
    title = "Validation Curve (SVM)"

    train_scores, val_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_range, train_mean, marker="o", label="Train F1 (weighted)")
    ax.plot(param_range, val_mean, marker="o", label="Cross-val F1 (weighted)")
    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
