"""Evaluation metrics helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from .logging import get_logger

LOGGER = get_logger(__name__)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC score using scikit-learn."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    LOGGER.info("ROC-AUC = %.4f", score)
    return float(score)


def ttfa(scores: np.ndarray, threshold: float, fps: float) -> Optional[float]:
    """Time to first alert (seconds) or None if threshold never crossed."""
    idx = np.where(scores >= threshold)[0]
    if idx.size == 0:
        LOGGER.info("No alerts triggered (threshold %.2f).", threshold)
        return None
    ttfa_value = float(idx[0] / max(fps, 1e-6))
    LOGGER.info("TTFA = %.2fs", ttfa_value)
    return ttfa_value


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_png_path: str | Path) -> Path:
    """Plot ROC curve and save to disk."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    LOGGER.info("Saved ROC plot to %s", out_path)
    return out_path


def histogram_scores(scores: np.ndarray, out_png_path: str | Path, bins: int = 20) -> Path:
    """Plot histogram of anomaly scores and save to disk."""
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.75)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    LOGGER.info("Saved score histogram to %s", out_path)
    return out_path


__all__ = ["roc_auc", "ttfa", "plot_roc", "histogram_scores"]
