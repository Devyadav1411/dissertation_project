"""
detection/detect.py — Anomaly detection via reconstruction error.

Algorithm
---------
1. Pass every sequence through the trained (frozen) autoencoder.
2. Compute per-sequence MSE between original and reconstruction.
3. Determine a threshold using one of two methods:
     statistical : mean + σ * std   (default σ = 3)
     percentile  : e.g. 95th percentile of the error distribution
4. Label a sequence as anomalous if its error exceeds the threshold.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from config import Config

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Scores sequences and classifies them as normal or anomalous.

    Parameters
    ----------
    model : nn.Module
        Trained TemporalAutoencoder (weights frozen during inference).
    threshold_method : str
        "statistical" (mean + k·std) or "percentile" (Nth percentile).
    percentile : int
        Percentile used when threshold_method == "percentile".
    sigma : int
        Multiplier for std when threshold_method == "statistical".
    """

    def __init__(
        self,
        model: nn.Module,
        threshold_method: str = Config.THRESHOLD_METHOD,
        percentile: int = Config.PERCENTILE,
        sigma: int = Config.STATISTICAL_SIGMA,
    ) -> None:
        self.model = model
        self.threshold_method = threshold_method
        self.percentile = percentile
        self.sigma = sigma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.threshold: float | None = None

    # ──────────────────────────────────────────────────────────────────────────

    def compute_reconstruction_error(self, data_loader: DataLoader) -> np.ndarray:
        """
        Forward-pass all batches through the autoencoder and compute per-sequence MSE.

        Parameters
        ----------
        data_loader : DataLoader
            Sequences to score (batches of shape [B, seq_len, num_features]).

        Returns
        -------
        np.ndarray  shape (N,)  — reconstruction error per sequence.
        """
        errors = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                reconstructed = self.model(batch_x)
                # Per-sequence MSE: mean over (seq_len, num_features)
                mse = ((batch_x - reconstructed) ** 2).mean(dim=(1, 2))
                errors.append(mse.cpu().numpy())

        return np.concatenate(errors)

    # ──────────────────────────────────────────────────────────────────────────

    def determine_threshold(self, errors: np.ndarray) -> float:
        """
        Compute the decision boundary from a reference error distribution.

        Typically called on **training-set** errors so the threshold is set
        without any look-ahead at anomalous data.

        Parameters
        ----------
        errors : np.ndarray
            Reconstruction errors from (normal) training sequences.

        Returns
        -------
        float — threshold value.
        """
        if self.threshold_method == "percentile":
            threshold = float(np.percentile(errors, self.percentile))
            logger.info(
                "Threshold (percentile=%d): %.6f", self.percentile, threshold
            )
        else:  # "statistical"
            threshold = float(errors.mean() + self.sigma * errors.std())
            logger.info(
                "Threshold (mean + %d·std): %.6f", self.sigma, threshold
            )

        self.threshold = threshold
        return threshold

    # ──────────────────────────────────────────────────────────────────────────

    def detect(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Full detection pipeline: score → threshold → predict.

        If :meth:`determine_threshold` has not been called yet, the threshold
        is derived from the current loader's error distribution.

        Parameters
        ----------
        data_loader : DataLoader

        Returns
        -------
        anomaly_scores : np.ndarray  shape (N,)
        predicted_labels : np.ndarray  shape (N,)  — 0 = normal, 1 = anomaly
        threshold : float
        """
        anomaly_scores = self.compute_reconstruction_error(data_loader)

        if self.threshold is None:
            self.determine_threshold(anomaly_scores)

        predicted_labels = (anomaly_scores > self.threshold).astype(int)

        logger.info(
            "Detection complete — anomalies: %d / %d (%.1f%%)",
            predicted_labels.sum(),
            len(predicted_labels),
            100.0 * predicted_labels.mean(),
        )
        return anomaly_scores, predicted_labels, self.threshold  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> Dict[str, object]:
        """
        Compute classification metrics.

        Parameters
        ----------
        predicted : np.ndarray  shape (N,)  — predicted binary labels
        actual : np.ndarray  shape (N,)  — ground-truth binary labels

        Returns
        -------
        dict with keys: precision, recall, f1, confusion_matrix, report
        """
        metrics = {
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, predicted),
            "report": classification_report(
                actual, predicted, target_names=["Normal", "Anomaly"],
                zero_division=0,
            ),
        }

        logger.info(
            "Metrics — Precision: %.4f | Recall: %.4f | F1: %.4f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        return metrics
