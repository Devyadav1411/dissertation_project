"""
visualization/visualize.py — All plotting utilities for the DIMM anomaly system.

Plots produced
--------------
1. original_vs_reconstructed  — per-feature overlay of input vs autoencoder output
2. reconstruction_error       — error over time with threshold line
3. anomaly_highlights         — raw signal with anomalous regions shaded red
4. feature_heatmap            — seaborn heatmap of per-feature reconstruction error
5. training_loss              — train vs validation MSE curves
6. confusion_matrix           — classification evaluation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates and saves all diagnostic plots for the anomaly detection pipeline.

    Parameters
    ----------
    save_dir : str
        Directory where all plots are written.
    """

    def __init__(self, save_dir: str = "outputs/plots") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Original vs Reconstructed
    # ──────────────────────────────────────────────────────────────────────────

    def plot_original_vs_reconstructed(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        feature_names: List[str],
        n_steps: int = 500,
    ) -> None:
        """
        Overlay original and reconstructed signals for each feature.

        Parameters
        ----------
        original : np.ndarray  shape (N, seq_len, num_features)  or  (T, num_features)
        reconstructed : np.ndarray  same shape as *original*
        feature_names : list[str]
        n_steps : int
            Number of time-steps to display (for readability).
        """
        # Flatten sequences to a 2-D time-series if needed
        orig_flat = self._flatten(original)[:n_steps]
        recon_flat = self._flatten(reconstructed)[:n_steps]

        n_feat = len(feature_names)
        fig, axes = plt.subplots(n_feat, 1, figsize=(14, 3 * n_feat), sharex=True)

        for i, (ax, name) in enumerate(zip(axes, feature_names)):
            ax.plot(orig_flat[:, i], label="Original", color="steelblue", linewidth=1.2)
            ax.plot(
                recon_flat[:, i],
                label="Reconstructed",
                color="darkorange",
                linewidth=1.2,
                linestyle="--",
            )
            ax.set_ylabel(name, fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time step")
        fig.suptitle("Original vs Reconstructed Signals", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "original_vs_reconstructed.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Reconstruction error over time
    # ──────────────────────────────────────────────────────────────────────────

    def plot_reconstruction_error(
        self,
        errors: np.ndarray,
        threshold: float,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot per-sequence reconstruction error with a horizontal threshold line.

        Parameters
        ----------
        errors : np.ndarray  shape (N,)
        threshold : float
        labels : np.ndarray | None  — ground-truth labels (for colour coding)
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        if labels is not None:
            # Map binary labels to colors (must use list of hex strings for scatter)
            color_map = ["steelblue" if lbl == 0 else "red" for lbl in labels]
            ax.scatter(
                np.arange(len(errors)),
                errors,
                c=color_map,
                s=4,
                alpha=0.7,
            )
            normal_patch = mpatches.Patch(color="steelblue", label="Normal")
            anomaly_patch = mpatches.Patch(color="red", label="Anomaly")
            ax.legend(handles=[normal_patch, anomaly_patch])
        else:
            ax.plot(errors, color="steelblue", linewidth=0.8, label="Reconstruction error")

        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.4f})")
        ax.set_xlabel("Sequence index")
        ax.set_ylabel("MSE Reconstruction Error")
        ax.set_title("Reconstruction Error over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, "reconstruction_error.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Anomaly highlights on raw signal
    # ──────────────────────────────────────────────────────────────────────────

    def plot_anomaly_highlights(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        n_steps: int = 2000,
    ) -> None:
        """
        Raw signal with anomalous regions shaded in semi-transparent red.

        Parameters
        ----------
        data : np.ndarray  shape (T, num_features)
        labels : np.ndarray  shape (T,)
        feature_names : list[str]
        n_steps : int  — display window
        """
        data = data[:n_steps]
        labels = labels[:n_steps]

        n_feat = len(feature_names)
        fig, axes = plt.subplots(n_feat, 1, figsize=(14, 3 * n_feat), sharex=True)

        for i, (ax, name) in enumerate(zip(axes, feature_names)):
            ax.plot(data[:, i], color="steelblue", linewidth=1.0, label=name)

            # Shade anomalous regions
            anomaly_mask = labels == 1
            _shade_regions(ax, anomaly_mask)

            ax.set_ylabel(name, fontsize=9)
            ax.grid(True, alpha=0.3)

        normal_patch = mpatches.Patch(color="steelblue", label="Normal")
        anomaly_patch = mpatches.Patch(color="red", alpha=0.3, label="Anomaly")
        fig.legend(handles=[normal_patch, anomaly_patch], loc="upper right")
        fig.suptitle("Raw Telemetry with Anomaly Highlights", fontsize=14, fontweight="bold")
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        self._save(fig, "anomaly_highlights.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Feature-wise error heatmap
    # ──────────────────────────────────────────────────────────────────────────

    def plot_feature_heatmap(
        self,
        errors_per_feature: np.ndarray,
        feature_names: List[str],
        n_sequences: int = 200,
    ) -> None:
        """
        Heatmap of per-feature reconstruction error for a sample of sequences.

        Parameters
        ----------
        errors_per_feature : np.ndarray  shape (N, num_features)
            Per-feature MSE for each sequence.
        feature_names : list[str]
        n_sequences : int  — max rows to display
        """
        data = errors_per_feature[:n_sequences]

        fig, ax = plt.subplots(figsize=(12, max(4, n_sequences // 20)))
        sns.heatmap(
            data.T,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=False,
            yticklabels=feature_names,
            cbar_kws={"label": "MSE"},
        )
        ax.set_xlabel("Sequence index")
        ax.set_ylabel("Feature")
        ax.set_title("Feature-wise Reconstruction Error Heatmap")
        fig.tight_layout()
        self._save(fig, "feature_heatmap.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Training loss curves (standalone — also available via Trainer)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_training_loss(self, history: Dict[str, list]) -> None:
        """
        Re-plot training / validation loss from history dict.

        Parameters
        ----------
        history : dict  — keys "train_loss" and "val_loss"
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["train_loss"], label="Training loss", linewidth=2)
        ax.plot(history["val_loss"], label="Validation loss", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, "training_loss.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Confusion matrix
    # ──────────────────────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Seaborn annotated confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray  shape (N,)
        y_pred : np.ndarray  shape (N,)
        """
        cm = sk_confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        self._save(fig, "confusion_matrix.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved → %s", path)

    @staticmethod
    def _flatten(arr: np.ndarray) -> np.ndarray:
        """
        Convert (N, seq_len, features) → (N * seq_len, features) if needed.
        If already 2-D pass through unchanged.
        """
        if arr.ndim == 3:
            N, L, F = arr.shape
            return arr.reshape(N * L, F)
        return arr


def _shade_regions(ax: plt.Axes, mask: np.ndarray) -> None:
    """
    Shade contiguous True regions of *mask* on *ax* with semi-transparent red.
    """
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            ax.axvspan(start, i, color="red", alpha=0.25, linewidth=0)
            in_region = False
    if in_region:
        ax.axvspan(start, len(mask), color="red", alpha=0.25, linewidth=0)
