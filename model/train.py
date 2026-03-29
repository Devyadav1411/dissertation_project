"""
model/train.py — Training loop for the Temporal Autoencoder.

Key design decisions
--------------------
* Only normal sequences (label == 0) are seen during training.
* Adam optimiser with a configurable learning rate.
* Early stopping monitors validation loss and saves the best checkpoint.
* Training / validation loss history is returned for downstream plotting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the training loop for a TemporalAutoencoder.

    Parameters
    ----------
    model : nn.Module
        The autoencoder to train.
    train_loader : DataLoader
        Batches of **normal-only** sequences.
    val_loader : DataLoader
        Batches of mixed sequences (for unbiased validation loss).
    config : Config
        Centralised configuration object.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config = Config,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Early stopping state
        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0

        # Create output directories
        Path(config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns
        -------
        dict with keys "train_loss" and "val_loss" (lists of floats per epoch).
        """
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "best_model.pt")

        logger.info("Training on device: %s", self.device)

        for epoch in range(1, self.config.EPOCHS + 1):
            # ── Training phase ────────────────────────────────────────────────
            self.model.train()
            train_loss = self._run_epoch(self.train_loader, training=True)

            # ── Validation phase ──────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(self.val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch, self.config.EPOCHS, train_loss, val_loss,
            )

            # ── Early stopping ────────────────────────────────────────────────
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self.save_model(best_model_path)
                logger.info("  ✓ New best model saved (val_loss=%.6f)", val_loss)
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        self.config.EARLY_STOPPING_PATIENCE,
                    )
                    break

        # Restore the best weights found during training
        if Path(best_model_path).exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            logger.info("Best model reloaded from %s", best_model_path)

        return history

    # ──────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Persist model state dict to *path*."""
        torch.save(self.model.state_dict(), path)
        logger.debug("Model saved → %s", path)

    # ──────────────────────────────────────────────────────────────────────────

    def plot_training_loss(self, history: Dict[str, list]) -> None:
        """
        Save a train/validation loss curve to the configured plots directory.

        Parameters
        ----------
        history : dict
            Output of :meth:`train`.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["train_loss"], label="Training loss", linewidth=2)
        ax.plot(history["val_loss"], label="Validation loss", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Temporal Autoencoder — Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(self.config.PLOT_SAVE_DIR, "training_loss.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Training loss plot saved → %s", save_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        """Execute one full pass over *loader* and return mean batch loss."""
        total_loss = 0.0
        n_batches = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            output = self.model(batch_x)
            loss = self.criterion(output, batch_x)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
