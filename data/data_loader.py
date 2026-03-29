"""
data/data_loader.py — Data preparation for the Temporal Autoencoder.

Steps
-----
1. Normalise every feature to [0, 1] with MinMaxScaler
   (scaler is saved so we can inverse-transform later).
2. Build fixed-length sliding-window sequences:
      shape: (num_sequences, sequence_length, num_features)
3. Split into train / validation / test sets.
   Training set contains ONLY normal data (label == 0).
4. Wrap each split in a PyTorch DataLoader.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import Config


class DIMMDataLoader:
    """
    Prepares raw DIMM telemetry DataFrames for model consumption.

    Parameters
    ----------
    sequence_length : int
        Number of consecutive time-steps in each input window.
    scaler_save_path : str | None
        If provided, the fitted MinMaxScaler is persisted to this path.
    """

    def __init__(
        self,
        sequence_length: int = Config.SEQUENCE_LENGTH,
        scaler_save_path: str | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.scaler_save_path = scaler_save_path
        self.scaler = MinMaxScaler()
        self._is_fitted = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────────────────────

    def normalize(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit (if not yet fitted) and transform the feature columns.

        Drops the 'timestamp' column if present.

        Parameters
        ----------
        data : pd.DataFrame
            Raw telemetry DataFrame.

        Returns
        -------
        np.ndarray  shape (num_samples, num_features)
        """
        feature_cols = [c for c in data.columns if c != "timestamp"]
        X = data[feature_cols].values.astype(np.float32)

        if not self._is_fitted:
            self.scaler.fit(X)
            self._is_fitted = True
            if self.scaler_save_path is not None:
                Path(self.scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.scaler_save_path, "wb") as f:
                    pickle.dump(self.scaler, f)
        return self.scaler.transform(X)

    # ──────────────────────────────────────────────────────────────────────────

    def create_sequences(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build overlapping fixed-length sequences with a sliding window.

        Parameters
        ----------
        data : np.ndarray  shape (num_samples, num_features)
            Normalised feature matrix.
        labels : np.ndarray  shape (num_samples,)
            Per-sample anomaly labels (0 / 1).

        Returns
        -------
        sequences : np.ndarray  shape (N, sequence_length, num_features)
        seq_labels : np.ndarray  shape (N,)
            A sequence is labelled 1 if *any* sample in the window is anomalous.
        """
        L = self.sequence_length
        N = len(data) - L + 1

        sequences = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(L, data.shape[1])
        ).reshape(N, L, data.shape[1])

        # A window is anomalous if it contains at least one anomalous step
        seq_labels = np.array(
            [int(labels[i : i + L].max()) for i in range(N)],
            dtype=int,
        )
        return sequences.astype(np.float32), seq_labels

    # ──────────────────────────────────────────────────────────────────────────

    def get_data_loaders(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        batch_size: int = Config.BATCH_SIZE,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Full pipeline: normalise → sequence → split → wrap in DataLoaders.

        The **training** DataLoader contains **only normal** sequences.
        Validation and test loaders include both normal and anomalous sequences
        (needed for threshold calibration and evaluation).

        Parameters
        ----------
        data : pd.DataFrame
        labels : np.ndarray  shape (num_samples,)
        batch_size : int

        Returns
        -------
        train_loader, val_loader, test_loader
        """
        # 1. Normalise
        X = self.normalize(data)

        # 2. Build sequences
        sequences, seq_labels = self.create_sequences(X, labels)

        # 3. Split indices deterministically
        n = len(sequences)
        n_train = int(n * Config.TRAIN_RATIO)
        n_val = int(n * Config.VAL_RATIO)

        train_seq = sequences[:n_train]
        train_lbl = seq_labels[:n_train]
        val_seq = sequences[n_train : n_train + n_val]
        val_lbl = seq_labels[n_train : n_train + n_val]
        test_seq = sequences[n_train + n_val :]
        test_lbl = seq_labels[n_train + n_val :]

        # 4. Keep only normal sequences in the training set
        normal_mask = train_lbl == 0
        train_seq = train_seq[normal_mask]
        train_lbl = train_lbl[normal_mask]

        # 5. Wrap in DataLoaders
        train_loader = self._make_loader(train_seq, train_lbl, batch_size, shuffle=True)
        val_loader = self._make_loader(val_seq, val_lbl, batch_size, shuffle=False)
        test_loader = self._make_loader(test_seq, test_lbl, batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_loader(
        sequences: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        X_tensor = torch.tensor(sequences, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
