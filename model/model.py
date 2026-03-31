"""
model/model.py — Temporal Autoencoder based on stacked LSTM layers.

Architecture
------------
Encoder
  LSTM₁  input_size=num_features  → hidden_size=hidden_dim (128)
  LSTM₂  input_size=hidden_dim    → hidden_size=latent_dim  (64)
  Output: last hidden state of LSTM₂  →  z  shape (batch, latent_dim)

Decoder
  RepeatVector  z repeated sequence_length times  → (batch, seq_len, latent_dim)
  LSTM₁  input_size=latent_dim  → hidden_size=latent_dim  (64)
  LSTM₂  input_size=latent_dim  → hidden_size=hidden_dim  (128)
  TimeDistributed Linear  128 → num_features  (applied per time-step)

Loss: Mean Squared Error (reconstruction loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class TemporalAutoencoder(nn.Module):
    """
    LSTM-based Temporal Autoencoder for unsupervised anomaly detection.

    Parameters
    ----------
    num_features : int
        Number of input/output channels (e.g. 6 for DIMM telemetry).
    hidden_dim : int
        Hidden size of the first LSTM layer in both encoder and decoder.
    latent_dim : int
        Bottleneck size — dimension of the compressed latent representation.
    sequence_length : int
        Length of the input / output sequences.
    """

    def __init__(
        self,
        num_features: int = Config.NUM_FEATURES,
        hidden_dim: int = Config.HIDDEN_DIM,
        latent_dim: int = Config.LATENT_DIM,
        sequence_length: int = Config.SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.enc_lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dec_lstm2 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # TimeDistributed Dense: applied independently to each time-step
        self.output_layer = nn.Linear(hidden_dim, num_features)

    # ──────────────────────────────────────────────────────────────────────────

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress a sequence to a latent vector.

        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, num_features)

        Returns
        -------
        z : Tensor  shape (batch, latent_dim)
        """
        out1, _ = self.enc_lstm1(x)          # (batch, seq_len, hidden_dim)
        _, (h_n, _) = self.enc_lstm2(out1)   # h_n: (1, batch, latent_dim)
        z = h_n.squeeze(0)                   # (batch, latent_dim)
        return z

    # ──────────────────────────────────────────────────────────────────────────

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original sequence from a latent vector.

        Parameters
        ----------
        z : Tensor  shape (batch, latent_dim)

        Returns
        -------
        reconstruction : Tensor  shape (batch, seq_len, num_features)
        """
        # RepeatVector — broadcast latent vector across the time dimension
        z_rep = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # (batch, seq_len, latent_dim)

        out1, _ = self.dec_lstm1(z_rep)   # (batch, seq_len, latent_dim)
        out2, _ = self.dec_lstm2(out1)    # (batch, seq_len, hidden_dim)

        # Apply linear layer to every time-step (TimeDistributed)
        reconstruction = self.output_layer(out2)  # (batch, seq_len, num_features)
        return reconstruction

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode → decode pass.

        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, num_features)

        Returns
        -------
        Tensor  shape (batch, seq_len, num_features)
        """
        z = self.encoder(x)
        return self.decoder(z)
