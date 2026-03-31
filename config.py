"""
config.py — Centralized configuration for the DIMM Anomaly Detection system.

All hyperparameters, paths, and settings live here so that every module
imports from a single source of truth.
"""


class Config:
    # ── Data generation ──────────────────────────────────────────────────────
    NUM_SAMPLES: int = 10_000      # Total number of time-steps to simulate
    ANOMALY_RATIO: float = 0.05    # Fraction of data that contains anomalies
    SEED: int = 42                 # Global random seed for reproducibility

    # ── Feature names ────────────────────────────────────────────────────────
    FEATURE_NAMES = [
        "temperature",
        "voltage",
        "current",
        "power",
        "memory_usage",
        "ecc_errors",
    ]
    NUM_FEATURES: int = 6

    # ── Data loading / sequencing ─────────────────────────────────────────────
    SEQUENCE_LENGTH: int = 50      # Sliding-window length (time steps per sample)
    BATCH_SIZE: int = 64

    TRAIN_RATIO: float = 0.70      # Share of sequences used for training
    VAL_RATIO: float = 0.15        # Share used for validation
    TEST_RATIO: float = 0.15       # Share used for testing

    # ── Model architecture ────────────────────────────────────────────────────
    HIDDEN_DIM: int = 128          # First LSTM hidden size
    LATENT_DIM: int = 64           # Bottleneck / latent representation size

    # ── Training ─────────────────────────────────────────────────────────────
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3
    EARLY_STOPPING_PATIENCE: int = 10

    # ── Anomaly detection ─────────────────────────────────────────────────────
    THRESHOLD_METHOD: str = "statistical"   # "statistical" or "percentile"
    PERCENTILE: int = 95
    STATISTICAL_SIGMA: int = 3

    # ── Output paths ─────────────────────────────────────────────────────────
    MODEL_SAVE_DIR: str = "saved_models"
    PLOT_SAVE_DIR: str = "outputs/plots"
    DATA_SAVE_DIR: str = "outputs/data"
