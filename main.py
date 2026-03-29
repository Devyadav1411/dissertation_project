"""
main.py — End-to-end pipeline for DIMM anomaly detection.

Execution order
---------------
1. Load configuration
2. Generate synthetic DIMM telemetry (normal + anomalies)
3. Prepare data  (normalise → sequence → split → DataLoaders)
4. Build Temporal Autoencoder
5. Train on normal-only data
6. Run anomaly detection on the test set
7. Evaluate and print metrics
8. Save all visualisations

Usage
-----
    python main.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

from config import Config
from data.generator import DIMMTelemetryGenerator
from data.data_loader import DIMMDataLoader
from model.model import TemporalAutoencoder
from model.train import Trainer
from detection.detect import AnomalyDetector
from visualization.visualize import Visualizer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Global reproducibility ────────────────────────────────────────────────────
def _set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    _set_global_seeds(cfg.SEED)

    # Create required output directories
    for d in (cfg.MODEL_SAVE_DIR, cfg.PLOT_SAVE_DIR, cfg.DATA_SAVE_DIR):
        os.makedirs(d, exist_ok=True)

    # ── 1. Generate synthetic data ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Generating synthetic DIMM telemetry")
    logger.info("=" * 60)

    generator = DIMMTelemetryGenerator(
        num_samples=cfg.NUM_SAMPLES, seed=cfg.SEED
    )
    df, labels = generator.generate_dataset(anomaly_ratio=cfg.ANOMALY_RATIO)

    logger.info(
        "Dataset: %d samples | anomalies: %d (%.1f%%)",
        len(df),
        labels.sum(),
        100.0 * labels.mean(),
    )

    # Save raw data for inspection
    df["label"] = labels
    df.to_csv(os.path.join(cfg.DATA_SAVE_DIR, "telemetry.csv"), index=False)
    df.drop(columns=["label"], inplace=True)
    logger.info("Raw telemetry saved to %s", cfg.DATA_SAVE_DIR)

    # ── 2. Prepare data ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Preparing data (normalise, sequence, split)")
    logger.info("=" * 60)

    scaler_path = os.path.join(cfg.DATA_SAVE_DIR, "scaler.pkl")
    loader = DIMMDataLoader(
        sequence_length=cfg.SEQUENCE_LENGTH,
        scaler_save_path=scaler_path,
    )
    train_loader, val_loader, test_loader = loader.get_data_loaders(
        df, labels, batch_size=cfg.BATCH_SIZE
    )

    logger.info(
        "Splits → train: %d | val: %d | test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # ── 3. Build model ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Building Temporal Autoencoder")
    logger.info("=" * 60)

    model = TemporalAutoencoder(
        num_features=cfg.NUM_FEATURES,
        hidden_dim=cfg.HIDDEN_DIM,
        latent_dim=cfg.LATENT_DIM,
        sequence_length=cfg.SEQUENCE_LENGTH,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", n_params)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Training on normal-only data")
    logger.info("=" * 60)

    trainer = Trainer(model, train_loader, val_loader, config=cfg)
    history = trainer.train()
    trainer.plot_training_loss(history)

    # ── 5. Detect anomalies ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Detecting anomalies on the test set")
    logger.info("=" * 60)

    detector = AnomalyDetector(
        model,
        threshold_method=cfg.THRESHOLD_METHOD,
        percentile=cfg.PERCENTILE,
        sigma=cfg.STATISTICAL_SIGMA,
    )

    # Calibrate threshold on the training set (normal data)
    train_errors = detector.compute_reconstruction_error(train_loader)
    threshold = detector.determine_threshold(train_errors)
    logger.info("Threshold: %.6f", threshold)

    # Score the test set
    test_errors, pred_labels, _ = detector.detect(test_loader)

    # Ground-truth labels for the test split
    true_labels = np.concatenate(
        [y.numpy() for _, y in test_loader]
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Evaluation")
    logger.info("=" * 60)

    metrics = detector.evaluate(pred_labels, true_labels)
    logger.info("\n%s", metrics["report"])

    # ── 7. Visualise ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7 — Generating visualisations")
    logger.info("=" * 60)

    viz = Visualizer(save_dir=cfg.PLOT_SAVE_DIR)

    # Training loss
    viz.plot_training_loss(history)

    # Confusion matrix
    viz.plot_confusion_matrix(true_labels, pred_labels)

    # Reconstruction error over time
    viz.plot_reconstruction_error(test_errors, threshold, labels=true_labels)

    # Original vs reconstructed (use test loader)
    model.eval()
    device = next(model.parameters()).device
    orig_list, recon_list = [], []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            orig_list.append(bx.cpu().numpy())
            recon_list.append(model(bx).cpu().numpy())

    original = np.concatenate(orig_list)
    reconstructed = np.concatenate(recon_list)

    viz.plot_original_vs_reconstructed(original, reconstructed, cfg.FEATURE_NAMES)

    # Feature-wise error heatmap
    with torch.no_grad():
        feat_errors_list = []
        for bx, _ in test_loader:
            bx = bx.to(device)
            recon = model(bx)
            # per-sequence, per-feature MSE
            fe = ((bx - recon) ** 2).mean(dim=1).cpu().numpy()
            feat_errors_list.append(fe)
    feat_errors = np.concatenate(feat_errors_list)
    viz.plot_feature_heatmap(feat_errors, cfg.FEATURE_NAMES)

    # Raw data with anomaly highlights (use first 2000 samples of raw df)
    feature_values = df[cfg.FEATURE_NAMES].values.astype(np.float32)
    viz.plot_anomaly_highlights(feature_values, labels, cfg.FEATURE_NAMES)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Outputs in '%s' and '%s'.",
                cfg.PLOT_SAVE_DIR, cfg.MODEL_SAVE_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
