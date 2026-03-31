"""
app/app.py — Streamlit dashboard for the DIMM Anomaly Detection system.

Launch
------
    streamlit run app/app.py

The dashboard walks through the full pipeline interactively:
  Generate Data → Train Model → Detect Anomalies → Inspect Results
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure the project root is on sys.path when run from the app/ directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.generator import DIMMTelemetryGenerator
from data.data_loader import DIMMDataLoader
from detection.detect import AnomalyDetector
from model.model import TemporalAutoencoder
from model.train import Trainer

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIMM Anomaly Detection",
    page_icon="🖥️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "df": None,
        "labels": None,
        "model": None,
        "history": None,
        "test_errors": None,
        "pred_labels": None,
        "true_labels": None,
        "threshold": None,
        "feat_errors": None,
        "loader": None,
        "test_loader": None,
        "train_loader": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Controls
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

st.sidebar.subheader("Data Generation")
num_samples = st.sidebar.slider(
    "Number of samples", min_value=1000, max_value=20000, value=5000, step=500
)
anomaly_ratio = st.sidebar.slider(
    "Anomaly ratio", min_value=0.01, max_value=0.20, value=0.05, step=0.01
)
anomaly_intensity = st.sidebar.slider(
    "Anomaly intensity", min_value=0.5, max_value=5.0, value=1.0, step=0.25
)
seq_length = st.sidebar.number_input(
    "Sequence length", min_value=10, max_value=200, value=50, step=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("Training")
epochs = st.sidebar.slider("Max epochs", min_value=5, max_value=100, value=20, step=5)
lr = st.sidebar.select_slider(
    "Learning rate",
    options=[1e-4, 5e-4, 1e-3, 5e-3],
    value=1e-3,
    format_func=lambda x: f"{x:.0e}",
)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)

st.sidebar.markdown("---")
gen_btn = st.sidebar.button("🔄 Generate Data", use_container_width=True)
train_btn = st.sidebar.button("🚀 Train Model", use_container_width=True)
detect_btn = st.sidebar.button("🔍 Detect Anomalies", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main area — Title
# ─────────────────────────────────────────────────────────────────────────────

st.title("🖥️ DIMM Hardware Telemetry — Anomaly Detection Dashboard")
st.markdown(
    "An end-to-end system using a **Temporal Autoencoder (LSTM)** trained only on "
    "normal data to detect anomalies in server DIMM telemetry."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate Data
# ─────────────────────────────────────────────────────────────────────────────

if gen_btn:
    with st.spinner("Generating synthetic telemetry…"):
        gen = DIMMTelemetryGenerator(num_samples=num_samples, seed=Config.SEED)
        # Use only spike + burst with configurable intensity for the demo
        df_normal = gen.generate_normal_data()
        labels = np.zeros(len(df_normal), dtype=int)

        n_anomaly = int(len(df_normal) * anomaly_ratio / 2)
        df_normal, labels = gen.inject_anomalies(
            df_normal, "spike", anomaly_ratio / 2, anomaly_intensity
        )
        df_normal, labels2 = gen.inject_anomalies(
            df_normal, "burst", anomaly_ratio / 2, anomaly_intensity
        )
        labels = np.maximum(labels, labels2)

        st.session_state.df = df_normal
        st.session_state.labels = labels

    st.success(
        f"✅ Generated **{len(df_normal):,}** samples | "
        f"anomalies: **{labels.sum():,}** ({100*labels.mean():.1f}%)"
    )

if st.session_state.df is not None:
    st.subheader("📊 Raw Telemetry Data")
    df: pd.DataFrame = st.session_state.df
    labels: np.ndarray = st.session_state.labels

    st.dataframe(df.head(200), use_container_width=True)

    feat = st.selectbox("Feature to plot", Config.FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df[feat].values, linewidth=0.8, color="steelblue")
    # Highlight anomalies
    anomaly_idx = np.where(labels == 1)[0]
    ax.scatter(anomaly_idx, df[feat].values[anomaly_idx], color="red", s=5, zorder=5)
    ax.set_title(f"{feat} — red dots are injected anomalies")
    ax.set_xlabel("Time step")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train Model
# ─────────────────────────────────────────────────────────────────────────────

if train_btn:
    if st.session_state.df is None:
        st.warning("⚠️ Generate data first!")
    else:
        # Patch config
        cfg = Config()
        cfg.SEQUENCE_LENGTH = int(seq_length)
        cfg.EPOCHS = int(epochs)
        cfg.LEARNING_RATE = float(lr)
        cfg.BATCH_SIZE = int(batch_size)

        with st.spinner("Preparing data and training the autoencoder…"):
            loader = DIMMDataLoader(sequence_length=int(seq_length))
            train_l, val_l, test_l = loader.get_data_loaders(
                st.session_state.df, st.session_state.labels, batch_size=int(batch_size)
            )

            model = TemporalAutoencoder(
                num_features=Config.NUM_FEATURES,
                hidden_dim=Config.HIDDEN_DIM,
                latent_dim=Config.LATENT_DIM,
                sequence_length=int(seq_length),
            )

            trainer = Trainer(model, train_l, val_l, config=cfg)
            history = trainer.train()

            st.session_state.model = model
            st.session_state.history = history
            st.session_state.loader = loader
            st.session_state.train_loader = train_l
            st.session_state.test_loader = test_l

        st.success("✅ Training complete!")

if st.session_state.history is not None:
    st.subheader("📈 Training Loss Curves")
    history = st.session_state.history
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Detect Anomalies
# ─────────────────────────────────────────────────────────────────────────────

if detect_btn:
    if st.session_state.model is None:
        st.warning("⚠️ Train the model first!")
    else:
        with st.spinner("Running anomaly detection…"):
            detector = AnomalyDetector(
                st.session_state.model,
                threshold_method=Config.THRESHOLD_METHOD,
            )
            # Calibrate on training errors
            train_errors = detector.compute_reconstruction_error(
                st.session_state.train_loader
            )
            threshold = detector.determine_threshold(train_errors)

            # Score test set
            test_errors, pred_labels, _ = detector.detect(
                st.session_state.test_loader
            )
            true_labels = np.concatenate(
                [y.numpy() for _, y in st.session_state.test_loader]
            )

            # Per-feature errors
            model = st.session_state.model
            model.eval()
            device = next(model.parameters()).device
            feat_errors_list = []
            with torch.no_grad():
                for bx, _ in st.session_state.test_loader:
                    bx = bx.to(device)
                    recon = model(bx)
                    fe = ((bx - recon) ** 2).mean(dim=1).cpu().numpy()
                    feat_errors_list.append(fe)
            feat_errors = np.concatenate(feat_errors_list)

            st.session_state.test_errors = test_errors
            st.session_state.pred_labels = pred_labels
            st.session_state.true_labels = true_labels
            st.session_state.threshold = threshold
            st.session_state.feat_errors = feat_errors

        st.success(
            f"✅ Detection complete — anomalies found: **{pred_labels.sum():,}** "
            f"/ {len(pred_labels):,}"
        )

if st.session_state.test_errors is not None:
    errors = st.session_state.test_errors
    pred = st.session_state.pred_labels
    true = st.session_state.true_labels
    thr = st.session_state.threshold

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.subheader("📋 Evaluation Metrics")
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(true, pred, zero_division=0)
    r = recall_score(true, pred, zero_division=0)
    f = f1_score(true, pred, zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{p:.3f}")
    col2.metric("Recall", f"{r:.3f}")
    col3.metric("F1 Score", f"{f:.3f}")

    # ── Reconstruction error plot ─────────────────────────────────────────────
    st.subheader("📉 Reconstruction Error over Time")
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = np.where(true == 1, "red", "steelblue")
    ax.scatter(np.arange(len(errors)), errors, c=colors, s=3, alpha=0.6)
    ax.axhline(thr, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={thr:.4f}")
    normal_p = mpatches.Patch(color="steelblue", label="Normal")
    anomaly_p = mpatches.Patch(color="red", label="Anomaly (true)")
    ax.legend(handles=[normal_p, anomaly_p, ax.get_lines()[0]])
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ── Feature heatmap ───────────────────────────────────────────────────────
    st.subheader("🔥 Feature-wise Error Heatmap")
    import seaborn as sns
    n_show = min(300, len(st.session_state.feat_errors))
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        st.session_state.feat_errors[:n_show].T,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=False,
        yticklabels=Config.FEATURE_NAMES,
        cbar_kws={"label": "MSE"},
    )
    ax.set_xlabel("Sequence index")
    ax.set_title("Feature-wise Reconstruction Error")
    st.pyplot(fig)
    plt.close(fig)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("🗃️ Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
    plt.close(fig)
