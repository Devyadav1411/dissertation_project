# DIMM Anomaly Detection using a Temporal Autoencoder

> **Dissertation Project** — End-to-end anomaly detection system for server DIMM hardware telemetry using an LSTM-based Temporal Autoencoder.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIMM Telemetry Pipeline                       │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Generator  │──▶│  DataLoader  │──▶│  Temporal Autoencoder│ │
│  │  (6 feats)  │   │  (normalize  │   │  Encoder (LSTM×2)    │ │
│  │  4 anomaly  │   │   sequence   │   │     ↓ latent z       │ │
│  │    types)   │   │   split)     │   │  Decoder (LSTM×2)    │ │
│  └─────────────┘   └──────────────┘   └──────────┬───────────┘ │
│                                                   │ reconstruct  │
│  ┌─────────────┐   ┌──────────────┐              │             │
│  │ Visualizer  │◀──│  Detector    │◀─────────────┘             │
│  │  6 plots    │   │ (MSE thresh) │                             │
│  └─────────────┘   └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
dissertation_project/
├── data/
│   ├── __init__.py
│   ├── generator.py        ← Synthetic DIMM telemetry generator
│   └── data_loader.py      ← Normalisation, sequencing, DataLoaders
│
├── model/
│   ├── __init__.py
│   ├── model.py            ← LSTM Temporal Autoencoder (PyTorch)
│   └── train.py            ← Training loop with early stopping
│
├── detection/
│   ├── __init__.py
│   └── detect.py           ← Reconstruction-error-based anomaly detector
│
├── visualization/
│   ├── __init__.py
│   └── visualize.py        ← 6 diagnostic plots (matplotlib / seaborn)
│
├── app/
│   └── app.py              ← Interactive Streamlit dashboard
│
├── config.py               ← Centralised configuration
├── main.py                 ← End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Devyadav1411/dissertation_project.git
cd dissertation_project

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This will:
1. Generate 10,000 synthetic DIMM telemetry samples with injected anomalies
2. Normalise, sequence and split the data
3. Train the Temporal Autoencoder on normal-only data
4. Detect anomalies on the test set
5. Print precision / recall / F1 metrics
6. Save all plots to `outputs/plots/`

### Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_SAMPLES` | 10 000 | Total time-steps to simulate |
| `ANOMALY_RATIO` | 0.05 | Fraction of anomalous data |
| `SEQUENCE_LENGTH` | 50 | Sliding-window length |
| `BATCH_SIZE` | 64 | Training batch size |
| `HIDDEN_DIM` | 128 | First LSTM hidden size |
| `LATENT_DIM` | 64 | Bottleneck dimension |
| `EPOCHS` | 50 | Max training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `EARLY_STOPPING_PATIENCE` | 10 | Epochs without improvement before stopping |
| `THRESHOLD_METHOD` | `statistical` | `statistical` (mean+3σ) or `percentile` |

---

## Features

### Synthetic Data Generator (`data/generator.py`)
- Six telemetry channels with realistic physical relationships
- Sinusoidal daily patterns with Gaussian noise
- Four anomaly types: **spike**, **drift**, **burst**, **sensor_failure**

### Temporal Autoencoder (`model/model.py`)
- **Encoder**: LSTM(128) → LSTM(64) → latent vector *z*
- **Decoder**: RepeatVector → LSTM(64) → LSTM(128) → TimeDistributed Linear

### Anomaly Detection (`detection/detect.py`)
- Per-sequence MSE reconstruction error
- Statistical threshold: mean + 3·std
- Percentile threshold: configurable (default 95th)

### Visualisations (`visualization/visualize.py`)
1. Original vs Reconstructed signals
2. Reconstruction error over time
3. Raw signal with anomaly highlights
4. Feature-wise error heatmap
5. Training / validation loss curves
6. Confusion matrix

---

## Sample Results

After running `main.py` you will find in `outputs/plots/`:

- `training_loss.png` — convergence curves
- `reconstruction_error.png` — per-sequence MSE with threshold line
- `original_vs_reconstructed.png` — all six features overlaid
- `anomaly_highlights.png` — shaded anomalous regions
- `feature_heatmap.png` — seaborn heatmap
- `confusion_matrix.png` — precision / recall breakdown

---

## Requirements

- Python ≥ 3.9
- torch ≥ 2.0.0
- numpy ≥ 1.24.0
- pandas ≥ 2.0.0
- scikit-learn ≥ 1.3.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0
- streamlit ≥ 1.28.0
- plotly ≥ 5.15.0

---

## License

MIT License — see [LICENSE](LICENSE) for details.
