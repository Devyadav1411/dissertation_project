"""
data/generator.py — Synthetic DIMM telemetry generator.

Generates realistic time-series data for six hardware telemetry features:
  temperature, voltage, current, power, memory_usage, ecc_errors

Feature relationships mirror real DIMM physics:
  memory_usage → power → temperature → ecc_errors
  voltage and current evolve independently with their own periodic patterns.

Anomaly types supported
-----------------------
spike        : sudden jump in temperature / power at random indices
drift        : gradual linear increase over a contiguous time window
burst        : ecc_error count spikes (high-lambda Poisson events)
sensor_failure: constant flat-line value for a feature over a window
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import pandas as pd


class DIMMTelemetryGenerator:
    """
    Simulates realistic DIMM hardware telemetry time-series data.

    Parameters
    ----------
    num_samples : int
        Number of time-steps to generate.
    seed : int
        Random seed for full reproducibility.
    """

    # ── Physical baseline constants ──────────────────────────────────────────
    _TEMP_BASE = 45.0        # °C  — idle temperature
    _VOLT_BASE = 1.2         # V   — DDR4 nominal voltage
    _CURR_BASE = 1.5         # A   — nominal current draw
    _PWR_BASE = 2.0          # W   — idle power
    _MEM_BASE = 50.0         # %   — average memory utilisation
    _ECC_BASE_LAMBDA = 0.1   # Poisson λ at nominal temperature

    def __init__(self, num_samples: int = 10_000, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate_normal_data(self) -> pd.DataFrame:
        """
        Generate a DataFrame of normal (anomaly-free) DIMM telemetry.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, temperature, voltage, current,
                     power, memory_usage, ecc_errors
        """
        t = np.arange(self.num_samples, dtype=float)

        # ── memory_usage: daily sinusoidal pattern + noise ────────────────────
        memory_usage = (
            self._MEM_BASE
            + 15.0 * np.sin(2 * np.pi * t / 1000)          # primary daily cycle
            + 5.0 * np.sin(2 * np.pi * t / 200)            # shorter sub-cycle
            + self._rng.normal(0, 1.5, self.num_samples)    # Gaussian noise
        )
        memory_usage = np.clip(memory_usage, 0, 100)

        # ── power: linear function of memory_usage with noise ─────────────────
        power = (
            self._PWR_BASE
            + 0.06 * memory_usage
            + self._rng.normal(0, 0.15, self.num_samples)
        )
        power = np.clip(power, 0, None)

        # ── temperature: linear function of power with thermal lag ────────────
        # Simulate thermal inertia via an exponential moving average
        alpha = 0.05  # smoothing factor — larger = faster thermal response
        temperature = np.zeros(self.num_samples)
        temperature[0] = self._TEMP_BASE + 2.0 * power[0]
        for i in range(1, self.num_samples):
            t_target = self._TEMP_BASE + 2.0 * power[i]
            temperature[i] = alpha * t_target + (1 - alpha) * temperature[i - 1]
        temperature += self._rng.normal(0, 0.5, self.num_samples)
        temperature = np.clip(temperature, 20, 100)

        # ── ecc_errors: Poisson distribution scaled by temperature ────────────
        lambda_ecc = self._ECC_BASE_LAMBDA * np.exp(
            0.015 * (temperature - self._TEMP_BASE)
        )
        ecc_errors = self._rng.poisson(lambda_ecc).astype(float)

        # ── voltage: sinusoidal around nominal with very small ripple ─────────
        voltage = (
            self._VOLT_BASE
            + 0.02 * np.sin(2 * np.pi * t / 500)
            + self._rng.normal(0, 0.005, self.num_samples)
        )
        voltage = np.clip(voltage, 1.0, 1.4)

        # ── current: correlated with power / voltage, own periodicity ─────────
        current = (
            power / voltage
            + 0.1 * np.sin(2 * np.pi * t / 300)
            + self._rng.normal(0, 0.05, self.num_samples)
        )
        current = np.clip(current, 0, None)

        timestamps = pd.date_range("2024-01-01", periods=self.num_samples, freq="1min")

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature": np.round(temperature, 3),
                "voltage": np.round(voltage, 4),
                "current": np.round(current, 4),
                "power": np.round(power, 4),
                "memory_usage": np.round(memory_usage, 3),
                "ecc_errors": ecc_errors,
            }
        )
        return df

    # ──────────────────────────────────────────────────────────────────────────

    def inject_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_type: str = "spike",
        anomaly_ratio: float = 0.05,
        intensity: float = 1.0,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject anomalies into *data* and return the modified DataFrame plus labels.

        Parameters
        ----------
        data : pd.DataFrame
            Normal telemetry DataFrame (output of generate_normal_data).
        anomaly_type : str
            One of: "spike", "drift", "burst", "sensor_failure".
        anomaly_ratio : float
            Fraction of total samples affected (approximate).
        intensity : float
            Scales the magnitude of injected anomalies (default 1.0 = realistic).

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Modified DataFrame and integer label array (0 = normal, 1 = anomaly).
        """
        df = data.copy()
        labels = np.zeros(len(df), dtype=int)
        n_anomaly = max(1, int(len(df) * anomaly_ratio))

        if anomaly_type == "spike":
            labels = self._inject_spike(df, labels, n_anomaly, intensity)
        elif anomaly_type == "drift":
            labels = self._inject_drift(df, labels, n_anomaly, intensity)
        elif anomaly_type == "burst":
            labels = self._inject_burst(df, labels, n_anomaly, intensity)
        elif anomaly_type == "sensor_failure":
            labels = self._inject_sensor_failure(df, labels, n_anomaly)
        else:
            raise ValueError(
                f"Unknown anomaly_type '{anomaly_type}'. "
                "Choose from: spike, drift, burst, sensor_failure."
            )

        return df, labels

    # ──────────────────────────────────────────────────────────────────────────

    def generate_dataset(
        self, anomaly_ratio: float = 0.05
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Full pipeline: generate normal data then inject a mix of all anomaly types.

        The anomaly budget is split equally across the four anomaly types.

        Parameters
        ----------
        anomaly_ratio : float
            Overall fraction of data that should be labelled as anomalous.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Final DataFrame and integer label array.
        """
        df = self.generate_normal_data()
        labels = np.zeros(len(df), dtype=int)

        per_type_ratio = anomaly_ratio / 4.0

        for atype in ("spike", "drift", "burst", "sensor_failure"):
            df, new_labels = self.inject_anomalies(
                df, anomaly_type=atype, anomaly_ratio=per_type_ratio
            )
            labels = np.maximum(labels, new_labels)

        return df, labels

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _inject_spike(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_anomaly: int,
        intensity: float,
    ) -> np.ndarray:
        """Sudden large jumps in temperature and power."""
        indices = self._rng.choice(len(df), size=n_anomaly, replace=False)
        for idx in indices:
            df.at[idx, "temperature"] += intensity * self._rng.uniform(15, 35)
            df.at[idx, "power"] += intensity * self._rng.uniform(2, 6)
        labels[indices] = 1
        return labels

    def _inject_drift(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_anomaly: int,
        intensity: float,
    ) -> np.ndarray:
        """Gradual linear increase in temperature over a contiguous window."""
        start = self._rng.integers(0, max(1, len(df) - n_anomaly))
        end = int(start) + n_anomaly
        window = np.arange(n_anomaly)
        drift = intensity * (window / n_anomaly) * 25.0  # up to 25 °C extra
        df.loc[start:end - 1, "temperature"] = (
            df.loc[start:end - 1, "temperature"].values + drift
        )
        df.loc[start:end - 1, "power"] = (
            df.loc[start:end - 1, "power"].values
            + intensity * (window / n_anomaly) * 3.0
        )
        labels[start:end] = 1
        return labels

    def _inject_burst(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_anomaly: int,
        intensity: float,
    ) -> np.ndarray:
        """ECC error spikes — high-lambda Poisson events."""
        indices = self._rng.choice(len(df), size=n_anomaly, replace=False)
        burst_lambda = intensity * 20.0
        df.loc[indices, "ecc_errors"] = self._rng.poisson(
            burst_lambda, size=n_anomaly
        ).astype(float)
        labels[indices] = 1
        return labels

    def _inject_sensor_failure(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_anomaly: int,
    ) -> np.ndarray:
        """Flat-line (stuck sensor) value for a randomly chosen feature."""
        feature = random.choice(["temperature", "voltage", "current", "power"])
        start = self._rng.integers(0, max(1, len(df) - n_anomaly))
        end = int(start) + n_anomaly
        flat_value = float(df[feature].iloc[int(start)])
        df.loc[start:end - 1, feature] = flat_value
        labels[start:end] = 1
        return labels
