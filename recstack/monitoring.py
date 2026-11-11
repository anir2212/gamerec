"""Drift measures and alert rules used on simulated weeks. There is no live traffic here."""

from __future__ import annotations

import math

import numpy as np

SMOOTHING = 1e-4


def _shares(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.sum() <= 0 or (counts < 0).any():
        raise ValueError("counts must be a nonnegative, nonempty vector")
    shares = counts / counts.sum() + SMOOTHING
    return shares / shares.sum()


def psi(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    """Population stability index: sum((current - reference) * ln(current / reference)) over bins."""
    if len(reference_counts) != len(current_counts):
        raise ValueError("PSI needs the same bins on both sides")
    reference, current = _shares(reference_counts), _shares(current_counts)
    return float(np.sum((current - reference) * np.log(current / reference)))


def js_divergence_bits(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical shares, at most 1."""
    if len(reference_counts) != len(current_counts):
        raise ValueError("JS divergence needs the same bins on both sides")
    reference, current = _shares(reference_counts), _shares(current_counts)
    middle = (reference + current) / 2
    return float(0.5 * np.sum(reference * np.log2(reference / middle)) + 0.5 * np.sum(current * np.log2(current / middle)))


def quantile_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Interior bin edges from reference quantiles; duplicate edges collapse."""
    if len(reference) == 0:
        raise ValueError("reference scores are empty")
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per bin; values outside the reference range fall into the end bins."""
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = quantile_edges(reference, bins)
    return psi(binned(reference, edges), binned(current, edges))


def drift_alert(drift_values: list[float], control_values: list[float]) -> list[bool]:
    """A drift week alerts only above the highest control week. Higher means more drift."""
    limit = max(control_values)
    return [value > limit for value in drift_values]


def recall_alert(drift_recall: list[float], control_recall: list[float], margin: float = 0.05) -> list[bool]:
    """A week alerts when recall falls more than ``margin`` below the control's all-week mean."""
    limit = sum(control_recall) / len(control_recall) - margin
    return [value < limit for value in drift_recall]


def first_alert_week(alerts: list[bool]) -> int | None:
    """One-based first alerting week, or None."""
    return next((index + 1 for index, flag in enumerate(alerts) if flag), None)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)
