"""Accuracy, macro-F1, bootstrap CIs, calibration and latency summaries."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(gold: list[str], pred: list[str | None]) -> float:
    return float(np.mean([g == p for g, p in zip(gold, pred)]))


def macro_f1(gold: list[str], pred: list[str | None]) -> float:
    labels = sorted(set(gold))
    return float(f1_score(gold, [p if p is not None else "<invalid>" for p in pred], labels=labels, average="macro"))


def bootstrap_ci(gold, pred, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    hit = np.array([g == p for g, p in zip(gold, pred)], dtype=float)
    idx = rng.integers(0, len(hit), size=(n, len(hit)))
    accs = hit[idx].mean(1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def paired_bootstrap_p(gold, pred_a, pred_b, n: int = 10000, seed: int = 0) -> float:
    """One-sided p-value that A is not better than B (share of resamples where acc_A <= acc_B)."""
    rng = np.random.default_rng(seed)
    a = np.array([g == p for g, p in zip(gold, pred_a)], dtype=float)
    b = np.array([g == p for g, p in zip(gold, pred_b)], dtype=float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    return float(np.mean(a[idx].mean(1) <= b[idx].mean(1)))


def ece(gold, pred, conf, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    hit = np.array([g == p for g, p in zip(gold, pred)], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def selective_accuracy(gold, pred, conf, coverage: float = 0.8) -> float:
    """Accuracy on the `coverage` most confident predictions (the rest routed to a human)."""
    conf = np.asarray(conf, dtype=float)
    k = max(1, int(round(len(conf) * coverage)))
    keep = np.argsort(-conf, kind="stable")[:k]
    return float(np.mean([gold[i] == pred[i] for i in keep]))


def latency_summary(lat_s: list[float]) -> dict[str, float]:
    a = np.asarray(lat_s) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


def summarise(gold, pred, conf=None) -> dict:
    lo, hi = bootstrap_ci(gold, pred)
    out = {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci95": [lo, hi],
        "macro_f1": macro_f1(gold, pred),
        "invalid_rate": float(np.mean([p is None for p in pred])),
    }
    if conf is not None:
        out["ece"] = ece(gold, pred, conf)
        out["acc_at_80pct_coverage"] = selective_accuracy(gold, pred, conf, 0.8)
    return out
