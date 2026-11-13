"""Метрики кредитного скоринга: AUC, Gini, KS, PR-AUC, Brier, PSI."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)


def ks_stat(y, p) -> float:
    """Колмогоров-Смирнов: максимальный разрыв между кумулятивными долями bad и good."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summarize(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "ks": ks_stat(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index: <0.1 стабильно, 0.1-0.25 умеренный сдвиг, >0.25 сильный сдвиг."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def decile_table(y, p, q: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    d["decile"] = pd.qcut(np.arange(len(d)), q, labels=False) + 1
    g = d.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"), avg_pd=("p", "mean"))
    g["default_rate"] = g.defaults / g.n
    g["lift"] = g.default_rate / d.y.mean()
    g["cum_capture"] = g.defaults.cumsum() / d.y.sum()
    return g.reset_index()
