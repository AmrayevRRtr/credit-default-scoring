"""Экономика кредитного решения. ВСЕ параметры - допущения, а не данные банка.

Клиент одобрен:  good -> доход  r * E,   bad -> потеря  LGD * E   (E - ожидаемая задолженность)
Ожидаемая прибыль одобрения: (1 - PD) * r * E - PD * LGD * E  >  0   <=>   PD < r / (r + LGD)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LendingAssumptions:
    interest_rate: float = 0.20   # r: доход за период на единицу задолженности (проценты, комиссии)
    lgd: float = 0.60             # потеря при дефолте на единицу задолженности (loss given default)

    @property
    def breakeven_pd(self) -> float:
        return self.interest_rate / (self.interest_rate + self.lgd)


def realized_profit(y, exposure, approve, a: LendingAssumptions) -> float:
    """Фактическая прибыль по реальным исходам для одобренных клиентов."""
    y, e, m = np.asarray(y), np.asarray(exposure), np.asarray(approve, bool)
    return float(((1 - y[m]) * a.interest_rate * e[m] - y[m] * a.lgd * e[m]).sum())


def strategy_table(y, exposure, strategies: dict, a: LendingAssumptions) -> pd.DataFrame:
    y = np.asarray(y)
    rows = []
    for name, approve in strategies.items():
        m = np.asarray(approve, bool)
        rows.append({"strategy": name, "approval_rate": m.mean(),
                     "default_rate_approved": float(y[m].mean()) if m.any() else np.nan,
                     "profit": realized_profit(y, exposure, m, a)})
    return pd.DataFrame(rows)


def threshold_curve(y, p, exposure, a: LendingAssumptions, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.02, 0.95, 94) if grid is None else grid
    return pd.DataFrame({"threshold": grid,
                         "profit": [realized_profit(y, exposure, np.asarray(p) < t, a) for t in grid],
                         "approval_rate": [float((np.asarray(p) < t).mean()) for t in grid]})


def empirical_threshold(y, p, exposure, a: LendingAssumptions) -> float:
    c = threshold_curve(y, p, exposure, a)
    return float(c.loc[c.profit.idxmax(), "threshold"])
