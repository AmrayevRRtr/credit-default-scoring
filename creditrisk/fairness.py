"""Аудит справедливости решений модели по группам (пол, возраст). Защищённые признаки в модель не подаются."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditrisk.data import age_group


def audit(df: pd.DataFrame, p, approve) -> pd.DataFrame:
    """df: тестовая выборка (sex, age, default). Возвращает метрики по группам."""
    d = df[["sex", "age", "default"]].copy()
    d["p"], d["approve"] = np.asarray(p), np.asarray(approve, bool)
    d["sex_g"] = d.sex.map({1: "male", 2: "female"})
    d["age_g"] = age_group(d.age)
    rows = []
    for attr in ("sex_g", "age_g"):
        for g, s in d.groupby(attr):
            good = s[s.default == 0]
            rows.append({"attribute": attr.replace("_g", ""), "group": g, "n": len(s),
                         "approval_rate": s.approve.mean(),
                         "approval_rate_among_good": good.approve.mean(),   # equal opportunity
                         "default_rate_approved": s[s.approve].default.mean(),
                         "mean_pd": s.p.mean(), "actual_default_rate": s.default.mean(),
                         "auc": roc_auc_score(s.default, s.p) if s.default.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def disparate_impact(table: pd.DataFrame) -> dict:
    """min/max approval rate среди групп атрибута; правило "четырёх пятых": значение >= 0.8 считается приемлемым."""
    return {a: float(t.approval_rate.min() / t.approval_rate.max()) for a, t in table.groupby("attribute")}
