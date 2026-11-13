"""Скоринг новых клиентов: PD, баллы скоркарты, решение и причины отказа."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd

from creditrisk.business import LendingAssumptions
from creditrisk.data import ID, ROOT, load_features


def load_artifacts():
    try:
        return joblib.load(ROOT / "models" / "pd_model.joblib"), joblib.load(ROOT / "models" / "scorecard.joblib")
    except Exception as e:   # модели, сохранённые другой версией sklearn/lightgbm, могут не открыться
        raise RuntimeError(
            "Не удалось загрузить модели из models/ (вероятно, другая версия scikit-learn/lightgbm). "
            "Переобучите их под своё окружение: python -m creditrisk.pipeline (около 6 минут)."
        ) from e


def score_clients(raw: pd.DataFrame, assumptions: LendingAssumptions | None = None) -> pd.DataFrame:
    """raw - таблица в формате исходного датасета (колонка default необязательна)."""
    a = assumptions or LendingAssumptions()
    pdm, sc = load_artifacts()
    df = load_features(raw)
    X = df[pdm["features"]]
    out = df[[ID, "limit_bal"]].copy()
    out["pd"] = pdm["model"].predict_proba(X)[:, 1]
    out["score"] = sc["scorecard"].score(X).values
    out["decision"] = np.where(out["pd"] < a.breakeven_pd, "approve", "decline")
    reasons = sc["scorecard"].reason_codes(X)
    out["decline_reasons"] = [r if d == "decline" else "" for r, d in zip(reasons, out["decision"])]
    return out
