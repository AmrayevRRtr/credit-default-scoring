"""Загрузка, очистка данных и feature engineering на SQL (DuckDB)."""
from __future__ import annotations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "credit_default.csv"

TARGET = "default"
ID = "client_id"
# Признаки, которые нельзя использовать в реальной кредитной модели (дискриминация).
# Они остаются в данных только для аудита справедливости.
PROTECTED = ["sex", "marriage"]


def load_clean(source=RAW_PATH) -> pd.DataFrame:
    """Читает сырой CSV (или DataFrame в том же формате), приводит названия и коды к понятному виду."""
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    ren = {"LIMIT_BAL": "limit_bal", "SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marriage", "AGE": "age"}
    for k in range(1, 7):
        ren[f"PAY_{k}"] = f"pay_{k}"
        ren[f"BILL_AMT{k}"] = f"bill_{k}"
        ren[f"PAY_AMT{k}"] = f"paid_{k}"
    df = df.rename(columns=ren)
    if TARGET in df.columns:
        df = df.drop_duplicates().reset_index(drop=True)   # 24 точных дубля в исходных данных
    # недокументированные коды -> "прочее"
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    df.insert(0, ID, np.arange(len(df)))
    return df


def _sum(expr: str, cols: list[str]) -> str:
    return " + ".join(expr.format(c=c) for c in cols)


_PAY = [f"pay_{k}" for k in range(1, 7)]
_BILL = [f"bill_{k}" for k in range(1, 7)]
_PAID = [f"paid_{k}" for k in range(1, 7)]

# paid_k - платёж в месяце k, он погашает счёт предыдущего месяца bill_{k+1}
_RATIOS = ",\n        ".join(
    f"CASE WHEN bill_{k+1} > 0 THEN LEAST(paid_{k} / bill_{k+1}, 2.0) END AS r{k}" for k in range(1, 6))

FEATURE_SQL = f"""
WITH base AS (
    SELECT *,
        {_RATIOS}
    FROM clients
)
SELECT
    {ID}, "{TARGET}", sex, marriage, education, age, limit_bal,
    {", ".join(_PAY)},
    GREATEST({", ".join(_PAY)}) AS max_delay,
    {_sum("({c} >= 1)::INTEGER", _PAY)} AS n_delay_months,
    {_sum("({c} = -1)::INTEGER", _PAY)} AS n_paid_duly,
    {_sum("({c} = -2)::INTEGER", _PAY)} AS n_no_use,
    (pay_1 + pay_2 + pay_3 - pay_4 - pay_5 - pay_6) / 3.0 AS delay_trend,
    bill_1 / limit_bal AS util_1,
    ({_sum("{c}", _BILL)}) / 6.0 / limit_bal AS util_avg,
    GREATEST({", ".join(_BILL)}) / limit_bal AS util_max,
    (bill_1 - bill_6) / limit_bal AS bill_change,
    {_sum("({c} > limit_bal)::INTEGER", _BILL)} AS n_overlimit,
    ({_sum("{c}", _PAID)}) / 6.0 / limit_bal AS paid_to_limit,
    paid_1 / limit_bal AS paid_1_to_limit,
    COALESCE((COALESCE(r1,0)+COALESCE(r2,0)+COALESCE(r3,0)+COALESCE(r4,0)+COALESCE(r5,0))
        / NULLIF((r1 IS NOT NULL)::INTEGER+(r2 IS NOT NULL)::INTEGER+(r3 IS NOT NULL)::INTEGER
                 +(r4 IS NOT NULL)::INTEGER+(r5 IS NOT NULL)::INTEGER, 0), 1.0) AS pay_ratio_avg,
    -- допущение: ожидаемая задолженность на момент решения ~ средний счёт за 6 месяцев (не больше лимита)
    LEAST(GREATEST(({_sum("{c}", _BILL)}) / 6.0, 0), limit_bal) AS exposure
FROM base
"""


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    if TARGET not in clean.columns:      # при скоринге новых клиентов исхода ещё нет
        clean[TARGET] = 0
    con = duckdb.connect()
    con.register("clients", clean)
    df = con.execute(FEATURE_SQL).df()
    con.close()
    return df


def load_features(source=RAW_PATH) -> pd.DataFrame:
    return build_features(load_clean(source))


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 44, 200], labels=["<30", "30-44", "45+"]).astype(str)


def model_columns(df: pd.DataFrame, use_protected: bool = False) -> list[str]:
    drop = {ID, TARGET, "exposure"} | (set() if use_protected else set(PROTECTED))
    return [c for c in df.columns if c not in drop]
