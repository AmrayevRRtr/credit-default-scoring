"""Демо: загрузите CSV клиентов (формат UCI Credit Default) и получите PD, баллы, решение и причины отказа.
Запуск: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from creditrisk.business import LendingAssumptions
from creditrisk.scoring import load_artifacts, score_clients

st.set_page_config(page_title="Credit Scoring", layout="wide")
st.title("Кредитный скоринг: вероятность дефолта (PD)")
st.caption("Модель: LightGBM + калибровка. Баллы и причины отказа - из интерпретируемой скоркарты (WoE + логрегрессия).")

with st.sidebar:
    st.header("Экономика кредита (допущения)")
    r = st.slider("Доход на единицу задолженности (r)", 0.05, 0.40, 0.20, 0.01)
    lgd = st.slider("Потеря при дефолте (LGD)", 0.2, 0.9, 0.60, 0.05)
    a = LendingAssumptions(r, lgd)
    st.metric("Порог одобрения: PD <", f"{a.breakeven_pd:.2f}")
    st.caption("Одобряем, если ожидаемая прибыль положительна: PD < r / (r + LGD).")

up = st.file_uploader("CSV с клиентами", type="csv")
raw = pd.read_csv(up) if up else pd.read_csv(ROOT / "data" / "raw" / "credit_default.csv").drop(columns=["default"]).head(500)
res = score_clients(raw, a)

c1, c2, c3 = st.columns(3)
c1.metric("Клиентов", len(res))
c2.metric("Доля одобренных", f"{(res.decision == 'approve').mean():.1%}")
c3.metric("Средняя PD", f"{res.pd.mean():.1%}")

st.subheader("Результаты скоринга")
st.dataframe(res.style.format({"pd": "{:.1%}", "score": "{:.0f}", "limit_bal": "{:,.0f}"}), width="stretch", height=380)

st.subheader("Разбор клиента")
cid = st.selectbox("client_id", res.client_id.tolist())
row = res[res.client_id == cid].iloc[0]
k1, k2, k3 = st.columns(3)
k1.metric("PD", f"{row.pd:.1%}"); k2.metric("Баллы", f"{row.score:.0f}"); k3.metric("Решение", row.decision)
if row.decision == "decline":
    st.warning(f"Основные причины отказа: {row.decline_reasons}")
pdm, sc = load_artifacts()
from creditrisk.data import load_clean, build_features
feats = build_features(load_clean(raw))
X = feats[pdm["features"]]
pts = sc["scorecard"].points_matrix(X[feats.client_id == cid]).T
pts.columns = ["баллы"]
st.bar_chart(pts)
st.download_button("Скачать результат (CSV)", res.round(4).to_csv(index=False).encode("utf-8"), "scored.csv")
