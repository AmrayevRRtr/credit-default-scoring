import numpy as np
import pandas as pd
import pytest

from creditrisk.business import LendingAssumptions, realized_profit, empirical_threshold
from creditrisk.data import ID, TARGET, load_features, model_columns, PROTECTED
from creditrisk.evaluation import ks_stat, psi
from creditrisk.woe import Scorecard, WoEBinner

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def df():
    return load_features()


def test_features_clean(df):
    assert df.isna().sum().sum() == 0
    assert df[ID].is_unique and set(df[TARGET].unique()) == {0, 1}
    assert 0.15 < df[TARGET].mean() < 0.3
    assert (df.exposure >= 0).all() and (df.exposure <= df.limit_bal).all()


def test_protected_excluded(df):
    cols = model_columns(df)
    assert not set(PROTECTED) & set(cols)
    assert set(PROTECTED) <= set(model_columns(df, use_protected=True))
    assert TARGET not in cols and "exposure" not in cols


def test_woe_informative_vs_noise():
    n = 5000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x * 1.5 - 1)))).astype(int)
    X = pd.DataFrame({"signal": x, "noise": rng.normal(size=n)})
    b = WoEBinner().fit(X, y)
    assert b.iv_["signal"] > 0.3 > b.iv_["noise"]
    assert b.transform(X).shape == X.shape
    assert all(t.n.min() >= 0.04 * n * 0.99 for t in b.tables_.values())   # редкие бины слиты


def test_scorecard_score_monotone_with_pd(df):
    cols = model_columns(df)
    sc = Scorecard().fit(df[cols], df[TARGET])
    p = sc.predict_proba(df[cols])[:, 1]
    s = sc.score(df[cols])
    assert np.corrcoef(s, np.log(p / (1 - p)))[0, 1] < -0.999      # больше баллов = меньше риск
    assert all(b <= 0 for b in sc.lr_.coef_[0])                    # WoE выше -> риск ниже
    assert len(sc.reason_codes(df[cols].head(3))) == 3


def test_breakeven_and_profit():
    a = LendingAssumptions(0.2, 0.6)
    assert a.breakeven_pd == pytest.approx(0.25)
    y, e = np.array([0, 1]), np.array([100.0, 100.0])
    assert realized_profit(y, e, [True, True], a) == pytest.approx(20 - 60)
    assert realized_profit(y, e, [False, False], a) == 0


def test_empirical_threshold_near_breakeven_for_calibrated_pd():
    a = LendingAssumptions(0.2, 0.6)
    p = rng.beta(2, 6, 40000)
    y = (rng.random(40000) < p).astype(int)
    t = empirical_threshold(y, p, np.ones(40000), a)
    assert abs(t - a.breakeven_pd) < 0.06


def test_metrics():
    assert psi(np.arange(1000), np.arange(1000)) == pytest.approx(0, abs=1e-6)
    assert psi(np.arange(1000), np.arange(1000) + 500) > 0.25
    assert ks_stat([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_scoring_end_to_end():
    from creditrisk.data import RAW_PATH, ROOT
    if not (ROOT / "models" / "pd_model.joblib").exists():
        pytest.skip("модели не обучены: python -m creditrisk.pipeline")
    from creditrisk.scoring import load_artifacts, score_clients
    try:
        load_artifacts()
    except RuntimeError as e:
        pytest.skip(str(e))
    raw = pd.read_csv(RAW_PATH).drop(columns=["default"]).head(50)
    res = score_clients(raw)
    assert len(res) == 50 and res.pd.between(0, 1).all()
    assert set(res.decision) <= {"approve", "decline"}
