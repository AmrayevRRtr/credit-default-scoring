"""Полный пайплайн: данные -> EDA -> модели -> калибровка -> бизнес-порог -> справедливость -> артефакты.
Запуск: python -m creditrisk.pipeline
"""
from __future__ import annotations
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from creditrisk import business as biz
from creditrisk import fairness
from creditrisk.data import ID, ROOT, TARGET, age_group, load_features, model_columns
from creditrisk.evaluation import decile_table, ks_stat, psi, summarize
from creditrisk.woe import Scorecard

warnings.filterwarnings("ignore")
SEED = 42
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


def to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return o.round(4).to_dict("records")
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    return o


def lgbm(**kw):
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1, subsample_freq=1, **kw)


# ------------------------------------------------------------------ графики
def fig_eda(df):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    r = df.groupby(df.pay_1.clip(-2, 4))[TARGET].mean() * 100
    ax[0, 0].bar(r.index.astype(str), r.values, color=BLUE)
    ax[0, 0].set(title="Дефолт по статусу платежа последнего месяца (pay_1), %", xlabel="pay_1 (-2/-1/0 - без просрочки, 1+ - месяцев просрочки)")
    r = df.groupby(pd.qcut(df.limit_bal, 5), observed=True)[TARGET].mean() * 100
    ax[0, 1].bar([f"{int(i.left/1000)}-{int(i.right/1000)}k" for i in r.index], r.values, color=GREEN)
    ax[0, 1].set(title="Дефолт по кредитному лимиту (квинтили), %")
    r = df.groupby(pd.cut(df.util_avg, [-1, 0.1, 0.3, 0.6, 0.9, 10]), observed=True)[TARGET].mean() * 100
    ax[1, 0].bar(["<=10%", "10-30%", "30-60%", "60-90%", ">90%"], r.values, color=ORANGE)
    ax[1, 0].set(title="Дефолт по средней утилизации лимита, %")
    r = df.groupby("n_delay_months")[TARGET].mean() * 100
    ax[1, 1].plot(r.index, r.values, marker="o", color=RED)
    ax[1, 1].set(title="Дефолт по числу месяцев с просрочкой за 6 мес., %", xlabel="месяцев с просрочкой")
    fig.tight_layout(); fig.savefig(FIG / "01_eda.png"); plt.close(fig)


def fig_quality(y, preds: dict, best: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for n, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{n} (AUC {summarize(y, p)['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].legend(fontsize=8)
    ax[0].set(title="ROC (hold-out)", xlabel="FPR", ylabel="TPR")
    p = preds[best]
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    cb, cg = np.cumsum(ys) / ys.sum(), np.cumsum(1 - ys) / (1 - ys).sum()
    x = np.arange(1, len(ys) + 1) / len(ys)
    ax[1].plot(x, cb, color=RED, label="bad (дефолт)"); ax[1].plot(x, cg, color=GREEN, label="good")
    k = np.argmax(np.abs(cb - cg))
    ax[1].vlines(x[k], cg[k], cb[k], color="k", ls="--", label=f"KS = {np.max(np.abs(cb - cg)):.3f}")
    ax[1].legend(); ax[1].set(title=f"KS-график ({best})", xlabel="доля клиентов (по возрастанию PD)", ylabel="кумулятивная доля")
    dt = decile_table(y, p)
    ax[2].bar(dt.decile, dt.default_rate * 100, color=BLUE)
    ax[2].axhline(np.mean(y) * 100, color="k", ls="--", lw=0.8)
    ax[2].set(title="Фактический дефолт по децилям PD, %", xlabel="дециль (1 = самый рискованный)")
    fig.tight_layout(); fig.savefig(FIG / "02_model_quality.png"); plt.close(fig)


def fig_calibration(y, p_raw, p_cal, score_card_scores, y_sc):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, p in [("LightGBM до калибровки", p_raw), ("LightGBM после калибровки", p_cal)]:
        f, m = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[0].plot(m, f, marker="o", label=lab)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].legend()
    ax[0].set(title="Калибровка PD (hold-out)", xlabel="предсказанная PD", ylabel="фактическая доля дефолтов")
    s, yy = np.asarray(score_card_scores), np.asarray(y_sc)
    bins = np.linspace(s.min(), s.max(), 30)
    ax[1].hist(s[yy == 0], bins, alpha=0.6, color=GREEN, label="good", density=True)
    ax[1].hist(s[yy == 1], bins, alpha=0.6, color=RED, label="bad", density=True)
    ax[1].legend(); ax[1].set(title="Распределение скоркарты (баллы)", xlabel="баллы")
    fig.tight_layout(); fig.savefig(FIG / "03_calibration_scores.png"); plt.close(fig)


def fig_iv(sc: Scorecard):
    iv = sc.iv_.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(iv.index, iv.values, color=[BLUE if f in sc.features_ else "#BBBBBB" for f in iv.index])
    ax.axvline(0.02, color="k", ls="--", lw=0.8)
    ax.set(title="Information Value (синие - вошли в скоркарту)", xlabel="IV")
    fig.tight_layout(); fig.savefig(FIG / "05_information_value.png"); plt.close(fig)


def fig_profit(curve_lgb, curve_sc, a, t_lgb, strat):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(curve_lgb.threshold, curve_lgb.profit / 1e6, color=BLUE, label="LightGBM")
    ax[0].plot(curve_sc.threshold, curve_sc.profit / 1e6, color=ORANGE, label="Скоркарта")
    ax[0].axvline(a.breakeven_pd, color="k", ls="--", lw=0.8, label=f"порог безубыточности {a.breakeven_pd:.2f}")
    ax[0].set(title="Прибыль vs порог одобрения PD (hold-out)", xlabel="одобряем, если PD < порога", ylabel="прибыль, млн NT$")
    ax[0].legend()
    s = strat.sort_values("profit")
    ax[1].barh(s.strategy, s.profit / 1e6, color=[RED if v < 0 else BLUE for v in s.profit])
    ax[1].set(title="Сравнение стратегий на hold-out", xlabel="прибыль, млн NT$")
    fig.tight_layout(); fig.savefig(FIG / "06_profit.png"); plt.close(fig)


def fig_fairness(tab):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = tab.attribute + ": " + tab.group
    ax[0].barh(labels, tab.approval_rate * 100, color=BLUE)
    ax[0].set(title="Доля одобренных, %")
    ax[1].barh(labels, tab.approval_rate_among_good * 100, color=GREEN)
    ax[1].set(title="Доля одобренных среди добросовестных клиентов, %")
    fig.tight_layout(); fig.savefig(FIG / "07_fairness.png"); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    a = biz.LendingAssumptions()
    df = load_features()
    cols = model_columns(df)
    tr, te = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    Xtr, ytr, Xte, yte = tr[cols], tr[TARGET].values, te[cols], te[TARGET].values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    M = {"n_clients": len(df), "n_features": len(cols), "default_rate": float(df[TARGET].mean()),
         "n_train": len(tr), "n_test": len(te), "features": cols}
    fig_eda(df)

    # ---- 1. тюнинг LightGBM по ROC-AUC
    search = RandomizedSearchCV(
        lgbm(), {"n_estimators": [200, 400, 600], "learning_rate": [0.01, 0.02, 0.03, 0.05],
                 "num_leaves": [4, 8, 16, 31], "min_child_samples": [20, 50, 100],
                 "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.5, 0.7, 0.9],
                 "reg_lambda": [0, 1, 5, 20]},
        n_iter=30, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    tuned = search.best_estimator_

    # ---- 2. кросс-валидация кандидатов (OOF-предсказания на train)
    candidates = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "Scorecard (WoE+LogReg)": Scorecard(),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "LightGBM": lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15),
        "LightGBM (tuned)": tuned,
    }
    rows = []
    for n, m in candidates.items():
        oof, fold_auc = np.zeros(len(ytr)), []
        for i_tr, i_va in cv.split(Xtr, ytr):          # один проход: OOF-предсказания + AUC по фолдам
            mm = clone(m).fit(Xtr.iloc[i_tr], ytr[i_tr])
            oof[i_va] = mm.predict_proba(Xtr.iloc[i_va])[:, 1]
            fold_auc.append(roc_auc_score(ytr[i_va], oof[i_va]))
        s = summarize(ytr, oof)
        rows.append({"model": n, "cv_roc_auc": s["roc_auc"], "cv_auc_std": float(np.std(fold_auc)), "cv_gini": s["gini"],
                     "cv_ks": s["ks"], "cv_pr_auc": s["pr_auc"]})
    cv_df = pd.DataFrame(rows)
    M["cv"], M["best_params"] = cv_df, search.best_params_

    # ---- 3. финальные модели на train, оценка на hold-out
    calibrated = CalibratedClassifierCV(tuned, method="isotonic", cv=5).fit(Xtr, ytr)
    tuned_fit = lgbm(**search.best_params_).fit(Xtr, ytr)
    scorecard = Scorecard().fit(Xtr, ytr)
    logreg = candidates["LogReg"].fit(Xtr, ytr)
    p_raw = tuned_fit.predict_proba(Xte)[:, 1]
    p_cal = calibrated.predict_proba(Xte)[:, 1]
    p_sc = scorecard.predict_proba(Xte)[:, 1]
    p_lr = logreg.predict_proba(Xte)[:, 1]
    sc_score = scorecard.score(Xte)
    preds = {"LogReg": p_lr, "Scorecard": p_sc, "LightGBM (calibrated)": p_cal}
    M["holdout"] = {"LogReg": summarize(yte, p_lr), "Scorecard": summarize(yte, p_sc),
                    "LightGBM_raw": summarize(yte, p_raw), "LightGBM_calibrated": summarize(yte, p_cal)}
    M["decile_table"] = decile_table(yte, p_cal)
    fig_quality(yte, preds, "LightGBM (calibrated)")
    fig_calibration(yte, p_raw, p_cal, sc_score, yte)
    fig_iv(scorecard)

    # ---- 4. ablation: что дало бы использование защищённых признаков (sex, marriage)
    cols_p = model_columns(df, use_protected=True)
    m_prot = lgbm(**search.best_params_).fit(tr[cols_p], ytr)
    p_prot = m_prot.predict_proba(te[cols_p])[:, 1]
    M["ablation_protected"] = {"auc_without": M["holdout"]["LightGBM_raw"]["roc_auc"],
                               "auc_with": summarize(yte, p_prot)["roc_auc"]}

    # ---- 5. стабильность (PSI)
    Xtr_sc = scorecard.score(Xtr)
    M["psi"] = {"score_train_vs_test": psi(Xtr_sc, sc_score),
                "top_features": {c: psi(Xtr[c], Xte[c]) for c in scorecard.features_[:8]}}

    # ---- 6. SHAP
    sv = shap.TreeExplainer(tuned_fit).shap_values(Xte.sample(2000, random_state=SEED))
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    plt.figure(); shap.summary_plot(sv, Xte.sample(2000, random_state=SEED), show=False, max_display=15)
    plt.title("SHAP: факторы PD (LightGBM)"); plt.tight_layout()
    plt.savefig(FIG / "04_shap_summary.png", bbox_inches="tight"); plt.close()
    M["shap_top"] = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False).head(10).round(4).to_dict()

    # ---- 7. бизнес: порог и стратегии
    oof_cal = cross_val_predict(CalibratedClassifierCV(tuned, method="isotonic", cv=5), Xtr, ytr,
                                cv=cv, method="predict_proba")[:, 1]
    t_emp = biz.empirical_threshold(ytr, oof_cal, tr.exposure.values, a)   # подобран на train (OOF), не на test
    t_be = a.breakeven_pd
    exp_te = te.exposure.values
    strategies = {
        "Отказать всем": np.zeros(len(te), bool),
        "Одобрить всех": np.ones(len(te), bool),
        "Правило: нет просрочки сейчас (pay_1 <= 0)": (te.pay_1 <= 0).values,
        "Скоркарта, PD < порога безубыточности": p_sc < t_be,
        "LightGBM, PD < порога безубыточности": p_cal < t_be,
        "LightGBM, порог по OOF-прибыли": p_cal < t_emp,
    }
    strat = biz.strategy_table(yte, exp_te, strategies, a)
    curve_l, curve_s = biz.threshold_curve(yte, p_cal, exp_te, a), biz.threshold_curve(yte, p_sc, exp_te, a)
    fig_profit(curve_l, curve_s, a, t_be, strat)
    sens = []
    for r in (0.10, 0.15, 0.20, 0.30):
        for lgd in (0.4, 0.6, 0.8):
            aa = biz.LendingAssumptions(r, lgd)
            st = biz.strategy_table(yte, exp_te, {"model": p_cal < aa.breakeven_pd,
                                                  "rule": (te.pay_1 <= 0).values,
                                                  "all": np.ones(len(te), bool)}, aa).set_index("strategy")
            sens.append({"interest_rate": r, "lgd": lgd, "breakeven_pd": aa.breakeven_pd,
                         "profit_model": st.loc["model", "profit"], "profit_rule": st.loc["rule", "profit"],
                         "profit_approve_all": st.loc["all", "profit"], "approval_rate_model": st.loc["model", "approval_rate"]})
    rng = np.random.default_rng(SEED)
    def _p(m_, idx):
        return biz.realized_profit(yte[idx], exp_te[idx], m_[idx], a)
    m_mod, m_rule = strategies["LightGBM, PD < порога безубыточности"], strategies["Правило: нет просрочки сейчас (pay_1 <= 0)"]
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, len(yte), len(yte))
        diffs.append(_p(m_mod, idx) - _p(m_rule, idx))
    boot = {"profit_diff_model_vs_rule_mean": float(np.mean(diffs)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_bootstrap_model_better": float(np.mean(np.array(diffs) > 0))}
    M["business"] = {"bootstrap_model_vs_rule": boot, "assumptions": a.__dict__, "breakeven_pd": t_be, "empirical_threshold_oof": t_emp,
                     "strategies": strat, "sensitivity": pd.DataFrame(sens)}

    # ---- 8. справедливость
    approve = p_cal < t_be
    fa = fairness.audit(te, p_cal, approve)
    fa_prot = fairness.audit(te, p_prot, p_prot < t_be)
    fig_fairness(fa)
    M["fairness"] = {"table": fa, "disparate_impact": fairness.disparate_impact(fa),
                     "disparate_impact_with_protected": fairness.disparate_impact(fa_prot)}

    # ---- 9. артефакты
    joblib.dump({"model": calibrated, "features": cols, "name": "LightGBM (tuned, calibrated)",
                 "breakeven_pd": t_be}, ROOT / "models" / "pd_model.joblib")
    joblib.dump({"scorecard": scorecard, "features": cols}, ROOT / "models" / "scorecard.joblib")
    scorecard.scorecard_table().round(4).to_csv(ROOT / "reports" / "scorecard_table.csv", index=False)
    out = te[[ID, TARGET, "limit_bal", "exposure"]].copy()
    out["pd"], out["score"] = p_cal, sc_score.values
    out["decision"] = np.where(approve, "approve", "decline")
    reasons = scorecard.reason_codes(Xte)
    out["decline_reasons"] = [r if d == "decline" else "" for r, d in zip(reasons, out.decision)]
    out.round(4).to_csv(ROOT / "outputs" / "scored_test_clients.csv", index=False)
    M["scorecard"] = {"n_features": len(scorecard.features_), "features": scorecard.features_,
                      "factor": scorecard.factor_, "offset": scorecard.offset_,
                      "iv_top": scorecard.iv_.head(10).round(3).to_dict()}
    with open(ROOT / "reports" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(M), f, ensure_ascii=False, indent=2)

    print(cv_df.round(4).to_string()); print(pd.DataFrame(M["holdout"]).round(4).to_string())
    print(M["ablation_protected"], M["psi"]["score_train_vs_test"])
    print(strat.round(3).to_string()); print("breakeven", round(t_be, 3), "empirical(OOF)", round(t_emp, 3))
    print(fa.round(3).to_string()); print(M["fairness"]["disparate_impact"], M["fairness"]["disparate_impact_with_protected"])
    print(boot)
    print(pd.DataFrame(sens).round(2).to_string()); print(M["shap_top"]); print(search.best_params_)


if __name__ == "__main__":
    main()
