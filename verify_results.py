"""Check the headline results without retraining.

Recomputes ROC-AUC / Gini / KS and the profit of the approval policy from
outputs/scored_test_clients.csv (hold-out predictions) and compares them with
reports/metrics.json written by the training pipeline.

Usage: python verify_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
m = json.load(open(ROOT / "reports" / "metrics.json", encoding="utf-8"))
d = pd.read_csv(ROOT / "outputs" / "scored_test_clients.csv")

auc = roc_auc_score(d["default"], d["pd"])
fpr, tpr, _ = roc_curve(d["default"], d["pd"])
ref = m["holdout"]["LightGBM_calibrated"]

a = m["business"]["assumptions"]
y, e = d["default"].to_numpy(), d["exposure"].to_numpy()
ok = (d["decision"] == "approve").to_numpy()
profit = float(((1 - y[ok]) * a["interest_rate"] * e[ok] - y[ok] * a["lgd"] * e[ok]).sum())
ref_profit = next(s["profit"] for s in m["business"]["strategies"] if s["strategy"].startswith("LightGBM, PD <"))

checks = [
    ("ROC-AUC", auc, ref["roc_auc"], 1e-3),
    ("Gini", 2 * auc - 1, ref["gini"], 2e-3),
    ("KS", float(np.max(tpr - fpr)), ref["ks"], 2e-3),
    ("Profit, NT$ (decision = approve)", profit, ref_profit, 1.0),
]
print(f"Hold-out clients: {len(d)}, default rate {y.mean():.3f}, approval rate {ok.mean():.3f}\n")
failed = False
for name, got, want, tol in checks:
    good = abs(got - want) <= tol
    failed |= not good
    print(f"{'OK  ' if good else 'FAIL'} {name:34s} recomputed={got:>14,.4f}  metrics.json={want:>14,.4f}")
sys.exit(1 if failed else 0)
