"""CLI: python score.py input.csv output.csv [--rate 0.2 --lgd 0.6]"""
from __future__ import annotations
import argparse

import pandas as pd

from creditrisk.business import LendingAssumptions
from creditrisk.scoring import score_clients

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Скоринг клиентов обученной моделью")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--rate", type=float, default=0.20, help="доход на единицу задолженности (r)")
    ap.add_argument("--lgd", type=float, default=0.60, help="потеря при дефолте (LGD)")
    args = ap.parse_args()
    a = LendingAssumptions(args.rate, args.lgd)
    res = score_clients(pd.read_csv(args.input), a)
    res.round(4).to_csv(args.output, index=False)
    print(f"Готово: {len(res)} клиентов, одобрено {(res.decision == 'approve').mean():.1%} "
          f"(порог PD < {a.breakeven_pd:.2f}). Файл: {args.output}")
