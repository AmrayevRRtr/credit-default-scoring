"""WoE-биннинг и классическая скоркарта (логистическая регрессия на WoE -> баллы).

Определения (good = 0, bad = default = 1):
    WoE_bin = ln( доля good в бине / доля bad в бине )
    IV      = sum( (доля good - доля bad) * WoE )
Скор:  Score = Offset + Factor * ln(odds_good), где Factor = PDO / ln 2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class WoEBinner:
    def __init__(self, max_bins: int = 8, min_share: float = 0.04, smoothing: float = 0.5):
        self.max_bins, self.min_share, self.smoothing = max_bins, min_share, smoothing

    def _edges(self, x: np.ndarray) -> np.ndarray:
        u = np.unique(x)
        if len(u) <= 1:
            return np.array([])
        if len(u) <= 15:                       # дискретный признак: каждое значение - свой бин
            edges = (u[:-1] + u[1:]) / 2
        else:                                  # непрерывный: квантили
            edges = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
        # сливаем редкие бины с соседом, у которого меньше наблюдений
        while len(edges):
            idx = np.digitize(x, edges, right=True)
            counts = np.bincount(idx, minlength=len(edges) + 1)
            small = np.where(counts < self.min_share * len(x))[0]
            if len(small) == 0:
                break
            i = small[np.argmin(counts[small])]
            if i == 0:
                edges = np.delete(edges, 0)
            elif i == len(edges):
                edges = np.delete(edges, -1)
            else:
                edges = np.delete(edges, i - 1 if counts[i - 1] <= counts[i + 1] else i)
        return edges

    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        y = np.asarray(y)
        self.edges_, self.tables_, self.iv_ = {}, {}, {}
        G, B = (y == 0).sum(), (y == 1).sum()
        s = self.smoothing
        for c in X.columns:
            x = X[c].to_numpy(float)
            e = self._edges(x)
            idx = np.digitize(x, e, right=True)
            k = len(e) + 1
            good = np.bincount(idx[y == 0], minlength=k)
            bad = np.bincount(idx[y == 1], minlength=k)
            dg, db = (good + s) / (G + s * k), (bad + s) / (B + s * k)
            woe = np.log(dg / db)
            lo = np.r_[-np.inf, e]
            hi = np.r_[e, np.inf]
            self.edges_[c] = e
            self.tables_[c] = pd.DataFrame({
                "feature": c, "bin": [self._label(a, b) for a, b in zip(lo, hi)], "n": good + bad,
                "bad_rate": bad / np.maximum(good + bad, 1), "woe": woe})
            self.iv_[c] = float(((dg - db) * woe).sum())
        return self

    @staticmethod
    def _label(lo, hi) -> str:
        f = lambda v: f"{v:.4g}"
        if np.isinf(lo):
            return f"<= {f(hi)}"
        if np.isinf(hi):
            return f"> {f(lo)}"
        return f"({f(lo)}, {f(hi)}]"

    def bin_index(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: np.digitize(X[c].to_numpy(float), self.edges_[c], right=True)
                             for c in self.edges_}, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        idx = self.bin_index(X)
        return pd.DataFrame({c: self.tables_[c]["woe"].to_numpy()[idx[c].to_numpy()] for c in self.edges_},
                            index=X.index)


class Scorecard(ClassifierMixin, BaseEstimator):
    """Скоркарта: WoE-биннинг -> отбор по IV и корреляции -> логрегрессия -> баллы."""

    def __init__(self, min_iv: float = 0.02, max_corr: float = 0.8, C: float = 1.0,
                 max_bins: int = 8, min_share: float = 0.04,
                 base_score: float = 600, base_odds: float = 50, pdo: float = 20):
        self.min_iv, self.max_corr, self.C = min_iv, max_corr, C
        self.max_bins, self.min_share = max_bins, min_share
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.binner_ = WoEBinner(self.max_bins, self.min_share).fit(X, y)
        W = self.binner_.transform(X)
        iv = pd.Series(self.binner_.iv_).sort_values(ascending=False)
        feats = [f for f in iv.index if iv[f] >= self.min_iv]
        # убираем сильно коррелированные (оставляем с большим IV)
        corr = W[feats].corr().abs()
        keep = []
        for f in feats:
            if all(corr.loc[f, k] < self.max_corr for k in keep):
                keep.append(f)
        # убираем признаки с "неправильным" знаком коэффициента (WoE выше -> риск ниже)
        while True:
            lr = LogisticRegression(C=self.C, max_iter=2000).fit(W[keep], y)
            bad = [f for f, b in zip(keep, lr.coef_[0]) if b > 0]
            if not bad:
                break
            drop = min(bad, key=lambda f: iv[f])
            keep.remove(drop)
        self.features_, self.lr_ = keep, lr
        self.iv_ = iv
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    def _woe(self, X):
        return self.binner_.transform(X[list(self.binner_.edges_)])[self.features_]

    def predict_proba(self, X):
        p = self.lr_.predict_proba(self._woe(X))[:, 1]
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------ баллы
    def points_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Баллы за каждый признак: -(b_i * WoE_i + b0 / n) * Factor + Offset / n."""
        W, n = self._woe(X), len(self.features_)
        b, b0 = self.lr_.coef_[0], self.lr_.intercept_[0]
        P = -(W.to_numpy() * b + b0 / n) * self.factor_ + self.offset_ / n
        return pd.DataFrame(P, columns=self.features_, index=X.index)

    def score(self, X: pd.DataFrame) -> pd.Series:
        return self.points_matrix(X).sum(axis=1).rename("score")

    def scorecard_table(self) -> pd.DataFrame:
        n = len(self.features_)
        b, b0 = self.lr_.coef_[0], self.lr_.intercept_[0]
        rows = []
        for f, bi in zip(self.features_, b):
            t = self.binner_.tables_[f].copy()
            t["points"] = -(t["woe"] * bi + b0 / n) * self.factor_ + self.offset_ / n
            t["iv"] = self.iv_[f]
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> list[str]:
        """Топ-k признаков, по которым клиент теряет больше всего баллов относительно лучшего бина."""
        P = self.points_matrix(X)
        best = self.scorecard_table().groupby("feature")["points"].max()
        short = (best.reindex(P.columns) - P).clip(lower=0)
        return [", ".join(short.columns[np.argsort(-r)[:k]][np.sort(-r)[:k] < 0])
                for r in short.to_numpy()]
