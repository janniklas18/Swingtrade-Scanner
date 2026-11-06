"""Kennzahlen für Trades und Portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_stats(r: pd.Series, boot_runs: int = 2000, seed: int = 1) -> dict:
    """Kennzahlen auf Basis der Trade-Ergebnisse in R (Vielfache des Risikos)."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    rng = np.random.default_rng(seed)
    arr = r.to_numpy()
    boot = rng.choice(arr, size=(boot_runs, n), replace=True).mean(axis=1) if n > 1 else arr
    return {
        "n": n,
        "hit_rate": len(wins) / n,
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "expectancy": arr.mean(),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.inf,
        "total_r": arr.sum(),
    }


def equity_stats(eq: pd.Series) -> dict:
    """Rendite, Drawdown und Sharpe einer Equity-Kurve."""
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    return {
        "total_return": total,
        "cagr": cagr,
        "max_dd": dd.min(),
        "volatility": ret.std() * np.sqrt(252),
        "sharpe": sharpe,
    }


def by_group(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in trades.groupby(col):
        s = trade_stats(g["r"], boot_runs=500)
        rows.append({col: key, **s})
    return pd.DataFrame(rows)
