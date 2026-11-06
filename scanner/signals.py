"""Indikatoren, Filter und die beiden Einstiegs-Setups.

Ein Signal entsteht immer am Schluss eines Tages t. Eingestiegen wird zur
Eröffnung von t+1. Alle Werte, die am Tag t verwendet werden, sind am Ende
von Tag t bekannt (kein Blick in die Zukunft).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indikatoren
# ---------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


# ---------------------------------------------------------------------------
# Aufbereitung je Aktie
# ---------------------------------------------------------------------------
def prepare(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    cfg: dict,
    min_turnover: float,
    rs_rank: pd.Series | None = None,
) -> pd.DataFrame:
    """Berechnet Indikatoren, Filter und Signale für eine Aktie.
    rs_rank: Rang der Kursentwicklung im gesamten Universum (0 = schwächste,
    1 = stärkste Aktie), am selben Tag berechnet."""
    f, sb, sp, ex = cfg["filters"], cfg["setups"]["breakout"], cfg["setups"]["pullback"], cfg["exits"]
    d = df.copy()
    c = d["close"]

    d["atr"] = atr(d, f["atr_period"])
    d["atr_pct"] = d["atr"] / c * 100
    d["sma_fast"] = sma(c, f["trend_fast_sma"])
    d["sma_slow"] = sma(c, f["trend_slow_sma"])
    d["turnover20"] = (c * d["volume"]).rolling(20, min_periods=20).mean()
    d["vol_avg20"] = d["volume"].shift(1).rolling(20, min_periods=20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_avg20"]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_pos"] = ((c - d["low"]) / rng).fillna(0.5)

    # Marktumfeld: Index über seiner Durchschnittslinie (am selben Tag bekannt)
    idx_close = index_df["close"].reindex(d.index, method="ffill")
    idx_sma = sma(index_df["close"], f["regime_sma"]).reindex(d.index, method="ffill")
    d["regime_ok"] = idx_close > idx_sma

    # Relative Stärke über 3 Monate gegenüber dem Index (für die Rangfolge)
    d["rs63"] = c.pct_change(63) - idx_close.pct_change(63)

    d["eligible"] = (
        d["regime_ok"]
        & (c > d["sma_fast"])
        & (d["sma_fast"] > d["sma_slow"])
        & d["atr_pct"].between(f["atr_pct_min"], f["atr_pct_max"])
        & (d["turnover20"] >= min_turnover)
        & (c >= f["min_price"])
    ).fillna(False)
    d["rs_rank"] = rs_rank.reindex(d.index) if rs_rank is not None else np.nan
    if f.get("rs_rank_min") is not None:
        d["eligible"] = d["eligible"] & (d["rs_rank"] >= f["rs_rank_min"]).fillna(False)

    # --- Setup 1: Ausbruch -------------------------------------------------
    hh = d["high"].shift(1).rolling(sb["lookback"], min_periods=sb["lookback"]).max()
    res = d["high"].shift(1).rolling(sb["resistance_lookback"], min_periods=60).max()
    b_stop = c - sb["stop_atr"] * d["atr"]
    b_risk = c - b_stop
    # Widerstand (52-Wochen-Hoch) darf nicht zwischen Einstieg und Ziel liegen
    resistance_ok = (res <= c) | ((res - c) >= ex["min_crv"] * b_risk)
    d["sig_breakout"] = (
        sb["enabled"]
        & d["eligible"]
        & (c > hh)
        & (d["vol_ratio"] >= sb["volume_ratio_min"])
        & (d["close_pos"] >= sb["close_position_min"])
        & resistance_ok
    ).fillna(False)
    d["b_stop"] = b_stop

    # --- Setup 2: Rücksetzer im Aufwärtstrend --------------------------------
    r = rsi(c, sp["rsi_period"])
    rsi_dip = r.shift(1).rolling(sp["rsi_window"], min_periods=1).min() < sp["rsi_max"]
    near = sma(c, sp["near_sma"])
    touched = d["low"].rolling(sp["rsi_window"] + 1).min() <= near + sp["near_sma_atr"] * d["atr"]
    reversal = c > d["high"].shift(1)
    p_stop = d["low"].rolling(sp["stop_lookback"]).min() - sp["stop_buffer_atr"] * d["atr"]
    p_target = d["high"].shift(1).rolling(sp["target_lookback"]).max()
    p_risk = c - p_stop
    p_crv = (p_target - c) / p_risk
    d["sig_pullback"] = (
        sp["enabled"]
        & d["eligible"]
        & rsi_dip
        & touched
        & reversal
        & (p_risk > 0)
        & (p_risk <= sp["max_stop_atr"] * d["atr"])
        & (p_crv >= ex["min_crv"])
    ).fillna(False)
    d["p_stop"] = p_stop
    d["p_target"] = p_target
    d["rsi"] = r
    return d


def signal_rows(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Alle Signaltage mit Setup, Stop und Ziel. Ausbruch hat Vorrang.
    Mit nachgezogenem Stop (exits.trail_atr) gibt es kein festes Ziel."""
    tr = cfg["setups"]["breakout"]["target_r"]
    trailing = cfg["exits"].get("trail_atr") is not None
    out = []
    pos = np.flatnonzero(d["sig_breakout"].to_numpy() | d["sig_pullback"].to_numpy())
    for p in pos:
        row = d.iloc[p]
        if row["sig_breakout"]:
            out.append((p, "Ausbruch", row["b_stop"], np.nan, np.nan if trailing else tr))
        else:
            target = np.nan if trailing else row["p_target"]
            out.append((p, "Rücksetzer", row["p_stop"], target, np.nan))
    return pd.DataFrame(out, columns=["pos", "setup", "stop", "target", "target_r"])
