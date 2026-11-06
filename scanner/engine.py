"""Backtest-Engine: einzelne Trades, Zufalls-Benchmark und Portfolio.

Annahmen (bewusst vorsichtig):
- Einstieg zur Eröffnung des Tages nach dem Signal.
- Eröffnet die Aktie unter dem Stop oder über dem Ziel, wird der Trade nicht
  eingegangen.
- Öffnet die Aktie während des Trades unter dem Stop (Kurslücke), wird zur
  Eröffnung verkauft, nicht zum Stopkurs.
- Werden Stop und Ziel am selben Tag berührt, zählt der Stop.
- Nach time_stop_days Handelstagen wird zum Schlusskurs verkauft.
- Kosten (Spread + Slippage) fallen bei Kauf und Verkauf an.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals as sig


@dataclass
class TradeResult:
    entry_pos: int
    exit_pos: int
    entry: float
    stop: float
    target: float
    exit: float
    reason: str
    r_gross: float
    r: float
    mfe_r: float
    mae_r: float


def simulate_trade(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    entry_pos: int,
    stop: float,
    target: float,
    target_r: float,
    time_stop: int,
    cost_bps: float,
    breakeven_after_r: float | None = None,
) -> TradeResult | str:
    """Simuliert einen Trade. Gibt bei ungültigem Einstieg einen Grund zurück."""
    n = len(c)
    if entry_pos >= n:
        return "keine Daten"
    entry = o[entry_pos]
    if entry <= stop:
        return "Eröffnung unter Stop"
    risk = entry - stop
    if np.isnan(target):
        target = entry + target_r * risk
    if entry >= target:
        return "Eröffnung über Ziel"

    cur_stop = stop
    last = min(entry_pos + time_stop - 1, n - 1)
    exit_px, reason, exit_pos = np.nan, "", last
    mfe, mae = 0.0, 0.0
    for d in range(entry_pos, last + 1):
        if d > entry_pos:
            if o[d] <= cur_stop:
                exit_px, reason, exit_pos = o[d], "Stop (Kurslücke)", d
                break
            if o[d] >= target:
                exit_px, reason, exit_pos = o[d], "Ziel (Kurslücke)", d
                break
        mfe = max(mfe, (h[d] - entry) / risk)
        mae = min(mae, (l[d] - entry) / risk)
        hit_stop, hit_tgt = l[d] <= cur_stop, h[d] >= target
        if hit_stop:  # bei beidem am selben Tag: Stop (vorsichtig)
            exit_px, exit_pos = cur_stop, d
            reason = "Einstand" if cur_stop >= entry else "Stop"
            break
        if hit_tgt:
            exit_px, reason, exit_pos = target, "Ziel", d
            break
        if d == last:
            exit_px, exit_pos = c[d], d
            reason = "Zeitstopp" if last == entry_pos + time_stop - 1 else "Datenende"
            break
        if breakeven_after_r and c[d] >= entry + breakeven_after_r * risk:
            cur_stop = max(cur_stop, entry)

    cost = (entry + exit_px) * cost_bps / 1e4
    return TradeResult(
        entry_pos=entry_pos,
        exit_pos=exit_pos,
        entry=entry,
        stop=stop,
        target=target,
        exit=exit_px,
        reason=reason,
        r_gross=(exit_px - entry) / risk,
        r=(exit_px - entry - cost) / risk,
        mfe_r=mfe,
        mae_r=mae,
    )


def _has_earnings(dates: list[pd.Timestamp] | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not dates:
        return False
    return any(start <= d <= end for d in dates)


def generate_trades(
    prices: dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: dict,
    market: dict,
    earnings: dict[str, list[pd.Timestamp]],
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    """Alle Trades aller Signale (je Aktie ohne Überlappung).
    Gibt Trades, Zähler für verworfene Signale und die aufbereiteten Daten zurück."""
    ex = cfg["exits"]
    use_blackout = cfg["filters"].get("earnings_blackout", True)
    sectors = universe.set_index("ticker")["sector"].to_dict()
    names = universe.set_index("ticker")["name"].to_dict()
    skipped: dict[str, int] = {}
    rows = []
    prepared: dict[str, pd.DataFrame] = {}

    for t, df in prices.items():
        if t.startswith("^") or len(df) < 260:
            continue
        d = sig.prepare(df, index_df, cfg, market["min_turnover"])
        prepared[t] = d
        o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        idx = d.index
        free_from = 0
        for s in sig.signal_rows(d, cfg).itertuples():
            if s.pos < free_from or s.pos + 1 >= len(d):
                continue
            entry_pos = s.pos + 1
            window_end = idx[min(entry_pos + ex["time_stop_days"] - 1, len(d) - 1)]
            if use_blackout and _has_earnings(earnings.get(t), idx[entry_pos], window_end):
                skipped["Quartalszahlen in Haltedauer"] = skipped.get("Quartalszahlen in Haltedauer", 0) + 1
                continue
            res = simulate_trade(
                o, h, l, c, entry_pos, s.stop, s.target, s.target_r,
                ex["time_stop_days"], market["cost_per_side_bps"], ex.get("breakeven_after_r"),
            )
            if isinstance(res, str):
                skipped[res] = skipped.get(res, 0) + 1
                continue
            free_from = res.exit_pos
            if res.reason == "Datenende":  # Trade noch offen, nicht werten
                skipped["noch offen am Datenende"] = skipped.get("noch offen am Datenende", 0) + 1
                continue
            sig_row = d.iloc[s.pos]
            rows.append(
                {
                    "ticker": t,
                    "name": names.get(t, t),
                    "sector": sectors.get(t, "Unbekannt"),
                    "setup": s.setup,
                    "signal_date": idx[s.pos],
                    "entry_date": idx[res.entry_pos],
                    "exit_date": idx[res.exit_pos],
                    "entry": res.entry,
                    "stop": res.stop,
                    "target": res.target,
                    "exit": res.exit,
                    "reason": res.reason,
                    "days": res.exit_pos - res.entry_pos + 1,
                    "plan_crv": (res.target - res.entry) / (res.entry - res.stop),
                    "risk_atr": (sig_row["close"] - res.stop) / sig_row["atr"],
                    "r_gross": res.r_gross,
                    "r": res.r,
                    "mfe_r": res.mfe_r,
                    "mae_r": res.mae_r,
                    "score": sig_row["rs63"],
                    "earnings_known": t in earnings,
                }
            )
    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values("entry_date").reset_index(drop=True)
    return trades, skipped, prepared


# ---------------------------------------------------------------------------
# Zufalls-Benchmark
# ---------------------------------------------------------------------------
def random_benchmark(
    trades: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    cfg: dict,
    market: dict,
    split_date: pd.Timestamp,
    runs: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Ersetzt jeden echten Trade durch einen Einstieg an einem zufälligen Tag,
    an dem dieselben Filter erfüllt waren (gleiche Aktie, gleiche Periode,
    gleiches Risiko in ATR, gleiches CRV, gleiche Ausstiegsregeln).
    Misst damit, ob das Einstiegssignal mehr bringt als der Filter allein."""
    rng = np.random.default_rng(seed)
    ex = cfg["exits"]
    eligible_pos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t, d in prepared.items():
        p = np.flatnonzero(d["eligible"].to_numpy()[:-1])
        is_mask = d.index[p] <= split_date
        eligible_pos[t] = (p[is_mask], p[~is_mask])

    out = []
    for run in range(runs):
        rs = {"is": [], "oos": []}
        for tr in trades.itertuples():
            d = prepared[tr.ticker]
            period = "is" if tr.signal_date <= split_date else "oos"
            pool = eligible_pos[tr.ticker][0 if period == "is" else 1]
            if len(pool) == 0:
                continue
            p = int(rng.choice(pool))
            close, a = d["close"].iat[p], d["atr"].iat[p]
            stop = close - tr.risk_atr * a
            res = simulate_trade(
                d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
                p + 1, stop, np.nan, tr.plan_crv, ex["time_stop_days"],
                market["cost_per_side_bps"], ex.get("breakeven_after_r"),
            )
            if not isinstance(res, str):
                rs[period].append(res.r)
        out.append(
            {
                "run": run,
                "is_mean_r": np.mean(rs["is"]) if rs["is"] else np.nan,
                "oos_mean_r": np.mean(rs["oos"]) if rs["oos"] else np.nan,
                "all_mean_r": np.mean(rs["is"] + rs["oos"]) if rs["is"] or rs["oos"] else np.nan,
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Portfolio-Simulation
# ---------------------------------------------------------------------------
def simulate_portfolio(
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    cfg: dict,
    market: dict,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Spielt die Trades mit begrenztem Kapital durch.
    Rangfolge bei mehreren Signalen am selben Tag: relative Stärke (rs63).
    Gibt Equity-Kurve, ausgeführte Trades und investierten Anteil zurück."""
    bt = cfg["backtest"]
    bps, fee = market["cost_per_side_bps"] / 1e4, market["commission_per_trade"]
    cash = float(bt["initial_capital"])
    closes = pd.DataFrame({t: prices[t]["close"] for t in trades["ticker"].unique()}).reindex(
        calendar
    ).ffill() if not trades.empty else pd.DataFrame(index=calendar)

    by_entry = {k: g.sort_values("score", ascending=False) for k, g in trades.groupby("entry_date")} if not trades.empty else {}
    open_pos: dict[int, dict] = {}
    taken = []
    equity_vals, invested_vals = [], []
    equity = cash

    for day in calendar:
        # 1) Einstiege zur Eröffnung (Kapazität aus Vortagen)
        for tr in by_entry.get(day, pd.DataFrame()).itertuples():
            if len(open_pos) >= bt["max_positions"]:
                break
            if any(p["ticker"] == tr.ticker for p in open_pos.values()):
                continue
            if sum(p["sector"] == tr.sector for p in open_pos.values()) >= bt["max_per_sector"]:
                continue
            risk_ps = tr.entry - tr.stop
            shares = np.floor(equity * bt["risk_per_trade"] / risk_ps)
            shares = min(shares, np.floor(equity * bt["max_position_pct"] / tr.entry))
            shares = min(shares, np.floor((cash - fee) / (tr.entry * (1 + bps))))
            if shares < 1:
                continue
            cost_in = shares * tr.entry * bps + fee
            cash -= shares * tr.entry + cost_in
            open_pos[tr.Index] = {
                "ticker": tr.ticker, "sector": tr.sector, "shares": shares,
                "entry": tr.entry, "exit": tr.exit, "exit_date": tr.exit_date, "cost_in": cost_in,
            }
        # 2) Ausstiege im Tagesverlauf
        for k in [k for k, p in open_pos.items() if p["exit_date"] <= day]:
            p = open_pos.pop(k)
            cost_out = p["shares"] * p["exit"] * bps + fee
            cash += p["shares"] * p["exit"] - cost_out
            pnl = p["shares"] * (p["exit"] - p["entry"]) - p["cost_in"] - cost_out
            taken.append({"trade_id": k, "shares": p["shares"], "pnl": pnl})
        # 3) Bewertung zum Schluss
        invested = sum(p["shares"] * closes.at[day, p["ticker"]] for p in open_pos.values())
        equity = cash + invested
        equity_vals.append(equity)
        invested_vals.append(invested / equity if equity > 0 else 0)

    eq = pd.Series(equity_vals, index=calendar, name="equity")
    inv = pd.Series(invested_vals, index=calendar, name="invested")
    taken_df = pd.DataFrame(taken)
    if not taken_df.empty:
        taken_df = trades.loc[taken_df["trade_id"]].assign(
            shares=taken_df["shares"].to_numpy(), pnl=taken_df["pnl"].to_numpy()
        )
    return eq, taken_df, inv
