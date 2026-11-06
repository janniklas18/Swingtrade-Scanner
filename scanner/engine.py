"""Backtest-Engine: einzelne Trades, Zufalls-Benchmark und Portfolio.

Annahmen (bewusst vorsichtig):
- Einstieg zur Eröffnung des Tages nach dem Signal.
- Eröffnet die Aktie unter dem Stop oder über dem Ziel, wird der Trade nicht
  eingegangen.
- Öffnet die Aktie während des Trades unter dem Stop (Kurslücke), wird zur
  Eröffnung verkauft, nicht zum Stopkurs.
- Werden Stop und Ziel am selben Tag berührt, zählt der Stop.
- Nachgezogener Stop (optional): Er wird nach jedem Schlusskurs neu gesetzt und
  gilt ab dem Folgetag. Er steigt nur, er fällt nie.
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
    trail_atr: float | None = None,
    atr: np.ndarray | None = None,
) -> TradeResult | str:
    """Simuliert einen Trade. Gibt bei ungültigem Einstieg einen Grund zurück."""
    n = len(c)
    if entry_pos >= n:
        return "keine Daten"
    entry = o[entry_pos]
    if entry <= stop:
        return "Eröffnung unter Stop"
    risk = entry - stop
    if trail_atr is not None:
        target = np.inf
    elif np.isnan(target):
        target = entry + target_r * risk
    if entry >= target:
        return "Eröffnung über Ziel"

    cur_stop = stop
    highest_close = entry
    last = min(entry_pos + time_stop - 1, n - 1)
    exit_px, reason, exit_pos = np.nan, "", last

    def stop_reason() -> str:
        if cur_stop >= entry and breakeven_after_r and trail_atr is None:
            return "Einstand"
        return "Nachgezogener Stop" if cur_stop > stop else "Stop"

    mfe, mae = 0.0, 0.0
    for d in range(entry_pos, last + 1):
        if d > entry_pos:
            if o[d] <= cur_stop:
                exit_px, exit_pos = o[d], d
                reason = stop_reason() + " (Kurslücke)"
                break
            if o[d] >= target:
                exit_px, reason, exit_pos = o[d], "Ziel (Kurslücke)", d
                break
        mfe = max(mfe, (h[d] - entry) / risk)
        mae = min(mae, (l[d] - entry) / risk)
        if l[d] <= cur_stop:  # bei Stop und Ziel am selben Tag: Stop (vorsichtig)
            exit_px, exit_pos, reason = cur_stop, d, stop_reason()
            break
        if h[d] >= target:
            exit_px, reason, exit_pos = target, "Ziel", d
            break
        if d == last:
            exit_px, exit_pos = c[d], d
            reason = "Zeitstopp" if last == entry_pos + time_stop - 1 else "Datenende"
            break
        # Stop für den nächsten Tag anpassen
        if breakeven_after_r and c[d] >= entry + breakeven_after_r * risk:
            cur_stop = max(cur_stop, entry)
        if trail_atr is not None:
            highest_close = max(highest_close, c[d])
            cur_stop = max(cur_stop, highest_close - trail_atr * atr[d])

    cost = (entry + exit_px) * cost_bps / 1e4
    return TradeResult(
        entry_pos=entry_pos,
        exit_pos=exit_pos,
        entry=entry,
        stop=stop,
        target=target if np.isfinite(target) else np.nan,
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


def regime_index_for(ticker: str, market: dict) -> str:
    """Index für das Marktumfeld: je Börsenplatz (Testmarkt) oder einer für alle."""
    for suffix, idx in (market.get("regime_index") or {}).items():
        if ticker.endswith(suffix):
            return idx
    return market["index"]


def regime_tickers(market: dict) -> list[str]:
    return sorted({market["index"], *(market.get("regime_index") or {}).values()})


def rs_rank_panel(prices: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """Rang der Kursentwicklung über `lookback` Tage im ganzen Universum, je Tag
    (0 = schwächste, 1 = stärkste Aktie). Nur Werte bis zum jeweiligen Tag."""
    closes = pd.DataFrame({t: df["close"] for t, df in prices.items() if not t.startswith("^")})
    ret = closes.ffill(limit=5).pct_change(lookback, fill_method=None)
    return ret.rank(axis=1, pct=True)


def _exit_args(cfg: dict) -> dict:
    ex = cfg["exits"]
    return {
        "time_stop": ex["time_stop_days"],
        "breakeven_after_r": ex.get("breakeven_after_r"),
        "trail_atr": ex.get("trail_atr"),
    }


def generate_trades(
    prices: dict[str, pd.DataFrame],
    indices: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    cfg: dict,
    market: dict,
    earnings: dict[str, list[pd.Timestamp]],
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    """Alle Trades aller Signale (je Aktie ohne Überlappung).
    Gibt Trades, Zähler für verworfene Signale und die aufbereiteten Daten zurück."""
    ex = cfg["exits"]
    xa = _exit_args(cfg)
    use_blackout = cfg["filters"].get("earnings_blackout", True)
    sectors = universe.set_index("ticker")["sector"].to_dict()
    names = universe.set_index("ticker")["name"].to_dict()
    ranks = None
    if cfg["filters"].get("rs_rank_min") is not None:
        ranks = rs_rank_panel(prices, cfg["filters"].get("rs_rank_lookback", 126))
    skipped: dict[str, int] = {}
    rows = []
    prepared: dict[str, pd.DataFrame] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for t, df in prices.items():
        if t.startswith("^") or len(df) < 260:
            continue
        idx_df = indices[regime_index_for(t, market)]
        d = sig.prepare(
            df, idx_df, cfg, market["min_turnover"], None if ranks is None else ranks[t]
        )
        prepared[t] = d
        o, h, l, c, a = (d[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
        idx = d.index
        free_from = 0
        for s in sig.signal_rows(d, cfg).itertuples():
            if s.pos < free_from or s.pos + 1 >= len(d):
                continue
            entry_pos = s.pos + 1
            window_end = idx[min(entry_pos + ex["time_stop_days"] - 1, len(d) - 1)]
            if use_blackout and _has_earnings(earnings.get(t), idx[entry_pos], window_end):
                skip("Quartalszahlen in Haltedauer")
                continue
            res = simulate_trade(
                o, h, l, c, entry_pos, s.stop, s.target, s.target_r,
                xa["time_stop"], market["cost_per_side_bps"], xa["breakeven_after_r"],
                xa["trail_atr"], a,
            )
            if isinstance(res, str):
                skip(res)
                continue
            free_from = res.exit_pos
            if res.reason == "Datenende":  # Trade noch offen, nicht werten
                skip("noch offen am Datenende")
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
    gleiches Risiko in ATR, gleiches Ziel bzw. gleicher nachgezogener Stop,
    gleiche Ausstiegsregeln). Misst damit, ob das Einstiegssignal mehr bringt
    als der Filter allein."""
    rng = np.random.default_rng(seed)
    xa = _exit_args(cfg)
    arrays, pools = {}, {}
    for t in trades["ticker"].unique():
        d = prepared[t]
        arrays[t] = tuple(d[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
        p = np.flatnonzero(d["eligible"].to_numpy()[:-1])
        is_mask = d.index[p] <= split_date
        pools[t] = (p[is_mask], p[~is_mask])

    tickers = trades["ticker"].to_numpy()
    periods = (trades["signal_date"] > split_date).to_numpy().astype(int)
    risk_atr = trades["risk_atr"].to_numpy()
    crv = trades["plan_crv"].to_numpy()
    out = []
    for run in range(runs):
        rs = {0: [], 1: []}
        for t, per, k, cr in zip(tickers, periods, risk_atr, crv):
            pool = pools[t][per]
            if len(pool) == 0:
                continue
            o, h, l, c, a = arrays[t]
            p = int(rng.choice(pool))
            res = simulate_trade(
                o, h, l, c, p + 1, c[p] - k * a[p], np.nan, cr, xa["time_stop"],
                market["cost_per_side_bps"], xa["breakeven_after_r"], xa["trail_atr"], a,
            )
            if not isinstance(res, str):
                rs[per].append(res.r)
        out.append(
            {
                "run": run,
                "is_mean_r": np.mean(rs[0]) if rs[0] else np.nan,
                "oos_mean_r": np.mean(rs[1]) if rs[1] else np.nan,
                "all_mean_r": np.mean(rs[0] + rs[1]) if rs[0] or rs[1] else np.nan,
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
    tax_rate: float | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Spielt die Trades mit begrenztem Kapital durch.
    Rangfolge bei mehreren Signalen am selben Tag: relative Stärke (rs63).
    Mit tax_rate wird am Jahresende Abgeltungsteuer auf die realisierten
    Gewinne des Jahres gezahlt; Verluste werden ins Folgejahr vorgetragen.
    Gibt Equity-Kurve, ausgeführte Trades und investierten Anteil zurück."""
    bt = cfg["backtest"]
    bps, fee = market["cost_per_side_bps"] / 1e4, market["commission_per_trade"]
    cash = float(bt["initial_capital"])
    if trades.empty:
        eq = pd.Series(cash, index=calendar, name="equity")
        return eq, pd.DataFrame(), pd.Series(0.0, index=calendar)
    closes = (
        pd.DataFrame({t: prices[t]["close"] for t in trades["ticker"].unique()})
        .reindex(calendar)
        .ffill()
    )
    by_entry = {k: g.sort_values("score", ascending=False) for k, g in trades.groupby("entry_date")}
    open_pos: dict[int, dict] = {}
    taken = []
    equity_vals, invested_vals = [], []
    equity = cash
    realized_year, loss_carry = 0.0, 0.0
    prev_year = calendar[0].year

    def pay_tax() -> float:
        nonlocal realized_year, loss_carry
        taxable = realized_year + loss_carry
        realized_year = 0.0
        if taxable > 0:
            loss_carry = 0.0
            return taxable * tax_rate
        loss_carry = taxable
        return 0.0

    for i, day in enumerate(calendar):
        if tax_rate and day.year != prev_year:
            cash -= pay_tax()
        prev_year = day.year
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
            realized_year += pnl
            taken.append({"trade_id": k, "shares": p["shares"], "pnl": pnl})
        # 3) Steuer auf das letzte (angebrochene) Jahr am Ende des Zeitraums
        if tax_rate and i == len(calendar) - 1:
            cash -= pay_tax()
        # 4) Bewertung zum Schluss
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


def etf_after_tax(values: pd.Series, tax_rate: float, exemption: float) -> pd.Series:
    """Wert eines ETF-Depots nach Steuern, wenn man an jedem Tag verkaufen würde:
    Steuer erst beim Verkauf, mit Teilfreistellung. (Vorabpauschale und
    Sparer-Pauschbetrag sind nicht berücksichtigt.)"""
    gain = (values - values.iloc[0]).clip(lower=0)
    return values - gain * (1 - exemption) * tax_rate
