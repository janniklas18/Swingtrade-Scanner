"""Backtest starten.

Beispiele:
    python run_backtest.py --market de
    python run_backtest.py --market us --start 2015-01-01
    python run_backtest.py --market de --source synthetic   # Test ohne Internet
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scanner import data, engine, metrics, report

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("backtest")


def run_market(key: str, cfg: dict, source: str, start: str, end: str | None, out_dir: Path) -> dict:
    m = cfg["markets"][key]
    bt = cfg["backtest"]
    split = pd.Timestamp(bt["in_sample_end"])

    universe = data.load_universe(m["universes"])
    tickers = universe["ticker"].tolist()
    log.info("%s: %d Titel im Universum", m["label"], len(tickers))

    prices = data.load_prices(tickers + [m["index"]], start, source=source, end=end)
    if m["index"] not in prices:
        raise RuntimeError(f"Keine Indexdaten für {m['index']} – Abbruch.")
    index_df = prices.pop(m["index"])
    earnings = data.load_earnings_dates(list(prices), source=source)
    log.info("Kursdaten: %d Titel, Termine der Quartalszahlen: %d Titel", len(prices), len(earnings))

    trades, skipped, prepared = engine.generate_trades(prices, index_df, universe, cfg, m, earnings)
    log.info("%d Trades erzeugt", len(trades))
    if trades.empty:
        raise RuntimeError("Keine Trades – Regeln zu streng oder keine Daten.")

    is_mask = trades["signal_date"] <= split
    stats_all = metrics.trade_stats(trades["r"], bt["bootstrap_runs"])
    stats_is = metrics.trade_stats(trades.loc[is_mask, "r"], bt["bootstrap_runs"])
    stats_oos = metrics.trade_stats(trades.loc[~is_mask, "r"], bt["bootstrap_runs"])

    rnd = engine.random_benchmark(trades, prepared, cfg, m, split, runs=bt["random_benchmark_runs"])

    calendar = index_df.index.union(
        pd.DatetimeIndex(pd.concat([trades["entry_date"], trades["exit_date"]]).unique())
    ).unique().sort_values()
    calendar = calendar[calendar >= trades["entry_date"].min() - pd.Timedelta(days=5)]
    eq, taken, invested = engine.simulate_portfolio(trades, prices, calendar, cfg, m)
    bench = index_df["close"].reindex(calendar).ffill()
    bench = bench / bench.iloc[0] * bt["initial_capital"]

    def split_stats(series):
        return {
            "all": metrics.equity_stats(series),
            "is": metrics.equity_stats(series[series.index <= split]),
            "oos": metrics.equity_stats(series[series.index > split]),
        }

    trades = trades.assign(year=trades["signal_date"].dt.year)
    ctx = {
        "cfg": cfg,
        "market": m,
        "source": source,
        "start": calendar[0],
        "end": calendar[-1],
        "split": split,
        "trades": trades,
        "skipped": skipped,
        "random": rnd,
        "stats_all": stats_all,
        "stats_is": stats_is,
        "stats_oos": stats_oos,
        "by_year": metrics.by_group(trades, "year"),
        "by_setup": metrics.by_group(trades, "setup"),
        "by_sector": metrics.by_group(trades, "sector"),
        "portfolio": {
            "equity": eq,
            "bench": bench,
            "invested": invested,
            "n_taken": len(taken),
            "stats": split_stats(eq),
            "bench_stats": split_stats(bench),
        },
        "coverage": {
            "requested": len(tickers),
            "with_data": len(prices),
            "missing": sorted(set(tickers) - set(prices)),
            "with_earnings": len(earnings),
        },
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = "" if source == "yahoo" else f"_{source}"
    base = f"{stamp}_{key}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base}.html").write_text(report.build_report(ctx), encoding="utf-8")
    trades.to_csv(out_dir / f"{base}_trades.csv", index=False)

    rand_oos = rnd["oos_mean_r"].dropna()
    color, title, _ = report.verdict(stats_oos, rand_oos)
    summary = {
        "file": f"{base}.html",
        "label": m["label"],
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "source": source,
        "n_oos": int(stats_oos.get("n", 0)),
        "exp_oos": report._r(stats_oos.get("expectancy")),
        "verdict": title,
        "stats": {"all": stats_all, "is": stats_is, "oos": stats_oos},
        "random_oos_mean": float(rand_oos.mean()) if len(rand_oos) else None,
        "portfolio": ctx["portfolio"]["stats"],
        "config": cfg,
    }
    (out_dir / f"{base}_summary.json").write_text(
        json.dumps(summary, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Report: %s (%s)", out_dir / f"{base}.html", title)
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def update_index(out_dir: Path) -> None:
    entries = []
    for p in sorted(out_dir.glob("*_summary.json"), reverse=True):
        try:
            entries.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    (out_dir / "index.html").write_text(report.build_index(entries), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest der Swing-Regeln")
    ap.add_argument("--market", choices=["de", "us", "both"], default="de")
    ap.add_argument("--source", choices=["yahoo", "stooq", "synthetic"], default="yahoo")
    ap.add_argument("--start", default=None, help="Beginn, z. B. 2012-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.yaml"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "backtests"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    start = args.start or cfg["backtest"]["start"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
        raise SystemExit("Startdatum bitte als JJJJ-MM-TT angeben")
    out_dir = Path(args.out)
    markets = ["de", "us"] if args.market == "both" else [args.market]
    for key in markets:
        run_market(key, cfg, args.source, start, args.end, out_dir)
    update_index(out_dir)


if __name__ == "__main__":
    main()
