"""Backtest starten.

Modi:
    backtest   Regeln aus strategy.yaml (oder --variant) auf einem Markt testen
    varianten  alle Varianten vergleichen – NUR In-Sample (bis in_sample_end)
    finaltest  gewählte Variante auf dem frischen Testmarkt (eu) und auf DE/US

Beispiele:
    python run_backtest.py --market de
    python run_backtest.py --mode varianten --market both
    python run_backtest.py --mode finaltest
    python run_backtest.py --market de --source synthetic   # Test ohne Internet
"""
from __future__ import annotations

import argparse
import copy
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
CHOICE_FILE = "variante_gewaehlt.json"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def variant_cfg(cfg: dict, name: str) -> dict:
    if name not in cfg["variants"]:
        raise SystemExit(f"Unbekannte Variante: {name}. Möglich: {', '.join(cfg['variants'])}")
    return deep_merge(cfg, cfg["variants"][name]["overrides"])


def load_market(key: str, cfg: dict, source: str, start: str, end: str | None) -> dict:
    m = cfg["markets"][key]
    universe = data.load_universe(m["universes"])
    tickers = universe["ticker"].tolist()
    extra = engine.regime_tickers(m) + [m["benchmark"]]
    log.info("%s: %d Titel im Universum", m["label"], len(tickers))
    prices = data.load_prices(tickers + extra, start, source=source, end=end)
    indices = {}
    for i in engine.regime_tickers(m):
        if i not in prices:
            raise RuntimeError(f"Keine Indexdaten für {i} – Abbruch.")
        indices[i] = prices.pop(i)
    bench = prices.pop(m["benchmark"], None)
    bench_label = m["benchmark_label"]
    if bench is None:
        log.warning("Keine Daten für %s, nutze den Index als Vergleich", m["benchmark"])
        bench, bench_label = indices[m["index"]], m["index_label"]
    earnings = data.load_earnings_dates(list(prices), source=source)
    log.info("Kursdaten: %d Titel, Termine der Quartalszahlen: %d Titel", len(prices), len(earnings))
    return {
        "key": key, "m": m, "universe": universe, "tickers": tickers, "prices": prices,
        "indices": indices, "bench": bench["close"], "bench_label": bench_label, "earnings": earnings,
    }


def evaluate(cfg: dict, md: dict, is_only: bool = False) -> dict:
    """Trades, Kennzahlen, Zufalls-Benchmark und Portfolio (vor/nach Steuern)."""
    m, bt, tax = md["m"], cfg["backtest"], cfg["tax"]
    split = pd.Timestamp(bt["in_sample_end"])
    trades, skipped, prepared = engine.generate_trades(
        md["prices"], md["indices"], md["universe"], cfg, m, md["earnings"]
    )
    if trades.empty:
        raise RuntimeError("Keine Trades – Regeln zu streng oder keine Daten.")
    if is_only:
        trades = trades[trades["signal_date"] <= split].reset_index(drop=True)
    trades = trades.assign(year=trades["signal_date"].dt.year)
    is_mask = trades["signal_date"] <= split

    stats = {
        "all": metrics.trade_stats(trades["r"], bt["bootstrap_runs"]),
        "is": metrics.trade_stats(trades.loc[is_mask, "r"], bt["bootstrap_runs"]),
        "oos": metrics.trade_stats(trades.loc[~is_mask, "r"], bt["bootstrap_runs"]),
    }
    rnd = engine.random_benchmark(trades, prepared, cfg, m, split, runs=bt["random_benchmark_runs"])

    dates = pd.concat([trades["entry_date"], trades["exit_date"]]).unique()
    calendar = md["bench"].index.union(pd.DatetimeIndex(dates)).unique().sort_values()
    calendar = calendar[calendar >= trades["entry_date"].min() - pd.Timedelta(days=5)]
    if is_only:
        calendar = calendar[calendar <= split]
    eq, taken, invested = engine.simulate_portfolio(trades, md["prices"], calendar, cfg, m)
    eq_tax, _, _ = engine.simulate_portfolio(trades, md["prices"], calendar, cfg, m, tax["rate"])
    bench = md["bench"].reindex(calendar).ffill().bfill()
    bench = bench / bench.iloc[0] * bt["initial_capital"]

    def periods(series: pd.Series, after_tax_etf: bool = False) -> dict:
        out = {}
        for name, part in (
            ("all", series),
            ("is", series[series.index <= split]),
            ("oos", series[series.index > split]),
        ):
            if after_tax_etf and len(part) > 1:
                part = engine.etf_after_tax(part, tax["rate"], tax["etf_partial_exemption"])
            out[name] = metrics.equity_stats(part)
        return out

    return {
        "cfg": cfg, "split": split, "trades": trades, "skipped": skipped, "random": rnd,
        "stats": stats, "calendar": calendar,
        "portfolio": {
            "equity": eq, "equity_tax": eq_tax, "bench": bench, "invested": invested,
            "n_taken": len(taken),
            "stats": periods(eq), "stats_tax": periods(eq_tax),
            "bench_stats": periods(bench), "bench_stats_tax": periods(bench, after_tax_etf=True),
        },
    }


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Modus: backtest / finaltest (ein Report je Markt)
# ---------------------------------------------------------------------------
def run_market(md: dict, cfg: dict, source: str, out_dir: Path, variant: str | None, tag: str) -> dict:
    m = md["m"]
    holdout = bool(m.get("holdout"))
    ev = evaluate(cfg, md)
    basis = "all" if holdout else "oos"
    rnd_col = "all_mean_r" if holdout else "oos_mean_r"
    trades = ev["trades"]
    ctx = {
        **ev,
        "market": m,
        "source": source,
        "start": ev["calendar"][0],
        "end": ev["calendar"][-1],
        "holdout": holdout,
        "variant_label": cfg["variants"][variant]["label"] if variant else None,
        "bench_label": md["bench_label"],
        "stats_all": ev["stats"]["all"],
        "stats_is": ev["stats"]["is"],
        "stats_oos": ev["stats"]["oos"],
        "by_year": metrics.by_group(trades, "year"),
        "by_setup": metrics.by_group(trades, "setup"),
        "by_sector": metrics.by_group(trades, "sector"),
        "coverage": {
            "requested": len(md["tickers"]),
            "with_data": len(md["prices"]),
            "missing": sorted(set(md["tickers"]) - set(md["prices"])),
            "with_earnings": len(md["earnings"]),
        },
    }
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    src = "" if source == "yahoo" else f"_{source}"
    base = f"{stamp}_{md['key']}{tag}{src}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base}.html").write_text(report.build_report(ctx), encoding="utf-8")
    trades.to_csv(out_dir / f"{base}_trades.csv", index=False)

    rand = ev["random"][rnd_col].dropna()
    _, title, _ = report.verdict(ev["stats"][basis], rand, holdout)
    label = m["label"] + (f" – {ctx['variant_label']}" if variant else "")
    summary = {
        "file": f"{base}.html",
        "label": ("Finaltest: " if tag == "_final" else "") + label,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "source": source,
        "n_oos": int(ev["stats"][basis].get("n", 0)),
        "exp_oos": report._r(ev["stats"][basis].get("expectancy")),
        "verdict": title,
        "stats": ev["stats"],
        "random_mean": float(rand.mean()) if len(rand) else None,
        "portfolio": ev["portfolio"]["stats"],
        "portfolio_after_tax": ev["portfolio"]["stats_tax"],
        "etf_after_tax": ev["portfolio"]["bench_stats_tax"],
        "variant": variant,
        "config": cfg,
    }
    write_json(out_dir / f"{base}_summary.json", summary)
    log.info("Report: %s (%s)", out_dir / f"{base}.html", title)
    return summary


# ---------------------------------------------------------------------------
# Modus: varianten (nur In-Sample)
# ---------------------------------------------------------------------------
def run_variants(cfg: dict, markets: list[str], source: str, start: str, end: str | None, out_dir: Path) -> dict:
    rows = []
    for key in markets:
        md = load_market(key, cfg, source, start, end)
        for name, v in cfg["variants"].items():
            vcfg = variant_cfg(cfg, name)
            log.info("%s – %s", key, v["label"])
            ev = evaluate(vcfg, md, is_only=True)
            s = ev["stats"]["is"]
            rnd = ev["random"]["is_mean_r"].dropna()
            p, pt, bt = ev["portfolio"]["stats"]["is"], ev["portfolio"]["stats_tax"]["is"], ev["portfolio"]["bench_stats_tax"]["is"]
            rows.append({
                "market": key, "market_label": cfg["markets"][key]["label"],
                "bench_label": md["bench_label"], "variant": name, "label": v["label"],
                "n": s.get("n", 0), "hit_rate": s.get("hit_rate"), "expectancy": s.get("expectancy"),
                "ci_low": s.get("ci_low"), "ci_high": s.get("ci_high"),
                "random": float(rnd.mean()) if len(rnd) else np.nan,
                "random_q95": float(rnd.quantile(0.95)) if len(rnd) else np.nan,
                "days": float(ev["trades"]["days"].mean()),
                "cagr": p.get("cagr"), "cagr_tax": pt.get("cagr"), "max_dd": p.get("max_dd"),
                "sharpe": p.get("sharpe"), "etf_cagr_tax": bt.get("cagr"),
                "etf_cagr": ev["portfolio"]["bench_stats"]["is"].get("cagr"),
                "n_taken": ev["portfolio"]["n_taken"],
            })
    df = pd.DataFrame(rows)

    # Vorab festgelegte Auswahlregel
    agg = df.groupby("variant").agg(
        min_exp=("expectancy", "min"), mean_cagr_tax=("cagr_tax", "mean"),
        mean_etf_tax=("etf_cagr_tax", "mean"),
        beats_random=("expectancy", lambda s: bool((s > df.loc[s.index, "random"]).all())),
    )
    ok = agg[agg["min_exp"] > 0]
    if ok.empty:
        chosen, eligible = agg["mean_cagr_tax"].idxmax(), False
    else:
        chosen, eligible = ok["mean_cagr_tax"].idxmax(), True
    choice = {
        "variant": chosen,
        "label": cfg["variants"][chosen]["label"],
        "eligible": eligible,
        "beats_random": bool(agg.loc[chosen, "beats_random"]),
        "beats_etf_after_tax": bool(agg.loc[chosen, "mean_cagr_tax"] > agg.loc[chosen, "mean_etf_tax"]),
        "markets": markets,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "source": source,
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    src = "" if source == "yahoo" else f"_{source}"
    base = f"{stamp}_varianten{src}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base}.html").write_text(
        report.build_variants_report(df, choice, cfg, source), encoding="utf-8"
    )
    df.to_csv(out_dir / f"{base}.csv", index=False)
    if source == "yahoo":
        write_json(out_dir / CHOICE_FILE, choice)
    best = df[df["variant"] == chosen]
    write_json(out_dir / f"{base}_summary.json", {
        "file": f"{base}.html",
        "label": "Variantenvergleich (nur In-Sample) – " + ", ".join(m.upper() for m in markets),
        "date": choice["date"],
        "source": source,
        "n_oos": int(best["n"].sum()),
        "exp_oos": report._r(best["expectancy"].mean()),
        "verdict": "Gewählt: " + choice["label"] + ("" if eligible else " (Bedingung nicht erfüllt)"),
    })
    log.info("Gewählt: %s (Bedingung erfüllt: %s)", choice["label"], eligible)
    return choice


# ---------------------------------------------------------------------------
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
    ap.add_argument("--mode", choices=["backtest", "varianten", "finaltest"], default="backtest")
    ap.add_argument("--market", choices=["de", "us", "both", "eu"], default="de")
    ap.add_argument("--variant", default=None, help="Name einer Variante oder 'auto' (gewählte)")
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

    if args.mode == "varianten":
        run_variants(cfg, [m for m in markets if m != "eu"] or ["de", "us"], args.source, start, args.end, out_dir)
    else:
        variant = args.variant if args.variant not in (None, "", "keine") else None
        if args.mode == "finaltest":
            if variant in (None, "auto"):
                path = out_dir / CHOICE_FILE
                if not path.exists():
                    raise SystemExit("Noch keine Variante gewählt – zuerst den Modus 'varianten' laufen lassen.")
                variant = json.loads(path.read_text(encoding="utf-8"))["variant"]
            markets = ["eu", "de", "us"]
        elif variant == "auto":
            variant = json.loads((out_dir / CHOICE_FILE).read_text(encoding="utf-8"))["variant"]
        run_cfg = variant_cfg(cfg, variant) if variant else cfg
        tag = "_final" if args.mode == "finaltest" else ""
        for key in markets:
            md = load_market(key, run_cfg, args.source, start, args.end)
            run_market(md, run_cfg, args.source, out_dir, variant, tag)
    update_index(out_dir)


if __name__ == "__main__":
    main()
