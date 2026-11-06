"""Tests der Ausstiegslogik und der Signale (laufen ohne Internet)."""
import numpy as np
import pandas as pd
import pytest
import yaml

from scanner import data, engine, signals
from scanner.engine import simulate_trade

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config" / "strategy.yaml").read_text(encoding="utf-8"))


def bars(rows):
    """rows: Liste von (open, high, low, close)."""
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def run(rows, stop=95.0, target=np.nan, target_r=2.0, time_stop=20, bps=0.0, be=None, trail=None, atr=1.0):
    o, h, l, c = bars(rows)
    a = np.full(len(c), atr)
    return simulate_trade(o, h, l, c, 1, stop, target, target_r, time_stop, bps, be, trail, a)


SIGNAL = (100, 101, 99, 100)  # Tag 0 = Signaltag


def test_ziel_erreicht():
    res = run([SIGNAL, (100, 104, 99, 103), (103, 111, 102, 110)])
    assert res.reason == "Ziel" and res.exit == pytest.approx(110) and res.r == pytest.approx(2.0)


def test_stop_erreicht():
    res = run([SIGNAL, (100, 101, 96, 97), (97, 98, 94, 95)])
    assert res.reason == "Stop" and res.exit == 95 and res.r == pytest.approx(-1.0)


def test_kursluecke_unter_stop_wird_zur_eroeffnung_verkauft():
    res = run([SIGNAL, (100, 101, 97, 98), (90, 92, 89, 91)])
    assert res.reason == "Stop (Kurslücke)" and res.exit == 90 and res.r == pytest.approx(-2.0)


def test_stop_und_ziel_am_selben_tag_zaehlt_als_stop():
    res = run([SIGNAL, (100, 112, 94, 105)])
    assert res.reason == "Stop" and res.r == pytest.approx(-1.0)


def test_eroeffnung_unter_stop_kein_trade():
    assert run([SIGNAL, (94, 96, 93, 95)]) == "Eröffnung unter Stop"


def test_eroeffnung_ueber_festem_ziel_kein_trade():
    assert run([SIGNAL, (106, 107, 105, 106)], target=105.0) == "Eröffnung über Ziel"


def test_zeitstopp_zum_schlusskurs():
    rows = [SIGNAL] + [(100, 101, 99, 100.5)] * 5
    res = run(rows, time_stop=3)
    assert res.reason == "Zeitstopp" and res.exit_pos == 3 and res.exit == 100.5


def test_kosten_werden_abgezogen():
    res = run([SIGNAL, (100, 111, 99, 110)], bps=10)
    # Kosten: (100 + 110) * 0,1 % = 0,21 -> (10 - 0,21) / 5
    assert res.r == pytest.approx((10 - 0.21) / 5)


def test_einstand_nach_1r():
    rows = [SIGNAL, (100, 106, 99, 105.5), (105, 105, 99, 100)]
    res = run(rows, be=1.0)
    assert res.reason == "Einstand" and res.exit == 100


def test_signale_ohne_blick_in_die_zukunft():
    """Ein Signal am Tag t darf sich nicht ändern, wenn spätere Kurse wegfallen."""
    px = data.synthetic_prices("TEST.DE", "2015-01-01", "2020-12-31")
    idx = data.synthetic_prices("^GDAXI", "2015-01-01", "2020-12-31")
    full = signals.prepare(px, idx, CFG, min_turnover=0)
    cut = px.index[900]
    part = signals.prepare(px[px.index <= cut], idx[idx.index <= cut], CFG, min_turnover=0)
    cols = ["sig_breakout", "sig_pullback", "eligible", "b_stop", "p_stop", "p_target"]
    pd.testing.assert_frame_equal(full.loc[:cut, cols], part[cols], check_freq=False)


def test_pipeline_synthetisch():
    m = CFG["markets"]["de"]
    uni = data.load_universe(["dax"]).head(10)
    prices = data.load_prices(uni["ticker"].tolist() + [m["index"]], "2014-01-01", source="synthetic")
    idx = prices.pop(m["index"])
    trades, _, _ = engine.generate_trades(prices, {m["index"]: idx}, uni, CFG, m, {})
    assert not trades.empty
    # keine überlappenden Trades je Aktie
    for _, g in trades.groupby("ticker"):
        g = g.sort_values("entry_date")
        assert (g["entry_date"].iloc[1:].to_numpy() > g["exit_date"].iloc[:-1].to_numpy()).all()
    # Einstieg immer nach dem Signal, Stop unter Einstieg
    assert (trades["entry_date"] > trades["signal_date"]).all()
    assert (trades["stop"] < trades["entry"]).all()


def test_nachgezogener_stop_steigt_und_verkauft():
    # Stop 95, Trail 3 ATR (ATR = 1): nach Schluss 110 liegt der Stop bei 107
    rows = [SIGNAL, (100, 104, 99, 103), (103, 111, 102, 110), (110, 110, 106, 108)]
    res = run(rows, trail=3.0)
    assert res.reason == "Nachgezogener Stop" and res.exit == pytest.approx(107)
    assert res.r == pytest.approx(7 / 5)


def test_nachgezogener_stop_faellt_nie():
    rows = [SIGNAL, (100, 106, 99, 105), (104, 104, 100, 101), (101, 102, 101.5, 101.8)]
    res = run(rows, trail=3.0)
    # nach Schluss 105 -> Stop 102; Schluss 101 senkt ihn nicht -> Ausstieg Tag 3 zur Eröffnung? Nein:
    # Tag 2 Tief 100 <= 102 -> Ausstieg zum Stop 102
    assert res.exit == pytest.approx(102) and res.exit_pos == 2


def test_nachgezogener_stop_ohne_festes_ziel():
    rows = [SIGNAL] + [(100 + i, 101 + i, 99.5 + i, 100.8 + i) for i in range(1, 30)]
    res = run(rows, trail=3.0, time_stop=25)
    assert res.reason == "Zeitstopp" and res.r > 4 and np.isnan(res.target)


def _portfolio_with_tax(pnls_by_year):
    """Baut künstliche Trades mit festen Gewinnen je Jahr."""
    cal = pd.bdate_range("2020-01-01", "2022-12-31")
    rows, prices = [], {}
    for i, (year, gain) in enumerate(pnls_by_year):
        t = f"T{i}"
        entry_d = pd.Timestamp(f"{year}-03-02")
        exit_d = pd.Timestamp(f"{year}-03-10")
        prices[t] = pd.DataFrame({"close": 100.0}, index=cal)
        rows.append({"ticker": t, "sector": f"S{i}", "entry_date": entry_d, "exit_date": exit_d,
                     "entry": 100.0, "stop": 90.0, "exit": 100.0 + gain, "score": 0.0})
    cfg = {"backtest": {"initial_capital": 10000, "risk_per_trade": 0.1, "max_positions": 5,
                        "max_per_sector": 2, "max_position_pct": 1.0}}
    m = {"cost_per_side_bps": 0, "commission_per_trade": 0}
    return pd.DataFrame(rows), prices, cal, cfg, m


def test_steuer_mit_verlustvortrag():
    # 2020: 100 Stück x -10 = -1.000 (Depot 9.000); 2021: 90 Stück x +20 = +1.800
    # -> steuerpflichtig nach Verlustvortrag: 800
    trades, prices, cal, cfg, m = _portfolio_with_tax([(2020, -10.0), (2021, 20.0)])
    eq0, _, _ = engine.simulate_portfolio(trades, prices, cal, cfg, m)
    eq1, _, _ = engine.simulate_portfolio(trades, prices, cal, cfg, m, tax_rate=0.25)
    assert eq0.iloc[-1] == pytest.approx(10800)
    assert eq1.iloc[-1] == pytest.approx(10800 - 0.25 * 800)


def test_etf_steuer_mit_teilfreistellung():
    v = pd.Series([100.0, 150.0, 200.0])
    after = engine.etf_after_tax(v, 0.26375, 0.30)
    assert after.iloc[-1] == pytest.approx(200 - 100 * 0.7 * 0.26375)


def test_regime_index_je_boersenplatz():
    m = CFG["markets"]["eu"]
    assert engine.regime_index_for("NESN.SW", m) == "^SSMI"
    assert engine.regime_index_for("MC.PA", m) == "^FCHI"
    assert engine.regime_index_for("ASML.AS", m) == "^AEX"
    assert engine.regime_index_for("SAP.DE", CFG["markets"]["de"]) == "^GDAXI"


def test_rang_ohne_blick_in_die_zukunft():
    prices = {t: data.synthetic_prices(t, "2015-01-01", "2019-12-31") for t in ("A.DE", "B.DE", "C.DE")}
    full = engine.rs_rank_panel(prices, 126)
    cut = full.index[600]
    part = engine.rs_rank_panel({t: df[df.index <= cut] for t, df in prices.items()}, 126)
    pd.testing.assert_frame_equal(full.loc[:cut], part, check_freq=False)
