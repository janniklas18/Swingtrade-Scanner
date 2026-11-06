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


def run(rows, stop=95.0, target=np.nan, target_r=2.0, time_stop=20, bps=0.0, be=None):
    o, h, l, c = bars(rows)
    return simulate_trade(o, h, l, c, 1, stop, target, target_r, time_stop, bps, be)


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
    trades, _, _ = engine.generate_trades(prices, idx, uni, CFG, m, {})
    assert not trades.empty
    # keine überlappenden Trades je Aktie
    for _, g in trades.groupby("ticker"):
        g = g.sort_values("entry_date")
        assert (g["entry_date"].iloc[1:].to_numpy() > g["exit_date"].iloc[:-1].to_numpy()).all()
    # Einstieg immer nach dem Signal, Stop unter Einstieg
    assert (trades["entry_date"] > trades["signal_date"]).all()
    assert (trades["stop"] < trades["entry"]).all()
