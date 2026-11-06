"""HTML-Report des Backtests (eine Datei, ohne externe Abhängigkeiten)."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, GREY, GREEN, RED = "#2563eb", "#9ca3af", "#16a34a", "#dc2626"


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------
def _num(x, digits=2, suffix=""):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "–" if not (isinstance(x, float) and np.isinf(x)) else "∞"
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".") + suffix


def _pct(x, digits=1):
    return "–" if x is None or pd.isna(x) else _num(x * 100, digits, " %")


def _r(x):
    return "–" if x is None or pd.isna(x) else (("+" if x > 0 else "") + _num(x, 2, " R"))


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#4b5563", labelsize=9)
    ax.grid(axis="y", color="#f3f4f6")


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------
def chart_equity(eq: pd.Series, bench: pd.Series, split: pd.Timestamp, bench_label: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(bench.index, bench.values, color=GREY, lw=1.4, label=f"{bench_label} (Kaufen & Halten)")
    ax.plot(eq.index, eq.values, color=BLUE, lw=1.8, label="Strategie")
    ax.axvline(split, color="#6b7280", ls="--", lw=1)
    ax.text(split, ax.get_ylim()[1], "  Out-of-Sample →", va="top", fontsize=8, color="#4b5563")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
    ax.set_ylabel("Depotwert (log. Skala)", fontsize=9, color="#4b5563")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style_ax(ax)
    return _img(fig)


def chart_by_year(by_year: pd.DataFrame, split_year: int) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    colors = [GREEN if v > 0 else RED for v in by_year["expectancy"]]
    alphas = [0.45 if y <= split_year else 0.85 for y in by_year["year"]]
    for x, (v, col, a) in enumerate(zip(by_year["expectancy"], colors, alphas)):
        ax.bar(x, v, color=col, alpha=a, width=0.65)
    ax.axhline(0, color="#6b7280", lw=1)
    ax.set_xticks(range(len(by_year)))
    ax.set_xticklabels([f"{y}\n{n} Tr." for y, n in zip(by_year["year"], by_year["n"])], fontsize=8)
    ax.set_ylabel("Ø Ergebnis je Trade (R)", fontsize=9, color="#4b5563")
    _style_ax(ax)
    return _img(fig)


def chart_r_hist(r: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    top = float(np.clip(np.ceil(r.quantile(0.99)), 4, 10))
    bins = np.arange(-2.5, top + 0.01, 0.25)
    vals = r.clip(-2.49, top - 0.01)
    neg, pos = vals[vals <= 0], vals[vals > 0]
    ax.hist(neg, bins=bins, color=RED, alpha=0.6)
    ax.hist(pos, bins=bins, color=GREEN, alpha=0.6)
    ax.axvline(r.mean(), color=BLUE, lw=1.5)
    ax.text(r.mean(), ax.get_ylim()[1] * 0.95, f"  Ø {_r(r.mean())}", color=BLUE, fontsize=9, va="top")
    ax.set_xlabel("Ergebnis je Trade in R (Vielfache des Risikos)", fontsize=9, color="#4b5563")
    _style_ax(ax)
    return _img(fig)


def chart_random(real: float, rand: pd.Series, period: str = "im Out-of-Sample") -> str:
    fig, ax = plt.subplots(figsize=(9, 1.3))
    ax.scatter(rand, np.zeros(len(rand)), color=GREY, s=30, alpha=0.7, label="Zufällige Einstiege (je Lauf)")
    ax.scatter([real], [0], color=BLUE, s=90, marker="D", label="Echte Signale", zorder=3)
    ax.axvline(0, color="#6b7280", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"Ø Ergebnis je Trade {period} (R)", fontsize=9, color="#4b5563")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    _style_ax(ax)
    ax.spines["left"].set_visible(False)
    return _img(fig)


# ---------------------------------------------------------------------------
# Tabellen
# ---------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]], first_left: bool = True) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="{"l" if (i == 0 and first_left) else ""}">{c}</td>' for i, c in enumerate(r)
        ) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _stats_rows(cols: list[dict]) -> list[list[str]]:
    def g(s, k):
        return s.get(k, np.nan) if s.get("n", 0) else np.nan

    return [
        ["Trades"] + [str(s.get("n", 0)) for s in cols],
        ["Trefferquote"] + [_pct(g(s, "hit_rate")) for s in cols],
        ["Ø Gewinn"] + [_r(g(s, "avg_win")) for s in cols],
        ["Ø Verlust"] + [_r(g(s, "avg_loss")) for s in cols],
        ["<b>Erwartung je Trade</b>"] + [f"<b>{_r(g(s, 'expectancy'))}</b>" for s in cols],
        ["95 %-Intervall"] + [
            f"{_r(g(s, 'ci_low'))} bis {_r(g(s, 'ci_high'))}" if s.get("n", 0) > 1 else "–" for s in cols
        ],
        ["Profit-Faktor"] + [_num(g(s, "profit_factor")) for s in cols],
    ]


# ---------------------------------------------------------------------------
# Urteil
# ---------------------------------------------------------------------------
def verdict(oos: dict, rand_oos: pd.Series, holdout: bool = False) -> tuple[str, str, str]:
    """Ampel auf Basis des Prüfzeitraums (beim frischen Testmarkt: alle Jahre)."""
    where = "Auf dem frischen Testmarkt" if holdout else "Im Out-of-Sample"
    n = oos.get("n", 0)
    if n < 30:
        return ("grey", "Zu wenige Trades für ein Urteil",
                f"{where} gab es nur {n} Trades. Ab etwa 30, besser 100, wird das Ergebnis aussagekräftig.")
    exp, lo = oos["expectancy"], oos["ci_low"]
    q95 = rand_oos.quantile(0.95) if len(rand_oos.dropna()) else np.nan
    beats = not np.isnan(q95) and exp > q95
    if exp <= 0:
        return ("red", "Kein Vorteil erkennbar",
                f"{where} verliert ein Trade im Schnitt {_r(exp)} nach Kosten. So nicht handeln.")
    if lo > 0 and beats:
        return ("green", "Messbarer Vorteil",
                f"Erwartung {_r(exp)} je Trade nach Kosten, das 95 %-Intervall liegt über null, und die Signale "
                f"schlagen 95 % der Zufallseinstiege. Das ist ein Hinweis auf einen echten Vorteil – keine Garantie.")
    why = []
    if lo <= 0:
        why.append("das 95 %-Intervall schließt null ein (Ergebnis könnte Zufall sein)")
    if not beats:
        why.append("zufällige Einstiege im selben Umfeld schneiden ähnlich gut ab (der Vorteil kommt eher vom Filter als vom Signal)")
    return ("yellow", "Positiv, aber nicht belegt", f"Erwartung {_r(exp)} je Trade, aber " + " und ".join(why) + ".")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
CSS = """
:root{--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;--soft:#f9fafb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}.meta{color:var(--muted);font-size:13px}
.box{border-radius:10px;padding:14px 16px;margin:18px 0;border:1px solid}
.green{background:#f0fdf4;border-color:#86efac}.yellow{background:#fefce8;border-color:#fde047}
.red{background:#fef2f2;border-color:#fca5a5}.grey{background:var(--soft);border-color:var(--line)}
.warn{background:#fff7ed;border-color:#fdba74}
.box b.t{display:block;font-size:17px;margin-bottom:2px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 10px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--soft);font-weight:600;color:#374151}td.l,th:first-child{text-align:left;white-space:normal;min-width:160px}
img{max-width:100%;height:auto;display:block;margin:6px 0}
ul{padding-left:20px}li{margin:3px 0}small,.note{color:var(--muted);font-size:13px}
"""


def rules_text(cfg: dict, m: dict) -> list[str]:
    """Die Regeln in Worten (für Reports)."""
    f, sb, sp, ex = cfg["filters"], cfg["setups"]["breakout"], cfg["setups"]["pullback"], cfg["exits"]
    trail = ex.get("trail_atr")
    regime = m["index_label"] if not m.get("regime_index") else "Landesindex"
    out = [
        f"<b>Filter:</b> {regime} über seiner {f['regime_sma']}-Tage-Linie; Aktie über SMA{f['trend_fast_sma']}, "
        f"SMA{f['trend_fast_sma']} über SMA{f['trend_slow_sma']}; ATR zwischen {_num(f['atr_pct_min'], 1)} und "
        f"{_num(f['atr_pct_max'], 1)} % vom Kurs; Ø Tagesumsatz mindestens {_num(m['min_turnover'] / 1e6, 0)} Mio. "
        f"{m['currency']}; keine Quartalszahlen in der Haltedauer (wo bekannt)."
        + (f" Nur Aktien unter den stärksten {_pct(1 - f['rs_rank_min'], 0)} des Universums "
           f"(Kursentwicklung der letzten {f.get('rs_rank_lookback', 126)} Handelstage)."
           if f.get("rs_rank_min") is not None else "")
    ]
    if sb["enabled"]:
        tgt = "kein festes Ziel" if trail else f"Ziel {_num(sb['target_r'], 1)} R"
        out.append(
            f"<b>Ausbruch:</b> Schluss über dem {sb['lookback']}-Tage-Hoch, Volumen mindestens "
            f"{_num(sb['volume_ratio_min'], 1)}× Durchschnitt, Schluss im oberen Teil der Tagesspanne. Stop "
            f"{_num(sb['stop_atr'], 1)} ATR unter dem Signalkurs, {tgt}. Kein Trade, wenn das 52-Wochen-Hoch "
            f"weniger als {_num(ex['min_crv'], 1)} R über dem Einstieg liegt."
        )
    if sp["enabled"]:
        out.append(
            f"<b>Rücksetzer:</b> RSI({sp['rsi_period']}) war in den letzten {sp['rsi_window']} Tagen unter "
            f"{sp['rsi_max']}, Kurs an der {sp['near_sma']}-Tage-Linie, dann Schluss über dem Vortageshoch. Stop "
            f"unter dem {sp['stop_lookback']}-Tage-Tief, Ziel am {sp['target_lookback']}-Tage-Hoch, nur bei "
            f"CRV ≥ {_num(ex['min_crv'], 1)}."
        )
    if trail:
        out.append(
            f"<b>Ausstieg:</b> Nachgezogener Stop {_num(trail, 1)} ATR unter dem höchsten Schlusskurs seit Einstieg "
            f"(steigt nur), oder nach {ex['time_stop_days']} Handelstagen zum Schlusskurs. Einstieg zur Eröffnung "
            "des Folgetags."
        )
    else:
        out.append(
            f"<b>Ausstieg:</b> Stop, Ziel oder nach {ex['time_stop_days']} Handelstagen zum Schlusskurs. "
            "Einstieg zur Eröffnung des Folgetags; Stop und Ziel am selben Tag zählen als Stop."
        )
    return out


def build_report(ctx: dict) -> str:
    t = ctx["trades"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    m = ctx["market"]
    rnd = ctx["random"]
    holdout = ctx.get("holdout", False)
    s_all, s_is, s_oos = ctx["stats_all"], ctx["stats_is"], ctx["stats_oos"]
    s_test = s_all if holdout else s_oos
    rand_col = "all_mean_r" if holdout else "oos_mean_r"
    rand_oos = rnd[rand_col].dropna() if not rnd.empty else pd.Series(dtype=float)
    rand_stats = {"n": int(s_test.get("n", 0)), "expectancy": rand_oos.mean() if len(rand_oos) else np.nan}
    color, title, text = verdict(s_test, rand_oos, holdout)
    bl = ctx.get("bench_label", m["index_label"])

    head = "Finaltest" if ctx.get("variant_label") else "Backtest"
    parts = [f"<h1>{head}: {html.escape(m['label'])}</h1>"]
    if ctx.get("variant_label"):
        parts.append(f"<p><b>Variante:</b> {html.escape(ctx['variant_label'])}</p>")
    parts.append(
        f'<div class="meta">Zeitraum {ctx["start"]:%d.%m.%Y} – {ctx["end"]:%d.%m.%Y} · '
        f"Out-of-Sample ab {split + pd.Timedelta(days=1):%d.%m.%Y} · Datenquelle: {html.escape(ctx['source'])} · "
        f"erstellt {datetime.now():%d.%m.%Y %H:%M}</div>"
    )
    if holdout:
        parts.append(
            '<div class="box grey"><b class="t">Frischer Testmarkt</b>Diese Aktien wurden bei der Entwicklung '
            "der Regeln nie angesehen. Deshalb zählt hier der gesamte Zeitraum als Prüfung, nicht nur die Jahre "
            "ab 2022.</div>"
        )
    if ctx["source"] == "synthetic":
        parts.append(
            '<div class="box warn"><b class="t">Testlauf mit künstlichen Kursen</b>'
            "Dieser Report prüft nur, ob die Pipeline technisch funktioniert. Die Zahlen sagen nichts über "
            "echte Märkte aus.</div>"
        )
    parts.append(f'<div class="box {color}"><b class="t">{title}</b>{text}</div>')

    # Trade-Kennzahlen
    parts.append("<h2>Ergebnis je Trade</h2>")
    parts.append(
        '<p class="note">R = Vielfaches des eingesetzten Risikos. +1 R heißt: so viel gewonnen, wie beim Stop '
        "verloren worden wäre. Alle Werte nach Spread und Slippage "
        f"({_num(m['cost_per_side_bps'] / 100, 2, ' %')} je Kauf und Verkauf).</p>"
    )
    parts.append('<div class="wrap">' + _table(
        ["", "Gesamt", f"In-Sample (bis {split:%Y})", f"Out-of-Sample (ab {split.year + 1})",
         "Zufall (gesamt)" if holdout else "Zufall (Out-of-Sample)"],
        [r[:4] + [r4] for r, r4 in zip(
            _stats_rows([s_all, s_is, s_oos]),
            [str(rand_stats["n"]), "–", "–", "–", f"<b>{_r(rand_stats['expectancy'])}</b>", "–", "–"],
        )],
    ) + "</div>")
    if len(rand_oos):
        parts.append("<h3>Echte Signale gegen zufällige Einstiege</h3>")
        parts.append(
            '<p class="note">Jeder echte Trade wird durch einen Einstieg an einem zufälligen Tag ersetzt, an dem '
            "dieselben Filter erfüllt waren (gleiche Aktie, gleiches Risiko, gleiche Ausstiegsregeln). "
            f"{len(rand_oos)} Wiederholungen. Liegt der blaue Punkt klar rechts der grauen, bringt das Signal "
            "selbst einen Mehrwert.</p>"
        )
        parts.append(chart_random(s_test.get("expectancy", np.nan), rand_oos,
                                  "gesamt" if holdout else "im Out-of-Sample"))
    if not t.empty:
        parts.append("<h3>Verteilung der Ergebnisse</h3>" + chart_r_hist(t["r"]))

    # Portfolio
    pf = ctx["portfolio"]
    parts.append("<h2>Portfolio-Simulation</h2>")
    bt = cfg["backtest"]
    parts.append(
        f'<p class="note">Start mit {_num(bt["initial_capital"], 0)} {m["currency"]}, '
        f"{_pct(bt['risk_per_trade'], 1)} Risiko je Trade, höchstens {bt['max_positions']} Positionen und "
        f"{bt['max_per_sector']} je Sektor. Bei mehreren Signalen am selben Tag haben Aktien mit höherer "
        f"relativer Stärke Vorrang. Ordergebühr {_num(m['commission_per_trade'], 2)} {m['currency']} je Order.</p>"
    )
    parts.append(chart_equity(pf["equity"], pf["bench"], split, bl))
    rows = []
    for label, key in (("Gesamt", "all"), ("In-Sample", "is"), ("Out-of-Sample", "oos")):
        s, b = pf["stats"][key], pf["bench_stats"][key]
        rows.append([
            label, _pct(s.get("cagr")), _pct(b.get("cagr")), _pct(s.get("max_dd")), _pct(b.get("max_dd")),
            _num(s.get("sharpe")), _num(b.get("sharpe")),
        ])
    parts.append('<div class="wrap">' + _table(
        ["", "Rendite p. a.", "ETF p. a.", "Max. Rückgang", "ETF Max. Rückgang",
         "Sharpe", "ETF Sharpe"], rows) + "</div>")
    parts.append(
        f'<p class="note">Ausgeführt wurden {pf["n_taken"]} von {len(t)} Signalen (Rest: Kapazität oder Sektorgrenze). '
        f"Im Schnitt waren {_pct(pf['invested'].mean(), 0)} des Kapitals investiert. Der Vergleich nutzt einen "
        f"ETF inklusive Dividenden ({html.escape(bl)}).</p>"
    )

    # Steuern
    tax = cfg["tax"]
    parts.append("<h3>Nach Steuern: Strategie gegen ETF</h3>")
    rows = []
    for label, key in (("Gesamt", "all"), ("In-Sample", "is"), ("Out-of-Sample", "oos")):
        s, st = pf["stats"][key], pf["stats_tax"][key]
        b, bt_ = pf["bench_stats"][key], pf["bench_stats_tax"][key]
        diff = st.get("cagr", np.nan) - bt_.get("cagr", np.nan)
        rows.append([label, _pct(s.get("cagr")), f"<b>{_pct(st.get('cagr'))}</b>", _pct(b.get("cagr")),
                     f"<b>{_pct(bt_.get('cagr'))}</b>",
                     ("+" if diff > 0 else "") + _num(diff * 100, 1, " %-Pkt.") if not pd.isna(diff) else "–"])
    parts.append('<div class="wrap">' + _table(
        ["", "Strategie vor Steuern", "Strategie nach Steuern", "ETF vor Steuern", "ETF nach Steuern",
         "Vorsprung nach Steuern"], rows) + "</div>")
    parts.append(
        f'<p class="note">Strategie: {_pct(tax["rate"], 3)} Abgeltungsteuer inkl. Soli auf die realisierten '
        "Gewinne jedes Jahres, Verluste werden vorgetragen. ETF: Steuer erst beim Verkauf am Ende des Zeitraums, "
        f"mit {_pct(tax['etf_partial_exemption'], 0)} Teilfreistellung. Nicht berücksichtigt: Kirchensteuer, "
        "Sparer-Pauschbetrag, Vorabpauschale und Wechselkurse.</p>"
    )

    # Aufschlüsselungen
    parts.append("<h2>Aufschlüsselung</h2>")
    by_year = ctx["by_year"]
    if not by_year.empty:
        parts.append("<h3>Nach Jahr</h3><p class=\"note\">Kräftige Balken = Out-of-Sample.</p>")
        parts.append(chart_by_year(by_year, split.year))
    for title_, df, col in (("Nach Setup", ctx["by_setup"], "setup"), ("Nach Sektor", ctx["by_sector"], "sector")):
        if df.empty:
            continue
        df = df.sort_values("n", ascending=False)
        parts.append(f"<h3>{title_}</h3>" + '<div class="wrap">' + _table(
            ["", "Trades", "Trefferquote", "Erwartung", "95 %-Intervall", "Profit-Faktor"],
            [[html.escape(str(r[col])), str(r["n"]), _pct(r["hit_rate"]), _r(r["expectancy"]),
              f"{_r(r['ci_low'])} bis {_r(r['ci_high'])}" if r["n"] > 1 else "–", _num(r["profit_factor"])]
             for _, r in df.iterrows()]) + "</div>")
    if not t.empty:
        reasons = t.groupby("reason")["r"].agg(["count", "mean"]).sort_values("count", ascending=False)
        parts.append("<h3>Ausstiegsgründe</h3>" + '<div class="wrap">' + _table(
            ["Grund", "Anzahl", "Anteil", "Ø Ergebnis"],
            [[html.escape(k), str(int(r["count"])), _pct(r["count"] / len(t)), _r(r["mean"])]
             for k, r in reasons.iterrows()]) + "</div>")
        parts.append(
            f'<p class="note">Ø Haltedauer {_num(t["days"].mean(), 1)} Handelstage. Stops durch Kurslücken '
            "werden zur Eröffnung ausgeführt – deshalb können Verluste größer als 1 R sein.</p>"
        )

    # Daten und verworfene Signale
    parts.append("<h2>Datenbasis</h2><ul>")
    cov = ctx["coverage"]
    parts.append(f"<li>{cov['with_data']} von {cov['requested']} Titeln mit Kursdaten.")
    if cov["missing"]:
        parts.append(f" Ohne Daten: {html.escape(', '.join(cov['missing'][:30]))}"
                     + (" …" if len(cov["missing"]) > 30 else ""))
    parts.append("</li>")
    parts.append(
        f"<li>Termine der Quartalszahlen verfügbar für {cov['with_earnings']} Titel. Für die übrigen konnte "
        "die Sperre vor Zahlen nicht angewendet werden – dort sind Kurslücken nach Zahlen im Ergebnis enthalten.</li>"
    )
    parts.append(
        "<li>Es werden die <b>heutigen</b> Indexmitglieder getestet. Firmen, die früher im Index waren und "
        "abgestürzt oder verschwunden sind, fehlen (Survivorship Bias). Die Ergebnisse sind dadurch eher zu gut.</li>"
    )
    if ctx["skipped"]:
        sk = "; ".join(f"{html.escape(k)}: {v}" for k, v in sorted(ctx["skipped"].items(), key=lambda x: -x[1]))
        parts.append(f"<li>Verworfene Signale: {sk}.</li>")
    parts.append("</ul>")

    # Letzte Trades
    if not t.empty:
        last = t.sort_values("entry_date", ascending=False).head(25)
        parts.append("<h2>Letzte 25 Trades</h2>" + '<div class="wrap">' + _table(
            ["Aktie", "Setup", "Einstieg", "Kurs", "Stop", "Ziel", "Ausstieg", "Grund", "Ergebnis"],
            [[f'<span title="{html.escape(r["ticker"])}">{html.escape(str(r["name"]))}</span>', r["setup"], f"{r['entry_date']:%d.%m.%Y}",
              _num(r["entry"]), _num(r["stop"]), _num(r["target"]), f"{r['exit_date']:%d.%m.%Y}",
              html.escape(r["reason"]), _r(r["r"])] for _, r in last.iterrows()]) + "</div>")

    # Regeln
    parts.append("<h2>Getestete Regeln</h2><ul>" + "".join(f"<li>{x}</li>" for x in rules_text(cfg, m)) + "</ul>")
    parts.append(
        '<p class="note">Wichtig: Die Regeln nur anhand der In-Sample-Jahre anpassen. Wer so lange an den '
        "Parametern dreht, bis auch das Out-of-Sample gut aussieht, verliert den ehrlichen Test. "
        "Kein Anlagerat; vergangene Ergebnisse sind keine Garantie.</p>"
    )

    return (
        '<!doctype html><html lang="de"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Backtest {html.escape(m['label'])}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )


def build_index(entries: list[dict]) -> str:
    rows = [[f'<a href="{html.escape(e["file"])}">{html.escape(e["label"])}</a>', e["date"],
             html.escape(e["source"]), str(e["n_oos"]), e["exp_oos"], e["verdict"]] for e in entries]
    table = _table(["Report", "Erstellt", "Daten", "Trades im Prüfzeitraum", "Erwartung", "Urteil"], rows)
    return (
        '<!doctype html><html lang="de"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Backtests</title><style>{CSS}</style></head><body><main>"
        "<h1>Backtests</h1><p class=\"meta\">Neueste zuerst.</p>"
        f'<div class="wrap">{table}</div></main></body></html>'
    )


# ---------------------------------------------------------------------------
# Variantenvergleich
# ---------------------------------------------------------------------------
MARKET_COLORS = ["#2563eb", "#f59e0b", "#10b981"]


def chart_variants(df: pd.DataFrame) -> str:
    markets = list(dict.fromkeys(df["market"]))
    variants = list(dict.fromkeys(df["variant"]))
    labels = [df.loc[df["variant"] == v, "label"].iloc[0].split(":")[0] for v in variants]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = 0.8 / len(markets)
    for j, mk in enumerate(markets):
        sub = df[df["market"] == mk].set_index("variant").reindex(variants)
        xs = np.arange(len(variants)) + (j - (len(markets) - 1) / 2) * width
        ax.bar(xs, sub["cagr_tax"] * 100, width=width * 0.9, color=MARKET_COLORS[j], alpha=0.85,
               label=f"{sub['market_label'].iloc[0]}: Strategie")
        etf = sub["etf_cagr_tax"].iloc[0] * 100
        ax.axhline(etf, color=MARKET_COLORS[j], ls="--", lw=1.2,
                   label=f"{sub['bench_label'].iloc[0]} nach Steuern ({_num(etf, 1)} %)")
        for x, v in zip(xs, sub["cagr_tax"] * 100):
            ax.text(x, v + (0.3 if v >= 0 else -0.9), _num(v, 1), ha="center", fontsize=8, color="#374151")
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"Variante {l}" for l in labels], fontsize=9)
    ax.set_ylabel("Rendite p. a. nach Steuern (%)", fontsize=9, color="#4b5563")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style_ax(ax)
    return _img(fig)


def build_variants_report(df: pd.DataFrame, choice: dict, cfg: dict, source: str) -> str:
    split = pd.Timestamp(cfg["backtest"]["in_sample_end"])
    parts = ["<h1>Variantenvergleich</h1>"]
    parts.append(
        f'<div class="meta">Nur In-Sample: {cfg["backtest"]["start"][:4]} bis {split:%d.%m.%Y} · '
        f"Datenquelle: {html.escape(source)} · erstellt {datetime.now():%d.%m.%Y %H:%M}</div>"
    )
    if source == "synthetic":
        parts.append('<div class="box warn"><b class="t">Testlauf mit künstlichen Kursen</b>'
                     "Die Zahlen sagen nichts über echte Märkte aus.</div>")

    lab = html.escape(choice["label"])
    if not choice["eligible"]:
        color, title = "red", "Keine Variante erfüllt die Bedingung"
        text = (f"Keine Variante hat in allen Märkten eine positive Erwartung je Trade. Am besten schneidet "
                f"{lab} ab. Ein Finaltest ist kaum sinnvoll – die Regeln sollten grundsätzlich überdacht werden.")
    else:
        notes = []
        if not choice["beats_random"]:
            notes.append("sie schlägt zufällige Einstiege im selben Umfeld nicht in allen Märkten")
        if not choice["beats_etf_after_tax"]:
            notes.append("sie liegt nach Steuern im Schnitt hinter dem ETF")
        color = "green" if not notes else "yellow"
        title = f"Gewählt: {lab}"
        text = "Gewählt nach der vorab festgelegten Regel (siehe unten)."
        if notes:
            text += " Vorsicht: " + " und ".join(notes) + "."
        text += (" Nächster Schritt: Finaltest auf dem frischen Testmarkt (Actions → Backtest → Modus "
                 "„finaltest“).")
    parts.append(f'<div class="box {color}"><b class="t">{title}</b>{text}</div>')

    parts.append("<h2>Rendite nach Steuern</h2>")
    parts.append('<p class="note">Balken: Portfolio-Simulation der Variante nach Abgeltungsteuer. Gestrichelt: '
                 "ETF inklusive Dividenden nach Steuern bei Verkauf am Ende.</p>")
    parts.append(chart_variants(df))

    for mk, sub in df.groupby("market", sort=False):
        parts.append(f"<h2>{html.escape(sub['market_label'].iloc[0])}</h2>")
        rows, rows2 = [], []
        for _, r in sub.iterrows():
            edge = r["expectancy"] - r["random"]
            name = html.escape(r["label"]) + (" ✓" if r["variant"] == choice["variant"] else "")
            rows.append([
                name, str(int(r["n"])), _pct(r["hit_rate"]), f"<b>{_r(r['expectancy'])}</b>",
                f"{_r(r['ci_low'])} bis {_r(r['ci_high'])}", _r(r["random"]),
                ("+" if edge > 0 else "") + _num(edge, 2, " R"), _num(r["days"], 1),
            ])
            rows2.append([
                name, _pct(r["cagr"]), f"<b>{_pct(r['cagr_tax'])}</b>", _pct(r["etf_cagr_tax"]),
                _pct(r["max_dd"]), _num(r["sharpe"]), str(int(r["n_taken"])),
            ])
        parts.append("<h3>Ergebnis je Trade</h3>" + '<div class="wrap">' + _table(
            ["Variante", "Trades", "Treffer", "Erwartung", "95 %-Intervall", "Zufall", "Signal − Zufall", "Ø Tage"],
            rows) + "</div>")
        parts.append("<h3>Portfolio</h3>" + '<div class="wrap">' + _table(
            ["Variante", "Rendite p. a.", "nach Steuern", f"{html.escape(sub['bench_label'].iloc[0])} nach Steuern",
             "Max. Rückgang", "Sharpe", "Trades"], rows2) + "</div>")

    parts.append("<h2>Auswahlregel und Vorgehen</h2><ul>")
    parts.append("<li><b>Auswahlregel (vorab festgelegt):</b> höchste Rendite p. a. nach Steuern im Schnitt über "
                 "alle getesteten Märkte, sofern die Erwartung je Trade in jedem Markt über null liegt.</li>")
    parts.append(f"<li><b>Nur In-Sample:</b> Alle Zahlen stammen aus den Jahren bis {split:%Y}. Die Jahre danach "
                 "werden hier bewusst nicht gezeigt, damit sie die Auswahl nicht beeinflussen.</li>")
    parts.append("<li><b>Finaltest:</b> Die gewählte Variante wird anschließend auf Aktien aus der Schweiz, "
                 "Frankreich und den Niederlanden geprüft, die nie zur Entwicklung genutzt wurden.</li>")
    parts.append("<li><b>Signal − Zufall:</b> Wie viel R je Trade das Einstiegssignal mehr bringt als ein zufälliger "
                 "Einstieg an einem Tag, an dem dieselben Filter erfüllt waren.</li></ul>")

    parts.append("<h2>Die Varianten</h2>")
    for name, v in cfg["variants"].items():
        from copy import deepcopy

        vcfg = deepcopy(cfg)
        for k, sub in (v.get("overrides") or {}).items():
            for kk, val in sub.items():
                if isinstance(val, dict):
                    vcfg[k][kk].update(val)
                else:
                    vcfg[k][kk] = val
        m = cfg["markets"]["de"]
        parts.append(f"<h3>{html.escape(v['label'])}</h3><ul>"
                     + "".join(f"<li>{x}</li>" for x in rules_text(vcfg, m)) + "</ul>")
    parts.append('<p class="note">Kein Anlagerat; vergangene Ergebnisse sind keine Garantie.</p>')
    return (
        '<!doctype html><html lang="de"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Variantenvergleich</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )
