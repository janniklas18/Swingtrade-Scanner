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
    return f"{x:,.{digits}f}{suffix}".replace(",", "X").replace(".", ",").replace("X", ".")


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
    bins = np.arange(-2.5, 4.01, 0.25)
    vals = r.clip(-2.49, 3.99)
    neg, pos = vals[vals <= 0], vals[vals > 0]
    ax.hist(neg, bins=bins, color=RED, alpha=0.6)
    ax.hist(pos, bins=bins, color=GREEN, alpha=0.6)
    ax.axvline(r.mean(), color=BLUE, lw=1.5)
    ax.text(r.mean(), ax.get_ylim()[1] * 0.95, f"  Ø {_r(r.mean())}", color=BLUE, fontsize=9, va="top")
    ax.set_xlabel("Ergebnis je Trade in R (Vielfache des Risikos)", fontsize=9, color="#4b5563")
    _style_ax(ax)
    return _img(fig)


def chart_random(real: float, rand: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 1.3))
    ax.scatter(rand, np.zeros(len(rand)), color=GREY, s=30, alpha=0.7, label="Zufällige Einstiege (je Lauf)")
    ax.scatter([real], [0], color=BLUE, s=90, marker="D", label="Echte Signale", zorder=3)
    ax.axvline(0, color="#6b7280", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Ø Ergebnis je Trade im Out-of-Sample (R)", fontsize=9, color="#4b5563")
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
def verdict(oos: dict, rand_oos: pd.Series) -> tuple[str, str, str]:
    """Ampel auf Basis des Out-of-Sample-Zeitraums."""
    n = oos.get("n", 0)
    if n < 30:
        return ("grey", "Zu wenige Trades für ein Urteil",
                f"Im Out-of-Sample gab es nur {n} Trades. Ab etwa 30, besser 100, wird das Ergebnis aussagekräftig.")
    exp, lo = oos["expectancy"], oos["ci_low"]
    q95 = rand_oos.quantile(0.95) if len(rand_oos.dropna()) else np.nan
    beats = not np.isnan(q95) and exp > q95
    if exp <= 0:
        return ("red", "Kein Vorteil erkennbar",
                f"Im Out-of-Sample verliert ein Trade im Schnitt {_r(exp)} nach Kosten. So nicht handeln.")
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
th{background:var(--soft);font-weight:600;color:#374151}td.l,th:first-child{text-align:left}
img{max-width:100%;height:auto;display:block;margin:6px 0}
ul{padding-left:20px}li{margin:3px 0}small,.note{color:var(--muted);font-size:13px}
"""


def build_report(ctx: dict) -> str:
    t = ctx["trades"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    m = ctx["market"]
    rnd = ctx["random"]
    s_all, s_is, s_oos = ctx["stats_all"], ctx["stats_is"], ctx["stats_oos"]
    rand_oos = rnd["oos_mean_r"].dropna() if not rnd.empty else pd.Series(dtype=float)
    rand_stats = {"n": int(s_oos.get("n", 0)), "expectancy": rand_oos.mean() if len(rand_oos) else np.nan}
    color, title, text = verdict(s_oos, rand_oos)

    parts = [f"<h1>Backtest: {html.escape(m['label'])}</h1>"]
    parts.append(
        f'<div class="meta">Zeitraum {ctx["start"]:%d.%m.%Y} – {ctx["end"]:%d.%m.%Y} · '
        f"Out-of-Sample ab {split + pd.Timedelta(days=1):%d.%m.%Y} · Datenquelle: {html.escape(ctx['source'])} · "
        f"erstellt {datetime.now():%d.%m.%Y %H:%M}</div>"
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
        ["", "Gesamt", f"In-Sample (bis {split:%Y})", f"Out-of-Sample (ab {split.year + 1})", "Zufall (Out-of-Sample)"],
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
        parts.append(chart_random(s_oos.get("expectancy", np.nan), rand_oos))
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
    parts.append(chart_equity(pf["equity"], pf["bench"], split, m["index_label"]))
    rows = []
    for label, key in (("Gesamt", "all"), ("In-Sample", "is"), ("Out-of-Sample", "oos")):
        s, b = pf["stats"][key], pf["bench_stats"][key]
        rows.append([
            label, _pct(s.get("cagr")), _pct(b.get("cagr")), _pct(s.get("max_dd")), _pct(b.get("max_dd")),
            _num(s.get("sharpe")), _num(b.get("sharpe")),
        ])
    parts.append('<div class="wrap">' + _table(
        ["", "Rendite p. a.", f"{m['index_label']} p. a.", "Max. Rückgang", f"{m['index_label']} Max. Rückgang",
         "Sharpe", f"{m['index_label']} Sharpe"], rows) + "</div>")
    parts.append(
        f'<p class="note">Ausgeführt wurden {pf["n_taken"]} von {len(t)} Signalen (Rest: Kapazität oder Sektorgrenze). '
        f"Im Schnitt waren {_pct(pf['invested'].mean(), 0)} des Kapitals investiert. Die Strategie ist also oft "
        "teilweise in Cash – ein Vergleich mit dem Index sollte das berücksichtigen.</p>"
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
    f, sb, sp, ex = cfg["filters"], cfg["setups"]["breakout"], cfg["setups"]["pullback"], cfg["exits"]
    parts.append("<h2>Getestete Regeln</h2><ul>")
    parts.append(
        f"<li><b>Filter:</b> {m['index_label']} über seiner {f['regime_sma']}-Tage-Linie; Aktie über SMA{f['trend_fast_sma']}, "
        f"SMA{f['trend_fast_sma']} über SMA{f['trend_slow_sma']}; ATR zwischen {_num(f['atr_pct_min'], 1)} und "
        f"{_num(f['atr_pct_max'], 1)} % vom Kurs; Ø Tagesumsatz mindestens {_num(m['min_turnover'] / 1e6, 0)} Mio. "
        f"{m['currency']}; keine Quartalszahlen in der Haltedauer (wo bekannt).</li>"
    )
    parts.append(
        f"<li><b>Ausbruch:</b> Schluss über dem {sb['lookback']}-Tage-Hoch, Volumen mindestens "
        f"{_num(sb['volume_ratio_min'], 1)}× Durchschnitt, Schluss im oberen Teil der Tagesspanne. Stop "
        f"{_num(sb['stop_atr'], 1)} ATR unter dem Signalkurs, Ziel {_num(sb['target_r'], 1)} R. Kein Trade, wenn das "
        "52-Wochen-Hoch zwischen Einstieg und Ziel liegt.</li>"
    )
    parts.append(
        f"<li><b>Rücksetzer:</b> RSI({sp['rsi_period']}) war in den letzten {sp['rsi_window']} Tagen unter {sp['rsi_max']}, "
        f"Kurs an der {sp['near_sma']}-Tage-Linie, dann Schluss über dem Vortageshoch. Stop unter dem "
        f"{sp['stop_lookback']}-Tage-Tief, Ziel am {sp['target_lookback']}-Tage-Hoch, nur bei CRV ≥ {_num(ex['min_crv'], 1)}.</li>"
    )
    parts.append(
        f"<li><b>Ausstieg:</b> Stop, Ziel oder nach {ex['time_stop_days']} Handelstagen zum Schlusskurs. "
        "Einstieg zur Eröffnung des Folgetags; Stop und Ziel am selben Tag zählen als Stop.</li></ul>"
    )
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
    table = _table(["Report", "Erstellt", "Daten", "Trades OOS", "Erwartung OOS", "Urteil"], rows)
    return (
        '<!doctype html><html lang="de"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Backtests</title><style>{CSS}</style></head><body><main>"
        "<h1>Backtests</h1><p class=\"meta\">Neueste zuerst.</p>"
        f'<div class="wrap">{table}</div></main></body></html>'
    )
