"""Kursdaten, Universen und Termine der Quartalszahlen.

Quellen:
- "yahoo":     Yahoo Finance über yfinance (Hauptquelle), Stooq als Ersatz
- "stooq":     nur Stooq (Tagesdaten als CSV)
- "synthetic": künstliche Kurse zum Testen der Pipeline ohne Internet

Alle Kurse werden je Ticker als Parquet-Datei in data/cache/ abgelegt und bei
späteren Läufen nur noch ergänzt.
"""
from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = ROOT / "config" / "universes"
CACHE_DIR = ROOT / "data" / "cache"
COLUMNS = ["open", "high", "low", "close", "volume"]
USER_AGENT = "Mozilla/5.0 (compatible; swing-scanner/1.0)"


# ---------------------------------------------------------------------------
# Universen
# ---------------------------------------------------------------------------
def load_universe(names: list[str]) -> pd.DataFrame:
    """Liest die Ticker-Listen; doppelte Ticker werden nur einmal geführt."""
    frames = []
    for name in names:
        path = UNIVERSE_DIR / f"{name}.csv"
        if name == "sp500":
            path = _ensure_sp500(path)
        df = pd.read_csv(path, dtype=str)
        df["universe"] = name
        frames.append(df)
    uni = pd.concat(frames, ignore_index=True)
    uni["sector"] = uni["sector"].fillna("Unbekannt")
    return uni.drop_duplicates("ticker").reset_index(drop=True)


def _ensure_sp500(path: Path) -> Path:
    """Lädt die aktuelle S&P-500-Liste (Datensatz auf GitHub, gepflegt nach
    Wikipedia); nutzt die gespeicherte Kopie, wenn das nicht klappt."""
    try:
        import requests

        url = (
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
            "main/data/constituents.csv"
        )
        text = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30).text
        table = pd.read_csv(io.StringIO(text))
        df = pd.DataFrame(
            {
                "ticker": table["Symbol"].str.replace(".", "-", regex=False),
                "name": table["Security"],
                "sector": table["GICS Sector"],
            }
        )
        if len(df) > 450:
            df.to_csv(path, index=False)
            log.info("S&P-500-Liste aktualisiert (%d Titel)", len(df))
    except Exception as exc:  # noqa: BLE001
        log.warning("S&P-500-Liste nicht geladen (%s), nutze gespeicherte Kopie", exc)
    if not path.exists():
        raise FileNotFoundError("Keine S&P-500-Liste verfügbar: " + str(path))
    return path


# ---------------------------------------------------------------------------
# Kursdaten
# ---------------------------------------------------------------------------
def load_prices(
    tickers: list[str], start: str, source: str = "yahoo", end: str | None = None
) -> dict[str, pd.DataFrame]:
    """Gibt je Ticker einen DataFrame mit open/high/low/close/volume zurück."""
    if source == "synthetic":
        return {t: synthetic_prices(t, start, end) for t in tickers}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    todo = []
    for t in tickers:
        cached = _read_cache(t)
        if cached is not None and cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=10):
            result[t] = cached
        todo.append(t)  # immer aktualisieren; Cache spart nur Historie

    if source == "yahoo":
        fresh = _download_yahoo(todo, start, end, result)
    else:
        fresh = {}
    missing = [t for t in todo if t not in fresh]
    if missing:
        log.info("%d Ticker über Stooq nachladen", len(missing))
        for t in missing:
            df = _download_stooq(t, start, end)
            if df is not None:
                fresh[t] = df

    for t, df in fresh.items():
        df = clean_prices(df, require_volume=not t.startswith("^"))
        if df.empty:
            continue
        old = result.get(t)
        if old is not None:
            df = pd.concat([old[old.index < df.index[0]], df])
        result[t] = df
        _write_cache(t, df)

    for t in tickers:
        if t in result:
            result[t] = result[t][result[t].index >= pd.Timestamp(start)]
            if end:
                result[t] = result[t][result[t].index <= pd.Timestamp(end)]
    lost = [t for t in tickers if t not in result or result[t].empty]
    if lost:
        log.warning("Keine Daten für %d Ticker: %s", len(lost), ", ".join(lost))
    return {t: df for t, df in result.items() if not df.empty}


def clean_prices(df: pd.DataFrame, require_volume: bool = True) -> pd.DataFrame:
    """Entfernt Lücken, Nullvolumen-Tage und unplausible Zeilen.
    Indizes haben oft kein Volumen, daher dort require_volume=False."""
    df = df[COLUMNS].astype(float).dropna()
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    if require_volume:
        df = df[df["volume"] > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('^', '_')}.parquet"


def _read_cache(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            return None
    return None


def _write_cache(ticker: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(ticker))


def _download_yahoo(tickers, start, end, cached) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i : i + 40]
        # Bei vorhandenem Cache nur die letzten Wochen laden
        starts = [
            (cached[t].index[-1] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            if t in cached
            else start
            for t in chunk
        ]
        chunk_start = min(starts)
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    start=chunk_start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Yahoo-Fehler (%s), Versuch %d", exc, attempt + 1)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.rename(columns=str.lower)
                if df[COLUMNS].dropna().empty:
                    continue
                out[t] = df[COLUMNS]
            except (KeyError, TypeError):
                continue
        time.sleep(1)
    log.info("Yahoo: %d von %d Tickern geladen", len(out), len(tickers))
    return out


def _stooq_symbol(ticker: str) -> str:
    if ticker == "^GDAXI":
        return "^dax"
    if ticker == "^GSPC":
        return "^spx"
    if ticker.endswith(".DE"):
        return ticker[:-3].lower() + ".de"
    return ticker.lower().replace("-", ".") + ".us"


def _download_stooq(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    try:
        import requests

        url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(ticker)}&i=d"
        text = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30).text
        if not text.startswith("Date"):
            return None
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
        df = df.rename(columns=str.lower)
        if "volume" not in df:
            df["volume"] = 1.0  # Indizes haben bei Stooq kein Volumen
        df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df[COLUMNS]
    except Exception as exc:  # noqa: BLE001
        log.warning("Stooq-Fehler für %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Termine der Quartalszahlen
# ---------------------------------------------------------------------------
def load_earnings_dates(tickers: list[str], source: str = "yahoo") -> dict[str, list[pd.Timestamp]]:
    """Vergangene und angekündigte Termine der Quartalszahlen je Ticker.
    Ticker ohne verfügbare Termine fehlen im Ergebnis."""
    if source == "synthetic":
        return {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "earnings.json"
    cache: dict[str, list[str]] = json.loads(path.read_text()) if path.exists() else {}
    import yfinance as yf

    for t in tickers:
        if t in cache:
            continue
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=100)
            dates = [] if df is None else sorted({d.strftime("%Y-%m-%d") for d in df.index})
        except Exception:  # noqa: BLE001
            dates = []
        cache[t] = dates
        time.sleep(0.3)
    path.write_text(json.dumps(cache))
    return {t: [pd.Timestamp(d) for d in v] for t, v in cache.items() if t in tickers and v}


# ---------------------------------------------------------------------------
# Synthetische Daten (nur zum Testen der Pipeline!)
# ---------------------------------------------------------------------------
def synthetic_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Erzeugt reproduzierbare, realistisch aussehende Kurse mit Trendphasen,
    Volatilitätsclustern und Volumenspitzen. Die Ergebnisse eines Backtests
    auf diesen Daten sagen NICHTS über echte Märkte aus."""
    import hashlib

    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    n = len(idx)
    is_index = ticker.startswith("^")
    base_vol = 0.011 if is_index else rng.uniform(0.015, 0.03)
    # Trendregime wechseln im Schnitt alle 120 Tage
    drift = np.zeros(n)
    spread = 0.0004 if is_index else 0.0012
    level = rng.normal(0.0003, spread)
    for i in range(n):
        if rng.random() < 1 / 120:
            level = rng.normal(0.0003, spread)
        drift[i] = level
    # GARCH-artige Volatilität
    vol = np.empty(n)
    v = base_vol
    for i in range(n):
        v = 0.94 * v + 0.06 * base_vol * rng.lognormal(0, 0.5)
        vol[i] = v
    ret = drift + vol * rng.standard_t(5, n) / np.sqrt(5 / 3)
    close = 50 * np.exp(np.cumsum(ret))
    gap = vol * rng.normal(0, 0.4, n)
    open_ = close / np.exp(ret) * np.exp(gap)
    rng_hi = np.abs(rng.normal(0, 0.6, n)) * vol
    rng_lo = np.abs(rng.normal(0, 0.6, n)) * vol
    high = np.maximum(open_, close) * np.exp(rng_hi)
    low = np.minimum(open_, close) * np.exp(-rng_lo)
    volume = rng.lognormal(13.5, 0.35, n) * (1 + 8 * np.abs(ret))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
    return df
