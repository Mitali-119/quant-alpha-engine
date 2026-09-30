"""Download, clean, and panel-structure daily OHLCV data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf

from config import CFG, PipelineConfig

MIN_TICKERS = 2
MIN_BARS_PER_TICKER = 60


class MarketDataError(RuntimeError):
    """Raised when Yahoo Finance returns no usable OHLCV panel."""


def _normalize_ohlcv_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Return columns as MultiIndex (Field, Ticker) when multiple tickers."""
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns.name = "Field"
        return raw

    level0 = {str(x).lower() for x in raw.columns.get_level_values(0)}
    price_fields = {"open", "high", "low", "close", "adj close", "volume"}
    if level0 & price_fields:
        raw.columns = raw.columns.set_names(["Field", "Ticker"])
        return raw

    raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)
    raw.columns = raw.columns.set_names(["Field", "Ticker"])
    return raw


def _canonical_field_name(name: object) -> str:
    text = str(name)
    lowered = text.lower()
    if lowered in {"adj close", "adj_close"}:
        return "Close"
    return text.title()


def _yf_download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    target = symbols[0] if len(symbols) == 1 else symbols
    return yf.download(
        target,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )


def _to_panel(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if raw.empty:
        raise MarketDataError("yfinance returned an empty frame.")

    raw = _normalize_ohlcv_columns(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        wanted = {"Open", "High", "Low", "Close", "Volume"}
        renamed = raw.copy()
        renamed.columns = pd.MultiIndex.from_tuples(
            [(_canonical_field_name(f), t) for f, t in renamed.columns],
            names=["Field", "Ticker"],
        )
        keep = [c for c in renamed.columns if c[0] in wanted]
        renamed = renamed.loc[:, keep]
        panel = (
            renamed.stack(level="Ticker")
            .rename_axis(index=["Date", "Ticker"])
            .sort_index()
        )
    else:
        panel = raw.rename(columns=_canonical_field_name).copy()
        panel["Ticker"] = tickers[0]
        panel = panel.set_index("Ticker", append=True)
        panel.index = panel.index.set_names(["Date", "Ticker"])
        panel = panel.sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise MarketDataError(f"Missing OHLCV columns: {missing}")
    return panel[required].sort_index()


def _clean_panel(panel: pd.DataFrame, requested: tuple[str, ...]) -> pd.DataFrame:
    panel = panel.groupby(level="Ticker", group_keys=False).ffill()
    panel = panel.dropna(how="any")

    counts = panel.groupby(level="Ticker").size()
    keep = counts[counts >= MIN_BARS_PER_TICKER].index.astype(str).tolist()
    dropped = [t for t in requested if t not in keep]
    if dropped:
        panel = panel[panel.index.get_level_values("Ticker").isin(keep)]

    alive = sorted(panel.index.get_level_values("Ticker").unique().astype(str).tolist())
    if len(alive) < MIN_TICKERS:
        raise MarketDataError(
            "Not enough tickers with usable history after dropna "
            f"(kept {alive or 'none'}; dropped {dropped or 'none'}). "
            "Check Yahoo Finance availability / network."
        )

    dates = pd.to_datetime(panel.index.get_level_values("Date"))
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    tickers_idx = panel.index.get_level_values("Ticker").astype(str)
    panel.index = pd.MultiIndex.from_arrays([dates, tickers_idx], names=["Date", "Ticker"])
    return panel.sort_index()


def download_ohlcv(
    tickers: Iterable[str] | None = None,
    *,
    years: int | None = None,
    cfg: PipelineConfig = CFG,
) -> pd.DataFrame:
    """
    Download daily OHLCV and return a MultiIndex frame indexed by [Date, Ticker].

    Tickers that fail or have too few bars are dropped. Remaining gaps are
    forward-filled per ticker, then any leftover NaNs are dropped.
    """
    tickers = tuple(tickers or cfg.tickers)
    years = years or cfg.lookback_years
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=int(years * 365.25) + 14)).isoformat()
    end_s = end.isoformat()

    try:
        raw = _yf_download(list(tickers), start, end_s)
        panel = _to_panel(raw, tickers)
    except Exception:
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for symbol in tickers:
            try:
                one = _yf_download([symbol], start, end_s)
                frames.append(_to_panel(one, (symbol,)))
            except Exception:
                failed.append(symbol)
        if not frames:
            raise MarketDataError(
                "Yahoo Finance download failed for every ticker. "
                f"Requested {list(tickers)}."
            )
        panel = pd.concat(frames).sort_index()

    return _clean_panel(panel, tickers)


def download_benchmark(
    ticker: str | None = None,
    *,
    index: pd.DatetimeIndex | None = None,
    cfg: PipelineConfig = CFG,
) -> pd.Series:
    """Download adjusted close for the benchmark (default SPY)."""
    ticker = ticker or cfg.benchmark_ticker
    start = (index.min() - pd.Timedelta(days=10)).date().isoformat() if index is not None else None
    end = (index.max() + pd.Timedelta(days=3)).date().isoformat() if index is not None else None
    try:
        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        raise MarketDataError(f"Benchmark download failed for {ticker}: {exc}") from exc
    if px is None or px.empty:
        raise MarketDataError(f"Benchmark download returned no data for {ticker}.")
    close = px["Close"] if "Close" in px.columns else px.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.name = ticker
    close = close.dropna()
    if index is not None:
        close = close.reindex(index).ffill()
    if close.dropna().empty:
        raise MarketDataError(f"Benchmark series for {ticker} is empty after cleaning.")
    return close
