"""Lightweight research dashboard with cached, failure-safe pipeline steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from alpha_factors import AlphaGenerator
from backtester import (
    align_benchmark,
    information_coefficient,
    long_short_weights,
    performance_summary,
    portfolio_returns,
)
from config import CFG
from data_loader import MarketDataError, download_benchmark, download_ohlcv
from model_trainer import make_xy, walk_forward_predict

CACHE_TTL_SECONDS = 3600
ARTIFACTS = Path(CFG.artifacts_dir)

st.set_page_config(page_title="Quant Alpha Engine", layout="wide")
st.title("Quantitative Alpha Signal Generator & ML Backtester")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_download_ohlcv(tickers: tuple[str, ...], years: int) -> pd.DataFrame:
    panel = download_ohlcv(tickers, years=years)
    if panel.empty:
        raise MarketDataError("OHLCV panel is empty after cleaning.")
    return panel


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    gen = AlphaGenerator()
    features = gen.build_features(panel)
    fwd = gen.next_day_returns(panel)
    y_rank = gen.next_day_return_rank(fwd)
    if features.dropna(how="all").empty:
        raise RuntimeError("Feature generation produced no valid rows.")
    return features, fwd, y_rank


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_walk_forward(features: pd.DataFrame, y_rank: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    X, y = make_xy(features, y_rank)
    if X.empty:
        raise RuntimeError("No overlapping feature/target rows for training.")
    oos, extras = walk_forward_predict(X, y)
    importances = {k: v for k, v in extras["importances"].items()}
    return oos, extras["fold_metrics"], importances


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_benchmark(dates: tuple[pd.Timestamp, ...], ticker: str) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    return download_benchmark(ticker, index=idx)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_backtest_block(
    pred: pd.Series,
    fwd: pd.Series,
    bench: pd.Series,
) -> dict[str, pd.Series | dict[str, float]]:
    weights = long_short_weights(pred)
    daily = portfolio_returns(weights, fwd)
    ic = information_coefficient(pred, fwd)
    bench_daily = align_benchmark(daily, bench)
    return {
        "daily": daily,
        "bench": bench_daily,
        "stats": performance_summary(daily, ic),
        "ic": ic,
    }


def _artifacts_ready() -> bool:
    needed = [
        ARTIFACTS / "daily_ridge.csv",
        ARTIFACTS / "daily_rf.csv",
        ARTIFACTS / "cv_metrics.csv",
        ARTIFACTS / "feature_importance_ridge.csv",
        ARTIFACTS / "feature_importance_rf.csv",
        ARTIFACTS / "summary_ridge.csv",
        ARTIFACTS / "summary_rf.csv",
    ]
    return all(path.exists() for path in needed)


def load_from_artifacts() -> dict:
    cv = pd.read_csv(ARTIFACTS / "cv_metrics.csv")
    importances = {
        "ridge": pd.read_csv(ARTIFACTS / "feature_importance_ridge.csv", index_col=0).squeeze("columns"),
        "rf": pd.read_csv(ARTIFACTS / "feature_importance_rf.csv", index_col=0).squeeze("columns"),
    }
    results: dict = {}
    for model in ("ridge", "rf"):
        daily_df = pd.read_csv(ARTIFACTS / f"daily_{model}.csv", index_col=0, parse_dates=True)
        stats_s = pd.read_csv(ARTIFACTS / f"summary_{model}.csv", index_col=0).squeeze("columns")
        stats = {str(k): float(v) for k, v in stats_s.items()}
        results[model] = {
            "daily": daily_df["strategy"],
            "bench": daily_df["benchmark"],
            "stats": stats,
            "ic": None,
        }
    return {"results": results, "importances": importances, "cv": cv, "source": "artifacts"}


def run_live_pipeline() -> dict:
    tickers = tuple(CFG.tickers)
    panel = cached_download_ohlcv(tickers, CFG.lookback_years)
    features, fwd, y_rank = cached_build_features(panel)
    oos, cv, importances = cached_walk_forward(features, y_rank)
    dates = tuple(pd.DatetimeIndex(oos.index.get_level_values("Date")).unique().sort_values())
    spy = cached_benchmark(dates, CFG.benchmark_ticker)

    results = {}
    for col in oos.columns:
        pred = oos[col].dropna()
        results[col] = cached_backtest_block(pred, fwd, spy)
    return {"results": results, "importances": importances, "cv": cv, "source": "live"}


def load_pipeline(*, force_live: bool) -> dict:
    if not force_live and _artifacts_ready():
        return load_from_artifacts()
    return run_live_pipeline()


force_live = st.sidebar.checkbox(
    "Recompute from Yahoo (slow)",
    value=False,
    help="Ignore saved artifacts and re-download / retrain. Cached for 1 hour.",
)

data = None
try:
    spinner_msg = (
        "Loading saved backtest artifacts..."
        if not force_live and _artifacts_ready()
        else "Downloading data & training model..."
    )
    with st.spinner(spinner_msg):
        data = load_pipeline(force_live=force_live)
except MarketDataError as exc:
    st.error(f"Market data error: {exc}")
    st.info("Run `python main.py` once to generate artifacts, then refresh this app.")
    st.stop()
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.info("Check network access to Yahoo Finance, or run `python main.py` first.")
    st.stop()

if data["source"] == "artifacts":
    st.sidebar.caption("Showing saved `artifacts/` from the last `python main.py` run.")
else:
    st.sidebar.caption("Live download + walk-forward (cached 1 hour).")

model = st.sidebar.selectbox("Model", list(data["results"].keys()), index=0)
block = data["results"][model]
stats = block["stats"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe (ann.)", f"{float(stats['sharpe']):.2f}")
c2.metric("Max Drawdown", f"{float(stats['max_drawdown']):.2%}")
c3.metric("Mean IC", f"{float(stats['mean_ic']):.4f}")
c4.metric("ICIR", f"{float(stats['ic_ir']):.2f}")

st.subheader("Summary")
st.dataframe(pd.Series(stats).to_frame("value"), width="stretch")

st.subheader("Cumulative return vs S&P 500 (SPY)")
cum = pd.DataFrame(
    {
        "strategy": (1.0 + block["daily"]).cumprod(),
        "SPY": (1.0 + block["bench"]).cumprod(),
    }
)
st.line_chart(cum)

left, right = st.columns(2)
with left:
    st.subheader("Feature importance")
    st.bar_chart(data["importances"][model])
with right:
    st.subheader("Walk-forward CV")
    st.dataframe(data["cv"], width="stretch")

st.caption(
    "OOS predictions come from date-based TimeSeriesSplit. "
    f"Positions are long top {CFG.long_short_quantile:.0%} / short bottom "
    f"{CFG.long_short_quantile:.0%}, dollar-neutral."
)
