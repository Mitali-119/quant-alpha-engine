"""Project-wide configuration for the alpha / ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    tickers: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
    )
    benchmark_ticker: str = "SPY"
    lookback_years: int = 5
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    ma_short: int = 5
    ma_long: int = 20
    bb_window: int = 20
    bb_std: float = 2.0
    vol_mom_window: int = 10
    vp_corr_window: int = 5

    n_splits: int = 5
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 200
    rf_max_depth: int = 6
    rf_random_state: int = 42

    long_short_quantile: float = 0.2
    trading_days: int = 252

    feature_cols: tuple[str, ...] = (
        "mom_ma_ratio_xs",
        "mean_rev_pctb_xs",
        "vol_adj_mom_xs",
        "vol_price_corr_xs",
    )


CFG = PipelineConfig()
