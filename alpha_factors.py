"""Vectorized statistical alpha features with cross-sectional neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG, PipelineConfig


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Z-score a feature across tickers on each date (market-neutral rank space)."""

    def _z(x: pd.Series) -> pd.Series:
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return x * 0.0
        return (x - x.mean()) / std

    return series.groupby(level="Date", group_keys=False).transform(_z)


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Percentile rank across tickers on each date (uniform on [0, 1])."""
    return series.groupby(level="Date", group_keys=False).rank(pct=True)


class AlphaGenerator:
    """
    Build point-in-time alpha features from a [Date, Ticker] OHLCV panel.

    Rolling statistics use only current and past observations for that ticker.
    Cross-sectional neutralization is applied last so each day's features are
    comparable and do not leak other days' distributions.
    """

    def __init__(self, cfg: PipelineConfig = CFG) -> None:
        self.cfg = cfg

    def _unstack_field(self, panel: pd.DataFrame, field: str) -> pd.DataFrame:
        return panel[field].unstack("Ticker").sort_index()

    def momentum_ma_ratio(self, close: pd.DataFrame) -> pd.DataFrame:
        """Short MA / long MA (trend)."""
        ma_s = close.rolling(self.cfg.ma_short, min_periods=self.cfg.ma_short).mean()
        ma_l = close.rolling(self.cfg.ma_long, min_periods=self.cfg.ma_long).mean()
        return ma_s / ma_l.replace(0, np.nan)

    def mean_reversion_pctb(self, close: pd.DataFrame) -> pd.DataFrame:
        """Blend of 20-day stochastic %K and Bollinger %B."""
        w = self.cfg.bb_window
        roll_mean = close.rolling(w, min_periods=w).mean()
        roll_std = close.rolling(w, min_periods=w).std(ddof=0)
        upper = roll_mean + self.cfg.bb_std * roll_std
        lower = roll_mean - self.cfg.bb_std * roll_std
        pct_b = (close - lower) / (upper - lower).replace(0, np.nan)

        low_n = close.rolling(w, min_periods=w).min()
        high_n = close.rolling(w, min_periods=w).max()
        stoch = (close - low_n) / (high_n - low_n).replace(0, np.nan)
        return 0.5 * (pct_b + stoch)

    def vol_adjusted_momentum(self, close: pd.DataFrame) -> pd.DataFrame:
        """10-day return / 10-day rolling std of daily returns."""
        w = self.cfg.vol_mom_window
        ret_w = close.pct_change(w)
        daily = close.pct_change()
        vol = daily.rolling(w, min_periods=w).std(ddof=0)
        return ret_w / vol.replace(0, np.nan)

    def volume_price_corr(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """5-day rolling correlation of volume vs price changes."""
        w = self.cfg.vp_corr_window
        d_px = close.pct_change()
        d_vol = volume.pct_change()
        return d_px.rolling(w, min_periods=w).corr(d_vol)

    def build_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return feature panel aligned to [Date, Ticker] plus raw alphas."""
        close = self._unstack_field(panel, "Close")
        volume = self._unstack_field(panel, "Volume")

        raw = {
            "mom_ma_ratio": self.momentum_ma_ratio(close),
            "mean_rev_pctb": self.mean_reversion_pctb(close),
            "vol_adj_mom": self.vol_adjusted_momentum(close),
            "vol_price_corr": self.volume_price_corr(close, volume),
        }

        frames: list[pd.Series] = []
        for name, wide in raw.items():
            long = wide.stack().rename(name)
            long.index = long.index.set_names(["Date", "Ticker"])
            frames.append(long)
            frames.append(cross_sectional_zscore(long).rename(f"{name}_xs"))

        return pd.concat(frames, axis=1).sort_index()

    @staticmethod
    def next_day_returns(panel: pd.DataFrame) -> pd.Series:
        """Next-day simple return (Close_{t+1} / Close_t - 1)."""
        close = panel["Close"].unstack("Ticker")
        fwd = close.pct_change().shift(-1)
        y = fwd.stack()
        y.index = y.index.set_names(["Date", "Ticker"])
        y.name = "fwd_return"
        return y

    @staticmethod
    def next_day_return_rank(fwd_return: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank of next-day returns (target)."""
        ranked = cross_sectional_rank(fwd_return)
        ranked.name = "fwd_return_rank"
        return ranked
