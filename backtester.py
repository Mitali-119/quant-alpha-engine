"""Vectorized long/short backtest and institutional diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CFG, PipelineConfig


def long_short_weights(
    scores: pd.Series,
    *,
    quantile: float | None = None,
) -> pd.Series:
    """
    Dollar-neutral weights: long top quantile, short bottom quantile, equal |w|.

    Long and short sleeves each sum to +/- 1 (gross 200%, net 0).
    """
    q = quantile if quantile is not None else CFG.long_short_quantile

    def _w(x: pd.Series) -> pd.Series:
        valid = x.dropna()
        n = valid.shape[0]
        w = pd.Series(0.0, index=x.index)
        if n < 4:
            return w
        ranks = valid.rank(method="first")
        n_long = max(1, int(np.floor(n * q)))
        hi = ranks >= (n - n_long + 1)
        lo = ranks <= n_long
        if hi.any():
            w.loc[hi.index[hi]] = 1.0 / int(hi.sum())
        if lo.any():
            w.loc[lo.index[lo]] = -1.0 / int(lo.sum())
        return w

    return scores.groupby(level="Date", group_keys=False).apply(_w)


def portfolio_returns(weights: pd.Series, fwd_return: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [weights.rename("w"), fwd_return.rename("r")],
        axis=1,
        join="inner",
    ).dropna()
    daily = (aligned["w"] * aligned["r"]).groupby(level="Date").sum().sort_index()
    daily.name = "strategy"
    return daily


def information_coefficient(pred: pd.Series, fwd_return: pd.Series) -> pd.Series:
    """Daily Spearman IC between scores and realized forward returns."""

    def _ic(df: pd.DataFrame) -> float:
        if df.shape[0] < 3:
            return np.nan
        corr, _ = spearmanr(df["p"], df["r"])
        return float(corr)

    pair = pd.concat(
        [pred.rename("p"), fwd_return.rename("r")],
        axis=1,
        join="inner",
    ).dropna()
    ic = pair.groupby(level="Date").apply(_ic, include_groups=False)
    ic.name = "ic"
    return ic


def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def performance_summary(
    daily: pd.Series,
    ic: pd.Series,
    *,
    trading_days: int = CFG.trading_days,
) -> dict[str, float]:
    mu = float(daily.mean())
    sigma = float(daily.std(ddof=1))
    sharpe = (mu / sigma) * np.sqrt(trading_days) if sigma > 0 else np.nan
    wealth = (1.0 + daily).cumprod()
    ic_clean = ic.dropna()
    mean_ic = float(ic_clean.mean()) if len(ic_clean) else np.nan
    ic_std = float(ic_clean.std(ddof=1)) if len(ic_clean) > 1 else np.nan
    ic_ir = mean_ic / ic_std if ic_std and ic_std > 0 else np.nan
    return {
        "mean_daily_return": mu,
        "ann_return": float((1.0 + mu) ** trading_days - 1.0),
        "ann_vol": sigma * np.sqrt(trading_days),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(wealth),
        "cum_return": float(wealth.iloc[-1] - 1.0) if len(wealth) else np.nan,
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "ic_ir_ann": ic_ir * np.sqrt(trading_days) if ic_ir == ic_ir else np.nan,
        "hit_rate": float((daily > 0).mean()),
        "n_days": float(daily.shape[0]),
    }


def align_benchmark(daily: pd.Series, bench_close: pd.Series) -> pd.Series:
    b = bench_close.reindex(daily.index).ffill().pct_change()
    b.name = "benchmark"
    return b.fillna(0.0)


def plot_cumulative(
    daily: pd.Series,
    bench_daily: pd.Series,
    path: Path,
) -> None:
    strat_w = (1.0 + daily).cumprod()
    bench_w = (1.0 + bench_daily.reindex(daily.index).fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strat_w.index, strat_w.values, label="Long/Short Alpha")
    ax.plot(bench_w.index, bench_w.values, label="S&P 500 (SPY)", alpha=0.85)
    ax.set_title("Cumulative Return vs Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_backtest(
    oos_pred: pd.Series,
    fwd_return: pd.Series,
    bench_close: pd.Series,
    *,
    model_name: str,
    out_dir: Path | None = None,
    cfg: PipelineConfig = CFG,
) -> tuple[pd.Series, dict[str, float], pd.Series]:
    weights = long_short_weights(oos_pred)
    daily = portfolio_returns(weights, fwd_return)
    ic = information_coefficient(oos_pred, fwd_return)
    stats = performance_summary(daily, ic, trading_days=cfg.trading_days)
    bench = align_benchmark(daily, bench_close)
    out = Path(out_dir or cfg.artifacts_dir)
    plot_cumulative(daily, bench, out / f"cum_return_{model_name}.png")
    pd.DataFrame({"strategy": daily, "benchmark": bench}).to_csv(out / f"daily_{model_name}.csv")
    pd.Series(stats).to_csv(out / f"summary_{model_name}.csv")
    ic.to_csv(out / f"ic_{model_name}.csv")
    return daily, stats, ic
