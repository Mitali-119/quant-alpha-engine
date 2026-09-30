"""Run the full research pipeline from the command line."""

from __future__ import annotations

import json

from alpha_factors import AlphaGenerator
from backtester import run_backtest
from config import CFG
from data_loader import download_benchmark, download_ohlcv
from model_trainer import fit_final_models, make_xy, save_artifacts, walk_forward_predict


def main() -> None:
    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    panel = download_ohlcv()
    gen = AlphaGenerator()
    features = gen.build_features(panel)
    fwd = gen.next_day_returns(panel)
    y_rank = gen.next_day_return_rank(fwd)

    X, y = make_xy(features, y_rank)
    print(f"Panel rows={len(panel):,}  feature rows={len(X):,}  dates={X.index.get_level_values('Date').nunique()}")

    oos, extras = walk_forward_predict(X, y)
    models = fit_final_models(X, y)
    save_artifacts(models, extras["importances"], oos, extras["fold_metrics"])

    dates = pd_dates = X.index.get_level_values("Date").unique().sort_values()
    spy = download_benchmark(index=pd_dates)

    summaries: dict[str, dict[str, float]] = {}
    for col in oos.columns:
        pred = oos[col].dropna()
        _, stats, _ = run_backtest(pred, fwd, spy, model_name=col)
        summaries[col] = stats
        print(f"\n=== {col.upper()} ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:18s} {value: .4f}")
            else:
                print(f"  {key}: {value}")

    extras["fold_metrics"].to_csv(CFG.artifacts_dir / "cv_metrics.csv", index=False)
    (CFG.artifacts_dir / "summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nArtifacts written to {CFG.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
