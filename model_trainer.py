"""Walk-forward training of Ridge and RandomForest on panel alphas."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CFG, PipelineConfig


def make_xy(
    features: pd.DataFrame,
    fwd_rank: pd.Series,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_cols or list(CFG.feature_cols)
    x = features[cols]
    data = pd.concat([x, fwd_rank], axis=1, join="inner").dropna()
    return data[cols], data[fwd_rank.name]


def _date_splits(dates: pd.DatetimeIndex, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = pd.DatetimeIndex(sorted(pd.unique(dates)))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_i, test_i in tscv.split(unique):
        train_dates = set(unique[train_i])
        test_dates = set(unique[test_i])
        splits.append(
            (
                np.where(dates.isin(train_dates))[0],
                np.where(dates.isin(test_dates))[0],
            )
        )
    return splits


def _ridge(cfg: PipelineConfig = CFG) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=cfg.ridge_alpha)),
        ]
    )


def _forest(cfg: PipelineConfig = CFG) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=cfg.rf_random_state,
    )


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cfg: PipelineConfig = CFG,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | dict[str, pd.Series]]]:
    """
    Expanding-window CV on calendar dates (not shuffled rows).

    Returns OOS predictions for Ridge and RF, plus fold-level metrics.
    """
    dates = pd.DatetimeIndex(X.index.get_level_values("Date"))
    splits = _date_splits(dates, cfg.n_splits)

    oos = pd.DataFrame(index=X.index, columns=["ridge", "rf"], dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    importances: dict[str, list[pd.Series]] = {"ridge": [], "rf": []}

    for fold, (tr, te) in enumerate(splits, start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        ridge = _ridge(cfg)
        rf = _forest(cfg)
        ridge.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)

        p_ridge = ridge.predict(X_te)
        p_rf = rf.predict(X_te)
        oos.iloc[te, oos.columns.get_loc("ridge")] = p_ridge
        oos.iloc[te, oos.columns.get_loc("rf")] = p_rf

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "ridge_rmse": float(mean_squared_error(y_te, p_ridge) ** 0.5),
                "ridge_r2": float(r2_score(y_te, p_ridge)),
                "rf_rmse": float(mean_squared_error(y_te, p_rf) ** 0.5),
                "rf_r2": float(r2_score(y_te, p_rf)),
            }
        )

        importances["ridge"].append(
            pd.Series(
                np.abs(ridge.named_steps["model"].coef_),
                index=X.columns,
                name=f"fold_{fold}",
            )
        )
        importances["rf"].append(
            pd.Series(rf.feature_importances_, index=X.columns, name=f"fold_{fold}")
        )

    oos = oos.dropna(how="all")
    fold_metrics = pd.DataFrame(fold_rows)
    importance_tables = {
        k: pd.concat(v, axis=1).mean(axis=1).sort_values(ascending=False).rename("importance")
        for k, v in importances.items()
    }
    return oos, {"fold_metrics": fold_metrics, "importances": importance_tables}


def fit_final_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cfg: PipelineConfig = CFG,
) -> dict[str, object]:
    """Fit on the full sample for artifact export (not for backtest scores)."""
    ridge = _ridge(cfg)
    rf = _forest(cfg)
    ridge.fit(X, y)
    rf.fit(X, y)
    return {"ridge": ridge, "rf": rf}


def save_artifacts(
    models: dict[str, object],
    importances: dict[str, pd.Series],
    oos: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    out_dir: Path | None = None,
) -> Path:
    out = Path(out_dir or CFG.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out / "models.joblib")
    oos.to_parquet(out / "oos_predictions.parquet")
    fold_metrics.to_csv(out / "cv_metrics.csv", index=False)
    for name, series in importances.items():
        series.to_csv(out / f"feature_importance_{name}.csv")
    return out
