# Quantitative Alpha Signal Generator & ML Backtester

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-alpha-engine-wnqnqzaqtbajwzsgwhlvfl.streamlit.app/)

> 🚀 **Live Interactive Demo:** [quant-alpha-engine-wnqnqzaqtbajwzsgwhlvfl.streamlit.app](https://quant-alpha-engine-wnqnqzaqtbajwzsgwhlvfl.streamlit.app/)

An end-to-end quantitative portfolio framework that extracts statistical signals (alphas) from raw market data, trains machine learning models (Ridge Regression, Random Forest) using walk-forward time-series validation, and evaluates strategies with quantitative metrics (Sharpe Ratio, Information Coefficient, Max Drawdown).

---

## 📌 Features
- **Vectorized Alpha Extraction:** Cross-sectional ranking, rolling volatility, mean reversion, and momentum factors.
- **Leakage-Free Validation:** Strict `TimeSeriesSplit` cross-validation to prevent lookahead bias.
- **Quant Metrics Engine:** Computes daily Information Coefficient (IC), IC Information Ratio (ICIR), Sharpe Ratio, and Maximum Drawdown against the S&P 500 benchmark.

# Quantitative Alpha Signal Generator & ML Backtesting Engine

Panel pipeline that:

1. Downloads daily OHLCV from Yahoo Finance
2. Builds vectorized, cross-sectionally neutralized alpha features
3. Trains Ridge and Random Forest with date-based walk-forward CV
4. Backtests a dollar-neutral long/short book and reports IC, Sharpe, and max drawdown

## Setup (Windows PowerShell)

```powershell
cd D:\Tech-pdfs\ML-PROJECT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the venv script is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run the research pipeline

```powershell
python main.py
```

Writes `artifacts/` (models, OOS predictions, IC series, Sharpe/MDD summaries, cumulative-return charts).

## Launch the dashboard

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), New app → Main file `app.py`.
3. First load downloads Yahoo data and trains; later loads use Streamlit cache until restart.

Yahoo Finance may rate-limit cloud IPs. If download fails in the cloud, persist a local parquet snapshot of the panel.

## GitHub Pages

GitHub Pages hosts static files only and cannot run this Streamlit/Python app. Use Streamlit Community Cloud (or similar) for the interactive dashboard.

## Notes

- Features on date `t` use `Close_t`. The label is the close-to-close return from `t` to `t+1` (research convention). A live book should delay execution.
- Universe is seven mega-cap names; quintile books are sparse. Treat results as a template, not a production allocation.
