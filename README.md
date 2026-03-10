## Project Overview

This project is a **quantitative research framework** for systematic factor investing, built around a cross-sectional momentum strategy on U.S. equities. Beyond producing a backtest, the codebase is designed to demonstrate end-to-end system design — from raw data acquisition through signal construction, portfolio simulation, and statistical validation — with explicit attention to the methodological pitfalls that matter in practice.

The primary theoretical reference is Asness, Moskowitz & Pedersen (2013) — *"Value and Momentum Everywhere"* (Journal of Finance). This implementation focuses on the U.S. equity leg of their framework: a 12-1 month cross-sectional momentum signal, quintile-ranked, constructed into a dollar-neutral long-short portfolio with monthly rebalancing.

**What this framework is designed to demonstrate:**

- **System design** — modular, reusable pipeline separating data, signal, portfolio, and analysis concerns; each layer is independently testable and replaceable
- **Factor validation** — IC decay analysis (month-end Spearman) to assess whether the signal has genuine predictive power across horizons, not just in-sample curve-fitting
- **Regime awareness** — explicit decomposition of performance into low/medium/high volatility environments, grounded in the AMP (2013) observation that momentum co-crashes with value during stress periods
- **Critical thinking on data quality** — survivorship bias acknowledgement, corporate action handling, missing data treatment, and outlier winsorization are all explicit pipeline steps rather than afterthoughts
- **Honest performance attribution** — transaction cost modelling, warm-up period impact, and long/short leg decomposition are broken out separately so the sources of return are traceable
- **LLM-assisted reporting** — integration with the Anthropic Claude API to auto-generate a structured performance narrative from backtest metrics, demonstrating practical applied LLM usage in a quant workflow

**Key design decisions:**
- Universe: ~460–500 current S&P 500 constituents (via Wikipedia / yfinance)
- Signal: 12-month return skipping the most recent month (`t-252` to `t-21`) to avoid short-term reversal
- Portfolio: Equal-weighted long-short, monthly rebalancing at month-end business days
- Transaction costs: Applied at each rebalance based on turnover
- Benchmark: SPY (S&P 500 ETF)
- Backtest period: Configurable via `config.py` (default: 2015–2025)

---

## Project Structure

```
.
├── main.py                          # Full pipeline entry point
├── config.py                        # All strategy parameters (edit here)
├── src/
│   ├── data_manager.py              # Data download, quality control, caching
│   ├── data_validator.py            # Missing values, outliers, corporate actions
│   ├── factor_engine.py             # Momentum calculation, signal generation, portfolio construction
│   ├── performance_analyzer.py      # Return/risk/benchmark metrics
│   └── visualizer.py                # Charts and performance report
├── LLM_test.py                      # LLM-powered narrative report generation (Anthropic API)
├── notebooks/
│   ├── analysis.ipynb                     # factor perforamnce + regime analysis (main exploration notebook)
│   └── sensitivity_decay_analysis.ipynb   # IC decay analysis (month-end Spearman)
├── data/                            # Auto-created; stores downloaded and processed CSVs
└── results/                         # Auto-created; stores charts, metrics JSON, reports
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key to be set up in terminal(only needed for LLM report)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **Never hard-code your API key in any source file.** The code reads it from the environment variable at runtime.

### 3. Run the full backtest pipeline

```bash
python main.py
```

This will:
1. Download S&P 500 price and volume data (or load from cache)
2. Validate and clean the data
3. Calculate and normalise the momentum factor
4. Generate quantile signals and construct the long-short portfolio
5. Download SPY benchmark and align return series
6. Calculate all performance metrics vs benchmark
7. Save charts, metrics JSON, and positions to `results/` and `data/`

### 4. Generate the LLM narrative report (optional)

```bash
python LLM_test.py
```

Requires `results/backtest_summary.json` to exist (produced by step 3).

### 5. Run the analysis notebook (recommended starting point)

```bash
jupyter notebook notebooks/analysis.ipynb
```

This is the main exploration notebook. It loads from `data/` and `results/` produced by `main.py` and provides:

- **Factor Analysis** — cross-sectional distribution of momentum scores, factor spread over time (90th–10th percentile), factor autocorrelation, and quantile stock counts over time
- **Performance Deep Dive** — best/worst months, yearly returns calendar, full metrics summary
- **Visualisations** — cumulative returns, drawdown, monthly heatmap, rolling 1-year Sharpe, rolling 60-day correlation with SPY
- **Regime Analysis** — classifies market dates into Low / Medium / High volatility regimes based on SPY 60-day realised vol, then computes annualised return, volatility, and Sharpe separately for each regime; includes a 4-panel chart overlaying cumulative returns with regime shading

### 6. Run the IC decay analysis notebook

```bash
jupyter notebook notebooks/sensitivity_decay_analysis.ipynb
```

---

## Configuration

All strategy parameters live in `config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `START_DATE` | `"2015-01-01"` | Backtest start date |
| `END_DATE` | `"2025-01-24"` | Backtest end date |
| `MOMENTUM_LOOKBACK` | `252` | Lookback window in trading days (~12 months) |
| `MOMENTUM_SKIP` | `21` | Skip period in trading days (~1 month) |
| `N_QUANTILES` | `5` | Number of quantile buckets |
| `LONG_QUANTILE` | `5` | Quantile to go long (top) |
| `SHORT_QUANTILE` | `1` | Quantile to go short (bottom) |
| `LONG_EXPOSURE` | `1.0` | Long leg gross exposure |
| `SHORT_EXPOSURE` | `1.0` | Short leg gross exposure |
| `TRANSACTION_COST` | `0.001` | One-way transaction cost (10 bps) |
| `REBALANCE_FREQ` | `"BM"` | Rebalance frequency (Business Month-end) |
| `NORMALIZE_METHOD` | `"z-score"` | Factor normalisation method |
| `RISK_FREE_RATE` | `0.02` | Annual risk-free rate for Sharpe calculation |
| `DATA_DIR` | `"data/"` | Directory for raw and processed data |
| `RESULTS_DIR` | `"results/"` | Directory for charts and metrics |

---

## How It Works

### Step 1 — Data Acquisition (`data_manager.py`)
Downloads adjusted close prices and volumes from yfinance for all current S&P 500 constituents. Data is cached locally after the first run. Quality filters remove stocks with insufficient history, low price/volume, or suspected delistings.

### Step 2 — Data Validation (`data_validator.py`)
Cleans the raw data: forward-fills short gaps (up to 5 days), drops stocks with >10% missing values, winsorizes daily returns at ±50%, and detects potential unadjusted corporate actions.

### Step 3 — Factor Calculation (`factor_engine.py`)
Computes the 12-1 momentum score for every stock on every date, following the signal construction in AMP (2013):

```
momentum_t = (price_{t-21} / price_{t-252}) - 1
```

The 12-month lookback with a 1-month skip is the standard AMP (2013) specification: the skip avoids contamination from short-term reversal while capturing the intermediate-horizon continuation effect. The raw factor is winsorized (1st–99th percentile) then cross-sectionally z-score normalised.

### Step 4 — Portfolio Construction (`factor_engine.py`)
Stocks are ranked into quintiles on each month-end rebalance date. The top quintile receives equal long weights summing to 100%; the bottom quintile receives equal short weights summing to −100%. Positions are held constant until the next rebalance. Transaction costs are deducted at each rebalance based on turnover.

### Step 5 — Performance Analysis (`performance_analyzer.py`)
Calculates the full suite of metrics: CAGR, annualised volatility, Sharpe, Sortino, Calmar, max drawdown, win rate, alpha, beta, tracking error, and information ratio vs SPY.

### Step 6 — Visualisation (`visualizer.py`)
Produces cumulative return curves, drawdown chart, monthly heatmap, rolling Sharpe, return distribution histogram, rolling correlation with SPY, and a combined comprehensive report.

---

## LLM Report Generation

`LLM_test.py` reads the backtest metrics JSON and uses the Anthropic Claude API to write a 3-paragraph quantitative performance narrative covering return attribution, drawdown behaviour, and strategy limitations. Key figures are injected directly into the prompt to reduce hallucination risk, and a simple validation check confirms the model cited the correct numbers before saving the output.

---

## Outputs

After a successful run you will find:

```
data/
├── prices_adjusted.csv          # Raw adjusted close prices
├── prices_adjusted_clean.csv    # Validated and cleaned prices
├── volumes.csv                  # Volume data
├── momentum_factor.csv          # Raw momentum scores
├── signals.csv                  # Quintile assignments
├── positions.csv                # Daily position weights
├── portfolio_returns.csv        # Gross daily returns
├── portfolio_returns_net.csv    # Net daily returns (after transaction costs)
└── returns_aligned.csv          # Strategy and benchmark returns, date-aligned

results/
├── cumulative_returns.png
├── drawdown.png
├── monthly_heatmap.png
├── rolling_sharpe.png
├── return_distribution.png
├── correlation.png
├── comprehensive_report.png
├── strategy_metrics.json
├── benchmark_metrics.json
└── backtest_summary.json
```

---

## Known Limitations / Future Work

These are active limitations I'm aware of — worth understanding before drawing conclusions from the results.

**Survivorship bias**
The universe is built from *current* S&P 500 constituents scraped from Wikipedia. This means stocks that were delisted, went bankrupt, or were removed from the index during the backtest window are excluded from the historical analysis. This mechanically inflates returns because we only hold stocks that "survived". A production implementation would require a point-in-time constituent database (e.g., Bloomberg, FactSet, CRSP).

**Alpha statistical significance**
The IC decay analysis (notebook) shows Spearman ICs of roughly −0.008 to −0.013 across all forward-return horizons, with t-statistics well below 1.96 (none statistically significant at the 5% level). This means the cross-sectional predictive power of the raw 12-1 momentum signal is not robustly detectable in the current sample, likely reflecting a combination of a crowded factor, recent macro regimes (COVID shock, rate cycles), and the survivorship bias noted above.

**Transaction cost model**
Costs are applied as a flat basis-point charge on turnover. This ignores market impact, bid-ask spreads, and the liquidity constraints that would apply to a real fund — particularly relevant for the short book, where borrowing costs and short squeeze risk are material.

**Look-ahead bias**
The data pipeline uses yfinance's `Adj Close`, which back-adjusts prices for splits and dividends using information available today. This can introduce subtle look-ahead bias into early periods of the backtest if adjustments changed over time.

**Single factor**
This is a pure momentum strategy. Combining it with orthogonal factors (value, quality, low-volatility) would likely improve the Sharpe ratio and reduce factor-specific drawdown risk, and is a natural extension.

**Regime sensitivity**
Momentum strategies are known to suffer severe drawdowns in sharp market reversals (e.g., the March 2020 COVID crash or the 2022 factor rotation). AMP (2013) itself documents that momentum and value tend to co-crash, and that the 2008–2009 period was particularly damaging for momentum globally. No regime filter or volatility scaling is currently applied, though the `calculate_rolling_volatility` method in `factor_engine.py` provides a starting point for this.

**Single asset class**
AMP (2013) demonstrates momentum across equities, fixed income, currencies, and commodities simultaneously, and shows that the diversification across these asset classes substantially improves risk-adjusted returns. This implementation covers only the U.S. equity leg, which means it captures none of the cross-asset diversification benefits that are central to the AMP (2013) thesis.

**Short selling constraints**
The backtest assumes frictionless short selling with no borrowing cost. In practice, the bottom quintile of S&P 500 stocks can have meaningful borrow rates, and some positions may be unavailable to short entirely.

---

## Primary Reference

> Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and Momentum Everywhere. *Journal of Finance*, 68(3), 929–985.

The 12-1 month signal construction, long-short portfolio, and monthly rebalancing convention in this project follow the U.S. equity specification in AMP (2013).
