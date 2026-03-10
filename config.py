# Data parameters
START_DATE = '2015-01-01'
END_DATE = '2025-01-25'
DATA_SOURCE = 'yfinance'

# Universe parameters
INDEX_TICKER = 'SPY'
MIN_PRICE = 5.0  # Filter out penny stocks
MIN_VOLUME = 100000  # Minimum average daily volume
MIN_DATA_DAYS = 252  # Minimum required data points (1 year)

# Factor parameters
MOMENTUM_LOOKBACK = 252  # 12 months
MOMENTUM_SKIP = 21  # Skip last 1 month to avoid short-term reversal
REBALANCE_FREQ = 'BM'  # Monthly rebalancing

# Portfolio construction parameters
N_QUANTILES = 5
LONG_QUANTILE = 5 # Top quintile
SHORT_QUANTILE = 1  # Bottom quintile
LONG_EXPOSURE = 1.3
SHORT_EXPOSURE = 0.3
NORMALIZE_METHOD = 'z-score' # rank, min-max

# Transaction costs
TRANSACTION_COST = 0.001  # 10 bps per trade (single-sided)

# Performance parameters
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252

# project dir
import os
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')