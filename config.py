# Data parameters
START_DATE = '2015-01-01'
END_DATE = '2025-01-25'
DATA_SOURCE = 'yfinance'

# Universe parameters
INDEX_TICKER = 'SPY'
MIN_PRICE = 5.0  # Filter out penny stocks
MIN_VOLUME = 100000  # Minimum average daily volume
MIN_DATA_DAYS = 252  # Minimum required data points (1 year)

# Use multi factor or single factor
USE_MULTI_FACTOR = True
FACTOR_EQUAL_WEIGHT = True 

FACTOR_LIST = ['momentum', 'low_vol', 'reversal'] # need to update after adding new factors
FACTOR_NUM = len(FACTOR_LIST)

# Momentum Factor parameters
MOMENTUM_LOOKBACK = 252  # 12 months
MOMENTUM_SKIP = 21  # Skip last 1 month to avoid short-term reversal

# LOW vol and revesal parameters
LOW_VOL_WINDOW= 126
SHORT_TERM_REVERSAL = 5

# Machine learning to assign dynamic weights
ML_SIGNAL_MODE = 'ridge' # none to turn off, xboost, ridge

REBALANCE_FREQ = 'BM'  # Monthly rebalancing

# Sector Neutralization
SECTOR_NEUTRAL = True # True/ False

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