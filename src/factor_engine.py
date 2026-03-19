import pandas as pd
import numpy as np
from config import *
# from logger import get_logger


class FACTOR_ENGINE:
    """
    Momentum Factor Implementation
    
    Theory:
    - Momentum: Stocks that have performed well in the past tend to continue
      performing well in the short-to-medium term (Jegadeesh & Titman, 1993)
    
    - 12-1 Month Momentum: Calculate return from t-252 to t-21
      Skip last month (t-21 to t) to avoid short-term reversal effect (AQR 2013)
    
    Mathematical Definition:
    Momentum_t = (Price_{t-21} / Price_{t-252}) - 1
    
    Implementation Notes:
    - Use vectorized operations (numpy/pandas) instead of loops
    - Handle NaN values properly (newly listed stocks)
    - Normalize factors for cross-sectional comparison
    """
    
    def __init__(self):
        """
        To accomendate multi-factors construction, the paramters for momentum & skip period will
        be moved to momentum parameters.

        Parameters:
        -----------
        lookback : int
            Total lookback period in days (default: 252 = 12 months)
        skip_period : int
            Period to skip from current date (default: 21 = 1 month)
        """
        # self.lookback = lookback
        # self.skip_period = skip_period
        # add validation logs for mid-calculation steps
        # self.logger = get_logger(__name__)
        pass

    def calculate_momentum(self, prices, skip_period=MOMENTUM_SKIP, lookback=MOMENTUM_LOOKBACK):
        """
        Calculate momentum factor using vectorized operations
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Adjusted close prices (dates x stocks)
        
        Returns:
        --------
        momentum : pd.DataFrame
            Momentum scores (dates x stocks)
        
        Formula:
        --------
        momentum_t = (price_{t-skip} / price_{t-lookback}) - 1
        
        Vectorization advantage:
        - Process all stocks simultaneously (no loops)
        - Leverage numpy's optimized C backend
        - 100x-1000x faster than Python loops for large datasets
        """
        
        print("\nCalculating Momentum Factor...")
        print(f"Lookback period: {lookback} days ({lookback/21:.1f} months)")
        print(f"Skip period: {skip_period} days ({skip_period/21:.1f} months)")
        
        # Vectorized calculation
        # shift(skip_period) gets price from skip_period days ago
        # shift(lookback) gets price from lookback days ago
        
        price_recent = prices.shift(skip_period)
        price_old = prices.shift(lookback)
        momentum = (price_recent / price_old) - 1

        return momentum
    
    def calculate_rolling_volatility(self, prices, window=60):
        """
        Calculate rolling volatility (optional factor for multi-factor models)
        
        This can be used as:
        1. A separate factor (Low Vol anomaly)
        2. Risk control (adjust position sizes by inverse volatility)
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        return volatility
    
    # add low_vol as a new factor
    def calculate_low_vol(self, prices, window = LOW_VOL_WINDOW):
        '''
        Directly use the above rolling vol function to calculate low vol factor
        '''
        low_vol = self.calculate_rolling_volatility(prices, window)
        return -low_vol
    
    # add short term reversal as a new factor
    def calculate_short_term_reversal(self, prices, window = SHORT_TERM_REVERSAL):
        reversal = - ((prices/ prices.shift(window)) - 1)
        return reversal
    
    def winsorize_factor(self, factor, lower=0.01, upper=0.99):
        """
        Winsorize extreme values to reduce outlier impact
        Cap values at percentiles to prevent extreme outliers from
        dominating the signal
        """

        lower_bound = factor.quantile(lower, axis=1)
        upper_bound = factor.quantile(upper, axis=1)
        
        winsorized = factor.clip(
            lower=lower_bound.values[:, np.newaxis],
            upper=upper_bound.values[:, np.newaxis],
            axis=0
        )
        
        return winsorized
    
    def normalize_factor(self, factor, method=NORMALIZE_METHOD):
        """
        Normalize factor values for cross-sectional comparison
        
        Methods:
        - 'z-score': Standardize to mean=0, std=1
        - 'rank': Convert to percentile ranks [0, 1]
        - 'min-max': Scale to [0, 1]
        
        Why normalize?
        - Makes factor values comparable across different time periods
        - Reduces impact of regime changes in volatility
        - Industry standard for factor combination
        """

        if method == 'z-score':
            # Cross-sectional z-score (for each date)
            mean = factor.mean(axis=1)
            std = factor.std(axis=1)
            normalized = factor.sub(mean, axis=0).div(std, axis=0)
            
        elif method == 'rank':
            # Convert to percentile ranks
            normalized = factor.rank(axis=1, pct=True)
            
        elif method == 'min-max':
            # Scale to [0, 1]
            min_val = factor.min(axis=1)
            max_val = factor.max(axis=1)
            normalized = factor.sub(min_val, axis=0).div(max_val - min_val, axis=0)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def sector_neutralize(self, factor, sector_map):
        """
        Remove industry bias by z-scoring within each GICS sector.
 
        Parameters
        ----------
        factor     : pd.DataFrame  (dates and tickers)
        sector_map : dict or pd.Series  {ticker into sector_string}
 
        Returns
        -------
        neutralized : pd.DataFrame  same shape as factor
        """
        sectors    = pd.Series(sector_map)
        neutralized = factor.copy()

        for sector in sectors.unique():
            tickers = sectors[sectors == sector].index.intersection(factor.columns)
            if len(tickers) < 3:          # skip tiny sectors
                continue
            sec = factor[tickers]
            sec_mean = sec.mean(axis=1)
            sec_std  = sec.std(axis=1).replace(0, np.nan)
            neutralized[tickers] = sec.sub(sec_mean, axis=0).div(sec_std, axis=0)
 
        return neutralized
    
    
    def process_factor(self, raw, method=NORMALIZE_METHOD,sector_map=None):
        processed = self.normalize_factor(self.winsorize_factor(raw), method)
        if sector_map is not None:
            processed = self.sector_neutralize(processed, sector_map)
        return processed
    
    
    def combine_factors(self, prices, method = NORMALIZE_METHOD, sector_map=None):
        '''
        combine factors: each factors will be winsorized, normalized and assign weights
        the weights can be assigned by equal weight or using machine learning: can be chosen in config.py
        then the composite factors have to be normalize again
        '''

        # avoid for loop to shorten processing time
        momentum_n = self.process_factor(self.calculate_momentum(prices), method, sector_map)
        low_vol_n = self.process_factor(self.calculate_low_vol(prices), method, sector_map)
        reversal_n = self.process_factor(self.calculate_short_term_reversal(prices), method, sector_map)

        if FACTOR_EQUAL_WEIGHT:
            weight = 1 / FACTOR_NUM
            factor_weights = {i: weight for i in FACTOR_LIST}
            
        else:
            pass # add ML logic for dynamic weight

        composite = (factor_weights['momentum'] * momentum_n 
        + factor_weights['low_vol'] * low_vol_n
        + factor_weights['reversal'] * reversal_n)

        composite = self.normalize_factor(composite, NORMALIZE_METHOD)
        
        total = sum(factor_weights.values())
        print(f"\n  Weights in momentum: {factor_weights['momentum']/total:.1%}  "
              f"low_vol: {factor_weights['low_vol']/total:.1%}  "
              f"reversal: {factor_weights['reversal']/total:.1%}")
 
        return composite, {
            'momentum': momentum_n,
            'low_vol':  low_vol_n,
            'reversal': reversal_n,
        }


    def generate_signals(self, factor, n_quantiles=N_QUANTILES):
        """
        Generate trading signals by ranking stocks into quantiles
        label all stocks by momentum scores and rank them by the scores, 1-5 short 1 and long 5
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor scores
        n_quantiles : int
            Number of quantiles to divide stocks into
        
        Returns:
        --------
        signals : pd.DataFrame
            Quantile assignments (1 = bottom, n_quantiles = top)
        
        Process:
        1. For each date, rank all stocks by factor value
        2. Divide into n_quantiles groups
        3. Assign labels 1 (worst) to n_quantiles (best)
        """
        
        print(f"\nGenerating signals...")
        print(f"Number of quantiles: {n_quantiles}")
        
        # Create empty signals DataFrame
        signals = pd.DataFrame(index=factor.index, columns=factor.columns)
        
        # Generate signals for each date
        for date in factor.index:
            row = factor.loc[date].dropna()
            skip = 0
            
            if len(row) < n_quantiles:
                skip += 1
                continue
            
            try:
                ranked = pd.qcut(row, q=n_quantiles, duplicates='drop', labels=False) + 1
                signals.loc[date, ranked.index] = ranked
            except ValueError:
                # If qcut fails due to duplicates, use rank-based approach
                # This ensures we always get n_quantiles even with duplicate values
                ranked = pd.qcut(row.rank(method='first'), q=n_quantiles, labels=False) + 1
                signals.loc[date, ranked.index] = ranked
        
        signals = signals.astype(float)
        if skip >0:
            print(f"Skipped {skip} dates due to insufficient stocks (< {n_quantiles} required)")

        print("\nSignal distribution (last date with data):")
        last_valid = signals.dropna(how='all').iloc[-1]
        if len(last_valid.dropna()) > 0:
            print(last_valid.value_counts().sort_index())

        unique_signals = last_valid.dropna().unique()
        print(f"\nUnique quantiles generated: {sorted(unique_signals)}")
        if len(unique_signals) < n_quantiles:
            print(f"WARNING: Only {len(unique_signals)} quantiles instead of {n_quantiles}")
        else:
            print(f"All {n_quantiles} quantiles successfully generated")
        return signals
    
    def get_rebalanced_df(self, signal, rebalance_freq):
        new_signal_df = signal.resample('BM').last()
        new_signal_df = new_signal_df.dropna()

        return new_signal_df


    def create_long_short_portfolio(self, signals, prices, 
                                   long_quantile=LONG_QUANTILE,
                                   short_quantile=SHORT_QUANTILE):
        """
        Construct Long-Short portfolio based on signals
        
        Strategy:
        - Long: Top quantile (highest momentum)
        - Short: Bottom quantile (lowest momentum)
        - Equal-weighted within each leg
        - MONTHLY rebalancing at month-end
        
        Returns:
        --------
        portfolio_returns : pd.Series
            Daily returns of the Long-Short portfolio
        positions : pd.DataFrame
            Position weights (-1, 0, or +1)
        """
        '''
        self.logger.info(
            f"Constructing Long-Short Portfolio | "
            f"long_q={long_quantile} | short_q={short_quantile} | rebal={REBALANCE_FREQ}"
        )
        '''

        
        print(f"\nConstructing Long-Short Portfolio...")
        print(f"Long: Quantile {long_quantile}")
        print(f"Short: Quantile {short_quantile}")
        print(f"Rebalancing: MONTHLY (end of month)")
    
        # 1.rebalance the signals
        # Create position matrix (all zeros initially)
        rebalanced_signals = self.get_rebalanced_df(signals, REBALANCE_FREQ)

        # 2. create long/short masks
        # long and short weight is 1/N N is number stocks in long/short
        long_mask = (rebalanced_signals == long_quantile)
        short_mask = (rebalanced_signals == short_quantile)

        # 3. create long/short weight
        long_counts = long_mask.sum(axis=1) 
        short_counts = short_mask.sum(axis=1)

        # 130/30: long = 130%, short = 30% or 100% short 100%
        long_weights = (LONG_EXPOSURE / long_counts).replace([np.inf, -np.inf], 0).fillna(0) # series
        short_weights = (-SHORT_EXPOSURE / short_counts).replace([np.inf, -np.inf], 0).fillna(0)# series

        # calculate the long and short position by multiplying 
        # the long mask and short mask dataframe with the long/short weights
        pos_long = long_mask.mul(long_weights, axis=0)
        pos_short = short_mask.mul(short_weights, axis=0)
        monthly_positions = pos_long + pos_short
        
        daily_positions = monthly_positions.reindex(prices.index, method = "ffill")
        daily_positions = daily_positions.shift(1) # because position today will be used to change position for tomorrow.
        daily_returns = prices.pct_change()

        # Account for transaction costs
        # Calculate turnover (sum of absolute position changes)
        turnover = monthly_positions.diff().abs().sum(axis=1)
        turnover.iloc[0] = monthly_positions.iloc[0].abs().sum()

        monthly_tc = turnover * TRANSACTION_COST
        daily_tc = monthly_tc.reindex(index=prices.index, fill_value=0).shift(1)

        portfolio_returns_gross = (daily_positions * daily_returns).sum(axis=1)
        portfolio_returns_net = portfolio_returns_gross - daily_tc

        print(f"\nTotal rebalancing dates: {len(monthly_positions)}")

        # Portfolio return = sum of (position * return) for each stock
        # No need to shift positions since we already set them to start NEXT day

        portfolio_returns_gross.to_csv(f'{DATA_DIR}/portfolio_returns.csv')
        portfolio_returns_net.to_csv(f'{DATA_DIR}/portfolio_returns_net.csv')

        return portfolio_returns_net, daily_positions
    
    # -------------------------------------------------------------------------
    # Private diagnostic helpers (DEBUG level — invisible in production)
    # -------------------------------------------------------------------------

    '''
    def _log_mask_diagnostics(
        self,
        long_mask: pd.DataFrame,
        short_mask: pd.DataFrame,
        long_weights: pd.Series,
        short_weights: pd.Series,
    ) -> None:
        """
        Inspect masks and weights BEFORE multiplication — mirrors the original
        commented-out debug block. Runs at DEBUG level only.
        """
        # --- long_mask structure ---
        self.logger.debug(
            f"long_mask | shape={long_mask.shape} | "
            f"dtypes={long_mask.dtypes.unique().tolist()} | "
            f"true_count={long_mask.sum().sum()} | "
            f"false_count={(~long_mask).sum().sum()}\n"
            f"  index[:3]={long_mask.index[:3].tolist()} | "
            f"  cols[:3]={long_mask.columns[:3].tolist()}\n"
            f"  sample (2x3):\n{long_mask.iloc[:2, :3].to_string()}"
        )
        self.logger.debug(
            f"short_mask | shape={short_mask.shape} | "
            f"true_count={short_mask.sum().sum()}"
        )

        # --- long_weights structure ---
        self.logger.debug(
            f"long_weights | shape={long_weights.shape} | dtype={long_weights.dtype} | "
            f"mean={long_weights.mean():.6f} | nan_count={long_weights.isna().sum()}\n"
            f"  sample:\n{long_weights.head(3).to_string()}"
        )
        self.logger.debug(
            f"short_weights | mean={short_weights.mean():.6f} | "
            f"nan_count={short_weights.isna().sum()}"
        )

        # --- Index alignment ---
        long_aligned = long_mask.index.equals(long_weights.index)
        short_aligned = short_mask.index.equals(short_weights.index)
        if not long_aligned:
            self.logger.warning(
                f"Index mismatch: long_mask vs long_weights\n"
                f"  long_mask[:3]:    {long_mask.index[:3].tolist()}\n"
                f"  long_weights[:3]: {long_weights.index[:3].tolist()}"
            )
        else:
            self.logger.debug("Index alignment check passed — long_mask & long_weights aligned")

        if not short_aligned:
            self.logger.warning(
                f"Index mismatch: short_mask vs short_weights\n"
                f"  short_mask[:3]:    {short_mask.index[:3].tolist()}\n"
                f"  short_weights[:3]: {short_weights.index[:3].tolist()}"
            )

        # --- Manual row test (first date) ---
        first_date = long_mask.index[0]
        test_mask = long_mask.loc[first_date]
        test_weight = long_weights.loc[first_date]
        true_stocks = test_mask[test_mask].index[:3]

        manual_rows = "\n".join(
            f"    {stock}: True × {test_weight:.6f} = {test_weight:.6f}"
            for stock in true_stocks
        )
        self.logger.debug(
            f"Manual row test (date={first_date}) | "
            f"weight={test_weight:.6f} | true_count={test_mask.sum()}\n"
            f"{manual_rows}"
        )

    def _log_position_diagnostics(
        self,
        pos_long: pd.DataFrame,
        pos_short: pd.DataFrame,
        monthly_positions: pd.DataFrame,
    ) -> None:
        """
        Inspect positions AFTER multiplication — shapes, overlap, weight sums,
        first-month sample, and actual vs expected weight values.
        """
        # --- Shapes ---
        self.logger.debug(
            f"Post-multiplication shapes | "
            f"pos_long={pos_long.shape} | pos_short={pos_short.shape} | "
            f"monthly_positions={monthly_positions.shape}"
        )

        # --- pos_long health check ---
        self.logger.debug(
            f"pos_long | NaN={pos_long.isna().sum().sum()} | "
            f"non-zero={(pos_long != 0).sum().sum()}\n"
            f"  sample (2x3):\n{pos_long.iloc[:2, :3].to_string()}"
        )

        # --- Overlap check ---
        overlap = ((pos_long != 0) & (pos_short != 0)).sum().sum()
        if overlap > 0:
            self.logger.warning(f"Overlap: {overlap} stocks are simultaneously long and short!")
        else:
            self.logger.debug("Overlap check passed — no stocks are both long and short")

        # --- Weight sums ---
        long_sum  = pos_long.sum(axis=1)
        short_sum = pos_short.sum(axis=1)
        net_sum   = monthly_positions.sum(axis=1)
        self.logger.debug(
            f"Weight sums | "
            f"long_mean={long_sum.mean():+.6f} (std={long_sum.std():.6f}) | "
            f"short_mean={short_sum.mean():+.6f} (std={short_sum.std():.6f}) | "
            f"net_mean={net_sum.mean():+.6f} (std={net_sum.std():.6f})"
        )

        # --- First month sample ---
        first_long  = pos_long.iloc[0][pos_long.iloc[0] > 0]
        first_short = pos_short.iloc[0][pos_short.iloc[0] < 0]
        self.logger.debug(
            f"First month | long_stocks={len(first_long)} | "
            f"short_stocks={len(first_short)} | "
            f"zero_stocks={(monthly_positions.iloc[0] == 0).sum()}"
        )

        # --- Actual vs expected weights ---
        if len(first_long) > 0:
            self.logger.debug(
                f"Long weight | actual={first_long.iloc[0]:.6f} | "
                f"expected={1.0 / len(first_long):.6f} | n={len(first_long)}\n"
                f"  values: {first_long.to_dict()}"
            )
        if len(first_short) > 0:
            self.logger.debug(
                f"Short weight | actual={first_short.iloc[0]:.6f} | "
                f"expected={-1.0 / len(first_short):.6f} | n={len(first_short)}\n"
                f"  values: {first_short.to_dict()}"
            )

    def _log_transaction_cost_diagnostics(
        self,
        turnover: pd.Series,
        monthly_tc: pd.Series,
        daily_tc: pd.Series,
    ) -> None:
        """Log transaction cost diagnostics at DEBUG level."""

        positive_tc = monthly_tc[monthly_tc > 0]

        self.logger.debug(
            f"Transaction costs | "
            f"rate={TRANSACTION_COST} ({TRANSACTION_COST * 10000:.2f} bps) | "
            f"avg_turnover={turnover.mean():.2%} | "
            f"cost_per_rebal={turnover.mean() * TRANSACTION_COST:.4%} | "
            f"annual_cost={turnover.mean() * TRANSACTION_COST * 12:.2%}"
        )

        if len(positive_tc) > 0:
            self.logger.debug(
                f"TC range | max={positive_tc.max():.4%} | min={positive_tc.min():.4%} | "
                f"avg_daily={daily_tc.mean():.4%}"
            )

        '''