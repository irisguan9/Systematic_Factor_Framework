import pandas as pd
import numpy as np
from scipy import stats
from config import *

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies
    
    Key Metrics:
    1. Return metrics: Total, Annualized, CAGR
    2. Risk metrics: Volatility, Downside Deviation, Max Drawdown
    3. Risk-adjusted: Sharpe, Sortino, Calmar
    4. Benchmark comparison: Alpha, Beta, Information Ratio, Correlation
    """
    
    def __init__(self, strategy_returns, benchmark_returns=None):
        """
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily returns of the strategy
        benchmark_returns : pd.Series, optional
            Daily returns of benchmark (e.g., SPY)
        """
        self.returns = strategy_returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.metrics = {}
        
    def calculate_cumulative_returns(self):
        """
        Calculate cumulative returns
        
        Formula: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        """
        cum_returns = (1 + self.returns).cumprod()
        
        self.metrics['cumulative_return'] = cum_returns.iloc[-1] - 1
        self.metrics['total_return'] = self.metrics['cumulative_return']
        
        return cum_returns
    
    def calculate_annualized_return(self):
        """
        Annualized return (CAGR)
        
        Formula: (Final Value / Initial Value) ^ (252 / n_days) - 1
        """
        cum_return = self.metrics.get('cumulative_return')
        if cum_return is None:
            self.calculate_cumulative_returns()
            cum_return = self.metrics['cumulative_return']
        
        n_years = len(self.returns) / TRADING_DAYS_PER_YEAR
        annualized_return = (1 + cum_return) ** (1 / n_years) - 1
        
        self.metrics['annualized_return'] = annualized_return
        self.metrics['cagr'] = annualized_return
        
        return annualized_return
    
    def calculate_volatility(self):
        """
        Annualized volatility (standard deviation)
        
        Formula: std(daily_returns) * sqrt(252)
        """
        daily_vol = self.returns.std()
        annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        self.metrics['volatility'] = annual_vol
        self.metrics['daily_volatility'] = daily_vol
        
        return annual_vol
    
    def calculate_sharpe_ratio(self, risk_free_rate=RISK_FREE_RATE):
        """
        Sharpe Ratio: Risk-adjusted return
        
        Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
        """
        ann_return = self.metrics.get('annualized_return')
        if ann_return is None:
            ann_return = self.calculate_annualized_return()
        
        vol = self.metrics.get('volatility')
        if vol is None:
            vol = self.calculate_volatility()
        
        sharpe = (ann_return - risk_free_rate) / vol
        self.metrics['sharpe_ratio'] = sharpe
        
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate=RISK_FREE_RATE):
        """
        Sortino Ratio: Sharpe ratio using only downside volatility
        
        Better metric than Sharpe for asymmetric return distributions
        Penalizes downside volatility only
        """
        ann_return = self.metrics.get('annualized_return')
        if ann_return is None:
            ann_return = self.calculate_annualized_return()
        
        # Downside deviation (only negative returns)
        downside_returns = self.returns[self.returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        if downside_vol == 0:
            sortino = np.inf
        else:
            sortino = (ann_return - risk_free_rate) / downside_vol
        
        self.metrics['sortino_ratio'] = sortino
        self.metrics['downside_volatility'] = downside_vol
        
        return sortino
    
    def calculate_max_drawdown(self):
        """
        Maximum Drawdown: Largest peak-to-trough decline
        
        This is THE most important metric for allocators
        Shows worst-case scenario for an investor
        
        Formula: 
        - Compute cumulative returns
        - For each point, find previous peak
        - Drawdown = (current - peak) / peak
        - Max DD = minimum drawdown
        """
        cum_returns = (1 + self.returns).cumprod()
        
        # Calculate running maximum (peak)
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown from peak
        drawdown = (cum_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        
        # Find when max drawdown occurred
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_date = running_max[:max_dd_date].idxmax()
        
        self.metrics['max_drawdown'] = max_dd
        self.metrics['max_drawdown_date'] = max_dd_date
        self.metrics['max_drawdown_peak'] = peak_date
        
        # Calculate recovery information
        if max_dd < 0:
            recovery_dates = cum_returns[max_dd_date:][cum_returns[max_dd_date:] >= running_max[max_dd_date]]
            if len(recovery_dates) > 0:
                recovery_date = recovery_dates.index[0]
                recovery_days = (recovery_date - max_dd_date).days
                self.metrics['max_drawdown_recovery_days'] = recovery_days
            else:
                self.metrics['max_drawdown_recovery_days'] = None  # Not yet recovered
        
        return max_dd, drawdown
    
    def calculate_calmar_ratio(self):
        """
        Calmar Ratio: Return / Max Drawdown
        
        Another risk-adjusted metric, popular with hedge funds
        Higher is better
        """
        ann_return = self.metrics.get('annualized_return')
        if ann_return is None:
            ann_return = self.calculate_annualized_return()
        
        max_dd = self.metrics.get('max_drawdown')
        if max_dd is None:
            max_dd, _ = self.calculate_max_drawdown()
        
        if max_dd == 0:
            calmar = np.inf
        else:
            calmar = ann_return / abs(max_dd)
        
        self.metrics['calmar_ratio'] = calmar
        
        return calmar
    
    def calculate_win_rate(self):
        """
        Percentage of positive return days
        """
        win_rate = (self.returns > 0).sum() / len(self.returns)
        self.metrics['win_rate'] = win_rate
        
        return win_rate
    
    def calculate_benchmark_statistics(self):
        """
        Calculate statistics relative to benchmark
        
        Metrics:
        - Alpha: Excess return vs benchmark
        - Beta: Sensitivity to benchmark
        - Correlation: Linear relationship
        - Information Ratio: Alpha / Tracking Error
        """
        if self.benchmark is None:
            return None
        
        # Align dates
        aligned = pd.concat([self.returns, self.benchmark], axis=1, join='inner')
        aligned.columns = ['strategy', 'benchmark']
        
        # 1. Correlation
        correlation = aligned['strategy'].corr(aligned['benchmark'])
        self.metrics['correlation_vs_benchmark'] = aligned['strategy'].corr(aligned['benchmark'])
        
        # 2. Beta (from regression)
        # strategy_return = alpha + beta * benchmark_return + error
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned['benchmark'], aligned['strategy']
        )
        
        beta = slope
        daily_alpha = intercept
        
        self.metrics['beta_vs_benchmark'] = beta
        
        # 3. Annualized Alpha
        # Convert daily alpha to annualized
        annualized_alpha = (1 + daily_alpha) ** TRADING_DAYS_PER_YEAR - 1
        self.metrics['alpha_vs_benchmark'] = annualized_alpha
        
        # 4. Tracking Error
        # Standard deviation of excess returns
        excess_returns = aligned['strategy'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        self.metrics['tracking_error'] = tracking_error
        
        # 5. Information Ratio
        # Alpha / Tracking Error
        if tracking_error > 0:
            ir = annualized_alpha / tracking_error
        else:
            ir = np.inf
        
        self.metrics['information_ratio'] = ir
        
        return {
            'correlation': correlation,
            'beta': beta,
            'alpha': annualized_alpha,
            'tracking_error': tracking_error,
            'information_ratio': ir
        }
    
    def calculate_all_metrics(self):
        """
        Calculate all performance metrics
        """
        print("\nCalculating Performance Metrics...")
        print("="*60)
        
        # Core metrics
        self.calculate_cumulative_returns()
        self.calculate_annualized_return()
        self.calculate_volatility()
        self.calculate_sharpe_ratio()
        self.calculate_sortino_ratio()
        self.calculate_max_drawdown()
        self.calculate_calmar_ratio()
        self.calculate_win_rate()
        
        # Benchmark metrics
        if self.benchmark is not None:
            self.calculate_benchmark_statistics()
        
        return self.metrics
    
    def print_summary(self):
        """
        Print formatted performance summary
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print("\nReturn Metrics:")
        print(f"  Total Return:        {self.metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {self.metrics['annualized_return']:>10.2%}")
        
        print("\nRisk Metrics:")
        print(f"  Volatility (Ann.):   {self.metrics['volatility']:>10.2%}")
        print(f"  Max Drawdown:        {self.metrics['max_drawdown']:>10.2%}")
        if self.metrics.get('max_drawdown_recovery_days'):
            print(f"  DD Recovery (days):  {self.metrics['max_drawdown_recovery_days']:>10.0f}")
        
        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:        {self.metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {self.metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {self.metrics['calmar_ratio']:>10.2f}")
        
        print("\nOther Metrics:")
        print(f"  Win Rate:            {self.metrics['win_rate']:>10.2%}")
        
        if self.benchmark is not None:
            print("\nBenchmark Comparison:")
            print(f"  Correlation:         {self.metrics['correlation_vs_benchmark']:>10.2f}")
            print(f"  Beta:                {self.metrics['beta_vs_benchmark']:>10.2f}")
            print(f"  Alpha (Ann.):        {self.metrics['alpha_vs_benchmark']:>10.2%}")
            print(f"  Information Ratio:   {self.metrics['information_ratio']:>10.2f}")
        
        print("\n" + "="*60)
    
    def get_monthly_returns(self):
        """
        Calculate monthly returns for calendar analysis
        """
        monthly_returns = self.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return monthly_returns
    
    def get_yearly_returns(self):
        """
        Calculate yearly returns
        """
        yearly_returns = self.returns.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return yearly_returns
    
    def get_rolling_sharpe(self, window=252):
        """
        Calculate rolling Sharpe ratio
        
        Useful for understanding strategy performance over time
        """
        rolling_return = self.returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        rolling_sharpe = (rolling_return - RISK_FREE_RATE) / rolling_vol
        
        return rolling_sharpe