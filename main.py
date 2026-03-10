import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('./src')

from src.data_manager import DataManager
from src.factor_engine import MomentumFactor
from src.performance_analyzer import PerformanceAnalyzer
from src.visualizer import PerformanceVisualizer
from config import *

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def main():
    """
    Execute complete strategy pipeline
    """
    
    print("\n" + "="*70)
    print(" MOMENTUM FACTOR STRATEGY - SYSTEMATIC BACKTEST")
    print("="*70)
    print(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Rebalance Frequency: {REBALANCE_FREQ}")
    print(f"Transaction Cost: {TRANSACTION_COST*100:.2f} bps")
    
    # =========================================================================
    # STEP 1: DATA ACQUISITION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA ACQUISITION & QUALITY CONTROL")
    print("="*70)
    
    dm = DataManager(start_date=START_DATE, end_date=END_DATE)
    
    # Check if data already exists
    if os.path.exists(f'{DATA_DIR}/prices_adjusted.csv'):
        print("\nFound existing data. Loading from cache...")
        prices = pd.read_csv(f'{DATA_DIR}/prices_adjusted.csv',
                           index_col=0, parse_dates=True)
        volumes = pd.read_csv(f'{DATA_DIR}/volumes.csv',
                            index_col=0, parse_dates=True)
        print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} stocks")
    else:
        print("\nDownloading fresh data...")
        prices, volumes = dm.prepare_clean_data()
        dm.save_data()
    
    # IMPORTANT: Validate data quality
    # Even cached data should be validated
    from data_validator import DataValidator
    validator = DataValidator()
    prices = validator.validate_prices(prices, volumes)
    
    # Save cleaned data (overwrite cache if needed)
    print("\nSaving validated data...")
    prices.to_csv(f'{DATA_DIR}/prices_adjusted_clean.csv')
    
    # =========================================================================
    # STEP 2: FACTOR CALCULATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: FACTOR CALCULATION")
    print("="*70)
    
    momentum = MomentumFactor(
        lookback=MOMENTUM_LOOKBACK,
        skip_period=MOMENTUM_SKIP
    )
    
    # Calculate momentum factor
    momentum_factor = momentum.calculate_momentum(prices)
    
    print("Winsorizing extreme values (1st-99th percentile)...")
    momentum_winsorized = momentum.winsorize_factor(momentum_factor)

     # Normalize factor (z-score)
    print("\nNormalizing factors (cross-sectional z-score)...")
    momentum_normalized = momentum.normalize_factor(momentum_winsorized, method='z-score')
    
    # Generate trading signals
    signals = momentum.generate_signals(momentum_normalized, n_quantiles=N_QUANTILES)
    
    # =========================================================================
    # STEP 3: Long/short PORTFOLIO CONSTRUCTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: PORTFOLIO CONSTRUCTION")
    print("="*70)
    
    portfolio_returns, positions = momentum.create_long_short_portfolio(
        signals, prices,
        long_quantile=LONG_QUANTILE,
        short_quantile=SHORT_QUANTILE
    )
    
    # =========================================================================
    # STEP 4: BENCHMARK DATA
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: BENCHMARK COMPARISON")
    print("="*70)
    
    print("\nDownloading SPY benchmark...")
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    benchmark_returns = spy_data['Adj Close'].pct_change()
    
    # Align dates
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    aligned.columns = ['Strategy', 'Benchmark']
    
    print(f"Strategy return series: {len(portfolio_returns)} days")
    print(f"Benchmark return series: {len(benchmark_returns)} days")
    print(f"Aligned series: {len(aligned)} days")
    
    # =========================================================================
    # STEP 5: PERFORMANCE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Analyze strategy
    strategy_analyzer = PerformanceAnalyzer(
        aligned['Strategy'],
        aligned['Benchmark']
    )
    strategy_metrics = strategy_analyzer.calculate_all_metrics()
    
    # Analyze benchmark (for comparison)
    benchmark_analyzer = PerformanceAnalyzer(aligned['Benchmark'])
    benchmark_metrics = benchmark_analyzer.calculate_all_metrics()
    
    # Print comparison
    print("\n" + "-"*70)
    print("STRATEGY vs BENCHMARK COMPARISON")
    print("-"*70)
    
    comparison_metrics = [
        ('Total Return', 'total_return'),
        ('Annualized Return', 'annualized_return'),
        ('Volatility', 'volatility'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown', 'max_drawdown'),
        ('Calmar Ratio', 'calmar_ratio'),
        ('Win Rate', 'win_rate')
    ]
    
    print(f"\n{'Metric':<20} {'Strategy':>15} {'Benchmark':>15} {'Difference':>15}")
    print("-"*70)
    
    for name, key in comparison_metrics:
        strat_val = strategy_metrics.get(key, 0)
        bench_val = benchmark_metrics.get(key, 0)
        diff = strat_val - bench_val
        
        if 'ratio' in key.lower() or 'rate' in key.lower():
            print(f"{name:<20} {strat_val:>14.2f} {bench_val:>14.2f} {diff:>+14.2f}")
        else:
            print(f"{name:<20} {strat_val:>14.2%} {bench_val:>14.2%} {diff:>+14.2%}")
    
    print("\nKey Insights:")
    if strategy_metrics['correlation_vs_benchmark'] < 0.3:
        print(f"  ✓ Low correlation ({strategy_metrics['correlation_vs_benchmark']:.2f}) - Good diversification!")
    else:
        print(f"  ⚠ Correlation ({strategy_metrics['correlation_vs_benchmark']:.2f}) - Consider combining with uncorrelated strategies")
    
    if strategy_metrics['sharpe_ratio'] > benchmark_metrics['sharpe_ratio']:
        print(f"  ✓ Higher Sharpe ratio - Better risk-adjusted returns")
    
    if abs(strategy_metrics['max_drawdown']) < abs(benchmark_metrics['max_drawdown']):
        print(f"  ✓ Lower max drawdown - Better downside protection")
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*70)
    
    viz = PerformanceVisualizer(
        aligned['Strategy'],
        aligned['Benchmark']
    )
    
    viz.save_all_charts(output_dir=RESULTS_DIR)
    
    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING RESULTS")
    print("="*70)
    
    # Save data
    momentum_factor.to_csv(f'{DATA_DIR}/momentum_factor.csv')
    signals.to_csv(f'{DATA_DIR}/signals.csv')
    positions.to_csv(f'{DATA_DIR}/positions.csv')
    aligned.to_csv(f'{DATA_DIR}/returns_aligned.csv')
    
    # Save metrics
    pd.Series(strategy_metrics).to_json(f'{RESULTS_DIR}/strategy_metrics.json', indent=2)
    pd.Series(benchmark_metrics).to_json(f'{RESULTS_DIR}/benchmark_metrics.json', indent=2)
    
    # Create summary report
    summary = {
        'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': f"{START_DATE} to {END_DATE}",
        'n_stocks': len(prices.columns),
        'n_days': len(aligned),
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics
    }
    
    pd.Series(summary).to_json(f'{RESULTS_DIR}/backtest_summary.json', indent=2)
    
    print("\nFiles saved:")
    print(f"  Data: {DATA_DIR}/")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Charts: {RESULTS_DIR}/*.png")

    # ========================================================================
    # COMPONENT PERFORMANCE ANALYSIS (MONTHLY REBALANCE)
    # ========================================================================
    print('\n' + "="*70)
    print('COMPONENT PERFORMANCE ANALYSIS (MONTHLY REBALANCE)')
    print("="*70)

    # Helper function to annualize returns
    def annualize_returns(returns):
        """Calculate annualized return using compound method"""
        if len(returns) == 0 or returns.isna().all():
            return 0.0
        # Remove NaN
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return 0.0
        cumulative_return = (1 + returns_clean).prod() - 1
        n_years = len(returns_clean) / 252
        if n_years <= 0:
            return 0.0
        annualized_return = (1 + cumulative_return) ** (1 / n_years)
        return annualized_return

    # Load daily positions (from step 3 - these are already daily with shift(1))
    daily_positions = pd.read_csv(f'{DATA_DIR}/positions.csv', 
                                index_col=0, parse_dates=True)

    # Separate long and short components
    daily_positions_long = daily_positions.copy()
    daily_positions_long[daily_positions_long < 0] = 0  # Keep only long

    daily_positions_short = daily_positions.copy()
    daily_positions_short[daily_positions_short > 0] = 0  # Keep only short

    # Calculate daily returns for each component
    daily_returns = prices.pct_change()

    portfolio_returns_long = (daily_positions_long * daily_returns).sum(axis=1)
    portfolio_returns_short = (daily_positions_short * daily_returns).sum(axis=1)
    portfolio_returns_gross = portfolio_returns_long + portfolio_returns_short

    # Annualize each component
    long_annual = annualize_returns(portfolio_returns_long)
    short_annual = annualize_returns(portfolio_returns_short)
    gross_annual = annualize_returns(portfolio_returns_gross)

    # Load net returns (with transaction costs)
    portfolio_returns_net = pd.read_csv(f'{DATA_DIR}/portfolio_returns_net.csv',
                                    index_col=0, parse_dates=True).squeeze()
    net_annual = annualize_returns(portfolio_returns_net)

    # Calculate transaction cost impact
    tc_impact = gross_annual - net_annual

    print(f"\nBased on MONTHLY rebalancing:")
    print(f"  Long-only annualized:       {long_annual:>8.2%}")
    print(f"  Short-only annualized:      {short_annual:>8.2%}")
    print(f"  Long-short gross:           {gross_annual:>8.2%}")
    print(f"  Transaction cost impact:    {tc_impact:>8.2%}")
    print(f"  Long-short net:             {net_annual:>8.2%}")

    # Verification
    print(f"\nVerification:")
    print(f"  Gross return from components: {long_annual + short_annual:>8.2%}")
    print(f"  Gross return calculated:      {gross_annual:>8.2%}")
    diff = abs((long_annual + short_annual) - gross_annual)
    if diff < 0.001:
        print(f"Math checks out (diff: {diff:.6f})")
    else:
        print(f"Warning: Math doesn't match (diff: {diff:.6f})")

    # ========================================================================
    # WARM-UP PERIOD ANALYSIS
    # ========================================================================
    print('\n' + "="*70)
    print('WARM-UP PERIOD IMPACT')
    print("="*70)

    # Find first date with any positions (non-zero)
    has_positions = (daily_positions != 0).any(axis=1)
    if has_positions.any():
        first_position_date = has_positions[has_positions].index[0]
        
        print(f"\nFirst position date: {first_position_date.strftime('%Y-%m-%d')}")
        
        warmup_days = (first_position_date - pd.to_datetime(START_DATE)).days
        
        if warmup_days > 5:
            print(f"Warm-up period: {START_DATE} to {first_position_date.strftime('%Y-%m-%d')}")
            print(f"Warm-up days: {warmup_days}")
            
            # Calculate SPY performance during warm-up
            warmup_spy = benchmark_returns.loc[:first_position_date]
            warmup_spy_cumulative = (1 + warmup_spy).prod() - 1
            
            # Handle Series vs scalar
            if isinstance(warmup_spy_cumulative, pd.Series):
                warmup_spy_cumulative = warmup_spy_cumulative.iloc[0]
            
            print(f"\nSPY return during warm-up: {warmup_spy_cumulative:>8.2%}")
            print(f"Strategy return during warm-up: {0:>8.2%} (no positions)")
            print(f"Opportunity cost: {-warmup_spy_cumulative:>8.2%}")
            
            # Calculate performance during active period only
            active_returns = portfolio_returns_net.loc[first_position_date:]
            active_cumulative = (1 + active_returns).prod() - 1
            active_n_years = len(active_returns) / 252
            active_annual = (1 + active_cumulative) ** (1 / active_n_years) - 1
            
            print(f"\nActive period only ({first_position_date.strftime('%Y-%m-%d')} onwards):")
            print(f"  Days: {len(active_returns)}")
            print(f"  Total return: {active_cumulative:>8.2%}")
            print(f"  Annualized: {active_annual:>8.2%}")
            
            # Full period comparison
            full_cumulative = (1 + portfolio_returns_net).prod() - 1
            full_n_years = len(portfolio_returns_net) / 252
            full_annual = (1 + full_cumulative) ** (1 / full_n_years) - 1
            
            print(f"\nFull period ({START_DATE} onwards):")
            print(f"  Days: {len(portfolio_returns_net)}")
            print(f"  Total return: {full_cumulative:>8.2%}")
            print(f"  Annualized: {full_annual:>8.2%}")
            
            print(f"\nWarm-up impact on annualized return: {full_annual - active_annual:>8.2%}")
        else:
            print(f"\n No significant warm-up period (only {warmup_days} days)")
            print(f"   Strategy starts trading immediately")
    else:
        print(f"WARNING: No positions found in entire period!")
    # ========================================================================
    # add more checks to understand why short makes negative returns
    # ========================================================================

    print("\n" + "="*70)
    print("DETAILED POSITION ANALYSIS")
    print("="*70)

    # Load data
    positions = pd.read_csv(f'{DATA_DIR}/positions.csv', index_col=0, parse_dates=True)
    daily_returns = prices.pct_change()

    # 1. Check position weights
    print("\n1. POSITION WEIGHT DISTRIBUTION")
    print("-"*70)

    long_weights = positions[positions > 0]
    short_weights = positions[positions < 0]

    print(f"Long positions:")
    print(f"  Mean weight: {long_weights.mean().mean():.6f}")
    print(f"  Sum per day: {positions[positions > 0].sum(axis=1).mean():.6f}")
    print(f"  Count per day: {(positions > 0).sum(axis=1).mean():.1f}")

    print(f"\nShort positions:")
    print(f"  Mean weight: {short_weights.mean().mean():.6f}")
    print(f"  Sum per day: {positions[positions < 0].sum(axis=1).mean():.6f}")
    print(f"  Count per day: {(positions < 0).sum(axis=1).mean():.1f}")

    # 2. Check a specific date
    print("\n2. SAMPLE DATE ANALYSIS (last rebalance date)")
    print("-"*70)

    # Find last rebalance date (where positions change)
    position_changes = positions.diff().abs().sum(axis=1)
    last_rebalance = position_changes[position_changes > 0].index[-1]

    print(f"Last rebalance date: {last_rebalance.strftime('%Y-%m-%d')}")

    pos_on_date = positions.loc[last_rebalance]
    long_pos = pos_on_date[pos_on_date > 0]
    short_pos = pos_on_date[pos_on_date < 0]

    print(f"\nLong positions on {last_rebalance.strftime('%Y-%m-%d')}:")
    print(f"  Count: {len(long_pos)}")
    print(f"  Sum of weights: {long_pos.sum():.6f}")
    print(f"  Individual weights: {long_pos.values[:5]}...")  # Show first 5

    print(f"\nShort positions on {last_rebalance.strftime('%Y-%m-%d')}:")
    print(f"  Count: {len(short_pos)}")
    print(f"  Sum of weights: {short_pos.sum():.6f}")
    print(f"  Individual weights: {short_pos.values[:5]}...")

    # 3. Verify portfolio construction math
    print("\n3. PORTFOLIO MATH VERIFICATION")
    print("-"*70)

    # Manually calculate returns for one day
    test_date = last_rebalance + pd.Timedelta(days=1)
    if test_date in daily_returns.index:
        pos_test = positions.loc[test_date]
        ret_test = daily_returns.loc[test_date]
        
        # Long contribution
        long_contrib = (pos_test[pos_test > 0] * ret_test[pos_test > 0]).sum()
        # Short contribution
        short_contrib = (pos_test[pos_test < 0] * ret_test[pos_test < 0]).sum()
        # Total
        total_contrib = long_contrib + short_contrib
        
        print(f"Test date: {test_date.strftime('%Y-%m-%d')}")
        print(f"Long contribution: {long_contrib:.6f}")
        print(f"Short contribution: {short_contrib:.6f}")
        print(f"Total return: {total_contrib:.6f}")
        
        # Load portfolio returns and compare
        portfolio_returns = pd.read_csv(f'{DATA_DIR}/portfolio_returns.csv',
                                    index_col=0, parse_dates=True).squeeze()
        actual_return = portfolio_returns.loc[test_date]
        print(f"Actual portfolio return: {actual_return:.6f}")
        print(f"Difference: {abs(total_contrib - actual_return):.10f}")

    # 4. Check if weights sum correctly
    print("\n4. POSITION WEIGHT BALANCE CHECK")
    print("-"*70)

    for i, date in enumerate(positions.index[::100]):  # Check every 100 days
        long_sum = positions.loc[date][positions.loc[date] > 0].sum()
        short_sum = positions.loc[date][positions.loc[date] < 0].sum()
        
        print(f"{date.strftime('%Y-%m-%d')}: Long sum = {long_sum:>8.4f}, "
            f"Short sum = {short_sum:>8.4f}, Net = {long_sum + short_sum:>8.4f}")
        
        if i >= 4:  # Show first 5 samples
            break

    print("\nExpected behavior:")
    print("  Long sum should be ≈ +1.0 (100% long)")
    print("  Short sum should be ≈ -1.0 (100% short)")
    print("  Net should be ≈ 0.0 (dollar neutral)")

    print("\n" + "="*70)
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" BACKTEST COMPLETE")
    print("="*70)
    
    strategy_analyzer.print_summary()
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Review charts in results/ directory")
    print("2. Examine comprehensive_report.png for overview")
    print("3. Check backtest_summary.json for all metrics")
    print("4. Run notebooks/sensitivity_decay_analysis.ipynb\n"
      "and notebooks/analysis.ipynb for deep dive analysis")
    
    return strategy_metrics, benchmark_metrics


if __name__ == '__main__':
    strategy_metrics, benchmark_metrics = main()
