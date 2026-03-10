"""
Data Validation and Cleaning Module
Handles missing values, outliers, and corporate action detection
"""

import pandas as pd
import numpy as np
from config import *


class DataValidator:
    """
    Comprehensive data validation and cleaning
    
    Handles:
    1. Missing data detection and imputation
    2. Outlier detection and treatment
    3. Corporate action detection (splits, dividends)
    4. Data quality metrics
    """
    
    def __init__(self):
        self.validation_report = {}
        
    def validate_prices(self, prices, volumes=None):
        """
        Comprehensive price data validation
        
        Returns cleaned prices and validation report
        """
        print("\n" + "="*70)
        print("DATA VALIDATION & CLEANING")
        print("="*70)
        
        original_shape = prices.shape
        self.validation_report['original_shape'] = original_shape
        
        # 1. Check for missing values
        prices_before = prices.columns.tolist()
        prices = self._handle_missing_values(prices)
        
        # have to delete missing data in the volumnes df
        if volumes is not None:
            dropped_stocks = set(prices_before) - set(prices.columns)
            if dropped_stocks:
                print(f"\nSyncing volumes: removing {len(dropped_stocks)} stocks...")
                volumes = volumes.drop(columns=list(dropped_stocks), errors='ignore')
                print(f"Volumes synced: {len(volumes.columns)} stocks")
        
        # 2. Detect and handle outliers
        prices_before = prices.columns.tolist()
        prices = self._handle_outliers(prices)
        
        if volumes is not None:
            dropped_stocks = set(prices_before) - set(prices.columns)
            if dropped_stocks:
                print(f"\nSyncing volumes: removing {len(dropped_stocks)} stocks...")
                volumes = volumes.drop(columns=list(dropped_stocks), errors='ignore')
                print(f" Volumes synced: {len(volumes.columns)} stocks")
        
        
        # 3. Detect corporate actions
        self._detect_corporate_actions(prices)
        
        # 4. Validate price levels
        self._validate_price_levels(prices)
        
        # 5. Check data consistency
        if volumes is not None:
            self._validate_consistency(prices, volumes)

             # 5. Validate consistency (now should be aligned!)
            # need to align
            if not prices.columns.equals(volumes.columns):
                print(f"\nFinal alignment...")
                common_cols = sorted(set(prices.columns) & set(volumes.columns))
                prices = prices[common_cols]
                volumes = volumes[common_cols]
                print(f"Both aligned to {len(common_cols)} stocks")
    
        # 6. Generate summary
        self._print_validation_summary(prices, original_shape)
        
        return prices
    
    def _handle_missing_values(self, prices):
        """
        Handle missing values in price data
        
        Strategy:
        1. Forward fill up to 5 days (assume trading halt)
        2. Drop stocks with >10% missing data
        3. Drop dates with >50% missing data
        """
        print("\n1. Missing Value Analysis")
        print("-" * 60)
        
        # Calculate missing percentages
        missing_by_stock = prices.isna().sum() / len(prices)
        missing_by_date = prices.isna().sum(axis=1) / len(prices.columns)
        
        total_missing = prices.isna().sum().sum()
        total_cells = prices.shape[0] * prices.shape[1]
        missing_pct = total_missing / total_cells
        
        print(f"  Total missing values: {total_missing:,} ({missing_pct:.2%})")
        print(f"  Stocks with >10% missing: {(missing_by_stock > 0.1).sum()}")
        print(f"  Dates with >50% missing: {(missing_by_date > 0.5).sum()}")
        
        # Store original missing info
        self.validation_report['missing_before'] = {
            'total': total_missing,
            'percentage': missing_pct,
            'stocks_gt_10pct': (missing_by_stock > 0.1).sum(),
            'dates_gt_50pct': (missing_by_date > 0.5).sum()
        }
        
        # Drop stocks with excessive missing data
        stocks_to_drop = missing_by_stock[missing_by_stock > 0.1].index
        if len(stocks_to_drop) > 0:
            print(f"\n  Dropping {len(stocks_to_drop)} stocks with >10% missing data:")
            print(f"    {list(stocks_to_drop[:5])}{'...' if len(stocks_to_drop) > 5 else ''}")
            prices = prices.drop(columns=stocks_to_drop)
        
        # Forward fill missing values (max 5 days)
        print(f"\n  Forward filling missing values (max 5 days)...")
        prices = prices.fillna(method='ffill', limit=5)
        
        # Drop dates with excessive missing data
        dates_to_drop = missing_by_date[missing_by_date > 0.5].index
        if len(dates_to_drop) > 0:
            print(f"  Dropping {len(dates_to_drop)} dates with >50% missing data")
            prices = prices.drop(index=dates_to_drop)
        
        # Check remaining missing values
        remaining_missing = prices.isna().sum().sum()
        print(f"\n  Remaining missing values: {remaining_missing:,}")
        
        if remaining_missing > 0:
            # Drop any remaining stocks with NaN
            prices = prices.dropna(axis=1, how='any')
            print(f"  Dropped {len(stocks_to_drop)} more stocks with remaining NaN")
        
        self.validation_report['missing_after'] = {
            'total': prices.isna().sum().sum(),
            'percentage': prices.isna().sum().sum() / (prices.shape[0] * prices.shape[1])
        }
        
        return prices
    
    def _handle_outliers(self, prices):
        """
        Detect and handle extreme price movements
        
        Outliers indicate:
        - Data errors
        - Stock splits not adjusted
        - Extreme corporate events
        """
        print("\n2. Outlier Detection")
        print("-" * 60)
        
        # Calculate daily returns
        returns = prices.pct_change()
        
        # Define outlier thresholds
        extreme_threshold = 0.5  # 50% single-day move
        suspicious_threshold = 0.3  # 30% single-day move
        
        # Detect extreme moves
        extreme_moves = (returns.abs() > extreme_threshold)
        suspicious_moves = (returns.abs() > suspicious_threshold) & (returns.abs() <= extreme_threshold)
        
        n_extreme = extreme_moves.sum().sum()
        n_suspicious = suspicious_moves.sum().sum()
        
        print(f"  Extreme moves (>50%): {n_extreme}")
        print(f"  Suspicious moves (30-50%): {n_suspicious}")
        
        self.validation_report['outliers'] = {
            'extreme_moves': n_extreme,
            'suspicious_moves': n_suspicious
        }
        
        # Report stocks with most outliers
        if n_extreme > 0:
            outlier_counts = extreme_moves.sum().sort_values(ascending=False)
            top_outliers = outlier_counts[outlier_counts > 0].head(5)
            
            print(f"\n  Stocks with most extreme moves:")
            for stock, count in top_outliers.items():
                print(f"    {stock}: {count} moves")
            
            # Flag stocks with >5 extreme moves as suspicious
            suspicious_stocks = outlier_counts[outlier_counts > 5].index
            if len(suspicious_stocks) > 0:
                print(f"\nWarning: {len(suspicious_stocks)} stocks have >5 extreme moves")
                print(f"Consider manual review: {list(suspicious_stocks)}")
        
        # Winsorize extreme returns (cap at +/-50%)
        returns_winsorized = returns.clip(-0.5, 0.5)
        
        # Reconstruct prices from winsorized returns
        if n_extreme > 0:
            print(f"\n  Winsorizing {n_extreme} extreme moves to ±50%")
            prices_clean = prices.iloc[0] * (1 + returns_winsorized).cumprod()
            prices_clean.iloc[0] = prices.iloc[0]  # Preserve first row
            return prices_clean
        
        return prices
    
    def _detect_corporate_actions(self, prices):
        """
        Detect potential corporate actions
        
        Note: yfinance uses Adj Close which should handle splits/dividends,
        but we still check for anomalies
        """
        print("\n3. Corporate Action Detection")
        print("-" * 60)
        
        returns = prices.pct_change()
        
        # Detect potential stock splits (large jumps in absolute price level)
        # Compare price levels across time
        price_ratios = prices / prices.shift(1)
        
        # Common split ratios
        split_ratios = [2.0, 3.0, 0.5, 0.333, 1.5, 0.667]  # 2:1, 3:1, 1:2, 1:3, 3:2, 2:3
        
        split_candidates = []
        for ratio in split_ratios:
            # Allow 5% tolerance
            matches = ((price_ratios > ratio * 0.95) & (price_ratios < ratio * 1.05))
            if matches.any().any():
                for col in matches.columns[matches.any()]:
                    dates = matches.index[matches[col]]
                    for date in dates:
                        split_candidates.append((col, date, price_ratios.loc[date, col]))
        
        if split_candidates:
            print(f"  Potential unadjusted splits detected: {len(split_candidates)}")
            print(f"    (These should already be handled by Adj Close)")
            
            # Show first few
            for stock, date, ratio in split_candidates[:3]:
                print(f"    {stock} on {date}: ratio {ratio:.2f}")
        else:
            print(f" No unadjusted splits detected")
        
        self.validation_report['corporate_actions'] = {
            'split_candidates': len(split_candidates)
        }
        
        # Detect potential dividend events (small negative returns on ex-div date)
        # These should also be in Adj Close, but check anyway
        small_negative_moves = (returns < -0.01) & (returns > -0.05)
        n_potential_divs = small_negative_moves.sum().sum()
        
        print(f"  Potential dividend events: {n_potential_divs}")
        print(f"    (Should be handled by Adj Close)")
    
    def _validate_price_levels(self, prices):
        """
        Validate that price levels are reasonable
        """
        print("\n4. Price Level Validation")
        print("-" * 60)
        
        # Check for negative prices (should never happen)
        negative_prices = (prices < 0).sum().sum()
        if negative_prices > 0:
            print(f"ERROR: {negative_prices} negative prices detected!")
        else:
            print(f"No negative prices")
        
        # Check for zero prices
        zero_prices = (prices == 0).sum().sum()
        if zero_prices > 0:
            print(f"Warning: {zero_prices} zero prices detected")
        else:
            print(f"  ✓ No zero prices")
        
        # Check for penny stocks (< $1)
        penny_stocks = (prices < 1).any()
        n_penny = penny_stocks.sum()
        if n_penny > 0:
            print(f"Warning: {n_penny} stocks traded below $1 at some point")
            print(f"    {list(penny_stocks[penny_stocks].index[:5])}")
        
        # Price statistics
        mean_price = prices.mean().mean()
        median_price = prices.median().median()
        
        print(f"\n  Price Statistics:")
        print(f"    Mean price: ${mean_price:.2f}")
        print(f"    Median price: ${median_price:.2f}")
        print(f"    Min price: ${prices.min().min():.2f}")
        print(f"    Max price: ${prices.max().max():.2f}")
        
        self.validation_report['price_levels'] = {
            'negative_prices': negative_prices,
            'zero_prices': zero_prices,
            'penny_stocks': n_penny,
            'mean_price': mean_price,
            'median_price': median_price
        }


    def _validate_consistency(self, prices, volumes):
        """
        Check consistency between prices and volumes
        Automatically aligns volumes to match prices structure
        
        Returns:
        --------
        volumes_aligned : pd.DataFrame or None
            Aligned volume data, or None if no volumes provided
        """
        print("\n5. Price-Volume Consistency")
        print("-" * 60)
        
        # Check if volumes provided
        if volumes is None:
            print(f"Warning: No volume data provided")
            return None
        
        # ========================================================================
        # 1. Check for suspicious price-volume patterns
        # ========================================================================
        returns = prices.pct_change()
        large_moves = returns.abs() > 0.05
        zero_volume = (volumes == 0)
        
        suspicious = (large_moves & zero_volume)
        n_suspicious = suspicious.sum().sum()
        
        if n_suspicious > 0:
            print(f"Warning: {n_suspicious} large price moves with zero volume")
        else:
            print(f"  ✓ Price-volume patterns consistent")
        
        # ========================================================================
        # 2. Check INDEX alignment (dates)
        # ========================================================================
        if not prices.index.equals(volumes.index):
            print(f"\n Index Mismatch:")
            print(f"    Prices:  {len(prices.index)} dates")
            print(f"    Volumes: {len(volumes.index)} dates")
            
            # Find common dates
            common_dates = prices.index.intersection(volumes.index)
            print(f"Common:  {len(common_dates)} dates")
            
            if len(common_dates) < len(prices.index) * 0.9:
                print(f" ERROR: Less than 90% overlap - data quality issue!")
            else:
                print(f"Will align to common dates")
        else:
            print(f"Indices aligned ({len(prices.index)} dates)")
        
        # ========================================================================
        # 3. Check COLUMN alignment (stocks)
        # ========================================================================
        print(f"\n  Column Alignment Check:")
        print(f"    Prices:  {len(prices.columns)} stocks")
        print(f"    Volumes: {len(volumes.columns)} stocks")
        
        if not prices.columns.equals(volumes.columns):
            price_cols = set(prices.columns)
            volume_cols = set(volumes.columns)
            
            print(f" Columns don't match - Diagnosing...")
            
            # Diagnose: Same stocks or different stocks?
            if price_cols == volume_cols:
                # ===== CASE A: Same stocks, different ORDER =====
                print(f"\n    💡 Diagnosis: SAME {len(price_cols)} stocks, DIFFERENT order")
                print(f"       Prices  first 3: {list(prices.columns[:3])}")
                print(f"       Volumes first 3: {list(volumes.columns[:3])}")
                print(f"       Prices  last 3:  {list(prices.columns[-3:])}")
                print(f"       Volumes last 3:  {list(volumes.columns[-3:])}")
                
                print(f"\n  Fix: Reordering volumes to match prices...")
                volumes = volumes[prices.columns]
                print(f"  Volumes reordered successfully")
                
            else:
                # ===== CASE B: Different stock sets =====
                only_in_prices = price_cols - volume_cols
                only_in_volumes = volume_cols - price_cols
                common_cols = price_cols & volume_cols
                
                print(f"\n Diagnosis: DIFFERENT stock sets")
                print(f"       Common stocks:      {len(common_cols)}")
                print(f"       Only in prices:     {len(only_in_prices)}")
                print(f"       Only in volumes:    {len(only_in_volumes)}")
                
                if only_in_prices:
                    print(f"         Missing in volumes: {list(only_in_prices)[:5]}")
                if only_in_volumes:
                    print(f"         Missing in prices:  {list(only_in_volumes)[:5]}")
                
                if len(common_cols) < len(price_cols) * 0.9:
                    print(f"\nWARNING: Less than 90% overlap!")
                    print(f"       This indicates a data quality issue")
                
                print(f"\n    🔧 Fix: Aligning to {len(common_cols)} common stocks...")
                common_cols_sorted = sorted(common_cols)
                volumes = volumes[common_cols_sorted]
                print(f"Volumes aligned to common stocks")
                print(f"Note: You should also align prices in the calling function")
        else:
            print(f"Columns perfectly aligned")
        
        # ========================================================================
        # 4. Final alignment verification
        # ========================================================================
        print(f"\n  Final Alignment Status:")
        print(f"    Indices match:  {prices.index.equals(volumes.index)}")
        print(f"    Columns match:  {prices.columns.equals(volumes.columns)}")
        
        if prices.index.equals(volumes.index) and prices.columns.equals(volumes.columns):
            print(f" Perfect alignment achieved!")
        elif prices.columns.equals(volumes.columns):
            print(f" Warning: Columns aligned, but indices differ")
            print(f" Warning: Recommend: Use .loc[common_dates] in calling function")
        else:
            print(f"Warning:Columns still differ - further alignment needed")
            print(f"Recommend: Align prices to match volumes in calling function")
        
        return volumes

    
    def _print_validation_summary(self, prices, original_shape):
        """
        Print summary of validation results
        """
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        final_shape = prices.shape
        
        print(f"\nData Shape:")
        print(f"  Original: {original_shape[0]} days × {original_shape[1]} stocks")
        print(f"  Final:    {final_shape[0]} days × {final_shape[1]} stocks")
        print(f"  Dropped:  {original_shape[0] - final_shape[0]} days, "
              f"{original_shape[1] - final_shape[1]} stocks")
        
        print(f"\nData Quality:")
        print(f"  Missing values: {self.validation_report['missing_after']['total']} "
              f"({self.validation_report['missing_after']['percentage']:.2%})")
        print(f"  Outliers handled: {self.validation_report['outliers']['extreme_moves']}")
        print(f"  Price range: ${prices.min().min():.2f} - ${prices.max().max():.2f}")
        
        print(f"\n✓ Data validation complete\n")
    
    def get_validation_report(self):
        return self.validation_report
