import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import requests
from io import StringIO
import os

from config import *


class DataManager:
    """
    Manages financial data acquisition and quality control
    
    Key considerations:
    1. Survivorship Bias: Free data from yfinance includes current constituents only.
       Ideally, we need Point-in-Time (PIT) data showing historical index composition.
       Mitigation: Filter stocks with insufficient history, document the limitation.
    
    2. Corporate Actions: yfinance's 'Adj Close' handles splits and dividends.
       Always use adjusted prices for return calculations.
    
    3. Data Quality: Check for outliers, missing data, and suspended trading.
    """
    
    def __init__(self, start_date=START_DATE, end_date=END_DATE):
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
        self.volume_data = None
        self.tickers = []
        
    def get_sp500_tickers(self):
        """
        Fetch current S&P 500 constituents
        
        NOTE: This introduces survivorship bias as we're using current constituents
        for historical analysis. In production, use a PIT database like Bloomberg/FactSet.
        """
        print("Fetching S&P 500 constituents...")
        
        # Download SPY holdings as proxy for S&P 500
        # Alternative: scrape from Wikipedia or use a static list
        try:
            # Use a static list for reliability (you can update this)

            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            }
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            sp500 = tables[0]
            tickers = sp500['Symbol'].tolist()
            
            print(f"Found {len(tickers)} tickers")
            return tickers
            
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []
    
    def download_data(self, tickers):
        """
        Download historical data with error handling
        """
        print(f"\nDownloading data for {len(tickers)} tickers...")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        # Download data in batches to avoid timeout
        batch_size = 50
        all_data = {}
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            try:
                data = yf.download(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    threads=True,
                    auto_adjust=False
                )

                if data.empty:
                    print(f"Empty data returned")

                # Store in dictionary
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            # Single ticker download - no MultiIndex
                            all_data[ticker] = data
                        else:
                            if isinstance(data.columns, pd.MultiIndex):
                                tickers_in_data = data.columns.get_level_values(1).unique()
                                if ticker not in tickers_in_data:
                                    print(f" Warning:  {ticker} not in downloaded data")
                                    continue
                                
                                # Extract this ticker's data
                                ticker_data = data.xs(ticker, level=1, axis=1)
                                all_data[ticker] = ticker_data
                            else:
                                if ticker in data.columns:
                                    all_data[ticker] = data[ticker].to_frame()
                                else:
                                    print(f"Warning: {ticker} not found in columns")
                        
                    except Exception as e:
                        print(f"Error extracting {ticker}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"  Error in batch: {e}")
                continue
        
        return all_data
    
    def check_data_quality(self, data, ticker):
        """
        Quality control checks for individual stock data
        
        Checks:
        1. Minimum data points (avoid recently listed stocks)
        2. Price outliers (potential bad data or extreme events)
        3. Volume spikes (potential delisting or corporate actions)
        4. Split detection (large price jumps)
        """
        if data is None or len(data) < MIN_DATA_DAYS:
            return False, "Insufficient data"
        
        try:
            # Check for adjusted close
            if 'Adj Close' not in data.columns:
                return False, "Missing Adj Close"
            
            # Check for minimum price (filter penny stocks)
            avg_price = data['Adj Close'].mean()
            if avg_price < MIN_PRICE:
                return False, f"Low price (${avg_price:.2f})"
            
            # Check for minimum volume
            avg_volume = data['Volume'].mean()
            if avg_volume < MIN_VOLUME:
                return False, f"Low volume ({avg_volume:.0f})"
            
            # Detect potential data errors (>50% single-day move)
            returns = data['Adj Close'].pct_change()
            extreme_moves = returns[abs(returns) > 0.5]
            if len(extreme_moves) > 2:  # Allow some flexibility for splits
                return False, f"Too many extreme moves ({len(extreme_moves)})"
            
            # Check for too many missing days
            missing_pct = data['Adj Close'].isna().sum() / len(data)
            if missing_pct > 0.1:
                return False, f"Missing data ({missing_pct:.1%})"
            
            return True, "Pass"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def detect_delistings(self, data, ticker):
        """
        Attempt to detect potential delistings
        
        NOTE: This is a heuristic approach. True delisting detection requires
        corporate action databases.
        
        Signals:
        - Sudden drop in volume to zero
        - Large price drop with no recovery
        - Data suddenly stops before end_date
        """
        try:
            # Check if data ends prematurely
            last_date = data.index[-1]
            end_date = pd.to_datetime(self.end_date)
            
            days_missing = (end_date - last_date).days
            if days_missing > 30:  # More than 1 month missing
                return True, f"Data ends {days_missing} days early"
            
            # Check for volume collapse in last 20 days
            recent_volume = data['Volume'].tail(20)
            if (recent_volume == 0).sum() > 10:
                return True, "Volume collapse"
            
            return False, "Active"
            
        except:
            return False, "Unknown"
    
    
    def prepare_clean_data(self):
        """
        Main pipeline: download and clean data
        """
        self.tickers = self.get_sp500_tickers()
        
        raw_data = self.download_data(self.tickers)
        
        print("\n" + "="*60)
        print("DATA QUALITY CONTROL")
        print("="*60)
        
        clean_tickers = []
        rejected = {
            'insufficient_data': [],
            'low_price': [],
            'low_volume': [],
            'data_quality': [],
            'potential_delisting': []
        }
        
        for ticker in self.tickers:
            if ticker not in raw_data:
                rejected['insufficient_data'].append(ticker)
                continue
            
            data = raw_data[ticker]
            
            # Quality check
            passed, reason = self.check_data_quality(data, ticker)
            if not passed:
                if 'Insufficient' in reason:
                    rejected['insufficient_data'].append(ticker)
                elif 'Low price' in reason:
                    rejected['low_price'].append(ticker)
                elif 'Low volume' in reason:
                    rejected['low_volume'].append(ticker)
                else:
                    rejected['data_quality'].append(ticker)
                continue
            
            # Delisting check
            is_delisted, delisting_reason = self.detect_delistings(data, ticker)
            if is_delisted:
                rejected['potential_delisting'].append(ticker)
                continue
            
            clean_tickers.append(ticker)
        
        print(f"\nClean tickers: {len(clean_tickers)}")
        for reason, tickers in rejected.items():
            if tickers:
                print(f"  {reason}: {len(tickers)} tickers")
        
        print("\nBuilding cleaned dataset...")
        print(f"Processing {len(clean_tickers)} clean tickers...")
        
        price_df = pd.DataFrame()
        volume_df = pd.DataFrame()
        
        failed_tickers = []
        for ticker in clean_tickers:
            try:
                data = raw_data[ticker]
                
                if not hasattr(data, 'columns'):
                    print(f" Warning: {ticker}: Data is not a DataFrame, type={type(data)}")
                    failed_tickers.append(ticker)
                    continue
                
                if 'Adj Close' not in data.columns:
                    print(f"Warning: {ticker}: Missing 'Adj Close' column. Available: {data.columns.tolist()[:5]}")
                    failed_tickers.append(ticker)
                    continue
                
                price_df[ticker] = data['Adj Close']
                volume_df[ticker] = data['Volume']
                
            except Exception as e:
                print(f" Warning: {ticker}: Error - {str(e)}")
                failed_tickers.append(ticker)
                continue
        
        print(f"\nSuccessfully processed: {len(price_df.columns)} stocks")
        if failed_tickers:
            print(f"Failed to process: {len(failed_tickers)} stocks")
            if len(failed_tickers) <= 10:
                print(f"  Failed tickers: {failed_tickers}")
        
        if len(price_df.columns) == 0:
            print("\nWarning: CRITICAL ERROR: No stocks successfully processed!")
            print("\nDebugging info:")
            print(f"  Total tickers fetched: {len(self.tickers)}")
            print(f"  Clean tickers after QC: {len(clean_tickers)}")
            print(f"  Successfully processed: 0")
            
            if clean_tickers and clean_tickers[0] in raw_data:
                sample_ticker = clean_tickers[0]
                sample_data = raw_data[sample_ticker]
                print(f"\nSample data structure for {sample_ticker}:")
                print(f"  Type: {type(sample_data)}")
                if hasattr(sample_data, 'columns'):
                    print(f"  Columns: {sample_data.columns.tolist()}")
                if hasattr(sample_data, 'shape'):
                    print(f"  Shape: {sample_data.shape}")
            
            return None, None
        
        price_df = price_df.fillna(method='ffill', limit=3)
        volume_df = volume_df.fillna(0)
        
        self.price_data = price_df
        self.volume_data = volume_df
        self.tickers = list(price_df.columns)
        
        print(f"\nFinal universe: {len(self.tickers)} stocks")
        if len(price_df) > 0:
            print(f"Date range: {price_df.index[0]} to {price_df.index[-1]}")
        else:
            print("Warning: price_df is empty!")
            return None, None
        
        print(f"Total observations: {len(price_df):,}")
        return price_df, volume_df
    
    def save_data(self):
        """Save cleaned data to disk"""
        print("\nSaving data...")

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        else:
            print("Using Existing Dir {DATA_DIR}")

        
        self.price_data.to_csv(f'{DATA_DIR}/prices_adjusted.csv')
        self.volume_data.to_csv(f'{DATA_DIR}/volumes.csv')
        
        # Save metadata
        metadata = {
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_stocks': len(self.tickers),
            'n_observations': len(self.price_data),
            'tickers': self.tickers
        }
        
        pd.Series(metadata).to_json(f'{DATA_DIR}/metadata.json')
        print(f"Data saved to {DATA_DIR}/")
