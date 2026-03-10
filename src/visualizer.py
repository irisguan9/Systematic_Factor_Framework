import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')

from config import *

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceVisualizer:
    """
    Create publication-quality performance visualizations
    """
    
    def __init__(self, strategy_returns, benchmark_returns=None):
        """
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily returns of strategy
        benchmark_returns : pd.Series, optional
            Daily returns of benchmark
        """
        self.returns = strategy_returns
        self.benchmark = benchmark_returns
        
        # Calculate cumulative returns
        self.cum_returns = (1 + self.returns).cumprod()
        if self.benchmark is not None:
            self.cum_benchmark = (1 + self.benchmark).cumprod()
    
    def plot_cumulative_returns(self, figsize=(12, 6)):
        """
        Plot cumulative return curves
        
        This is THE most important chart for investors
        Shows wealth growth over time
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strategy
        ax.plot(self.cum_returns.index, self.cum_returns.values,
                label='Momentum Strategy', linewidth=2, color='#2E86AB')
        
        # Plot benchmark if available
        if self.benchmark is not None:
            ax.plot(self.cum_benchmark.index, self.cum_benchmark.values,
                    label='SPY Benchmark', linewidth=2, color='#A23B72',
                    linestyle='--', alpha=0.7)
        
        ax.set_title('Cumulative Returns: Momentum Long-Short Strategy',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1)*100:.0f}%'))
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, figsize=(12, 5)):
        """
        Plot drawdown chart
        
        Shows risk perspective: how much you lose from peak
        """
        # Calculate drawdown
        running_max = self.cum_returns.expanding().max()
        drawdown = (self.cum_returns - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Fill area under drawdown curve
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values,
                color='darkred', linewidth=1.5)
        
        # Mark max drawdown
        max_dd_date = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.scatter([max_dd_date], [max_dd_value],
                  color='red', s=100, zorder=5,
                  label=f'Max DD: {max_dd_value:.2%}')
        
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown from Peak', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns_heatmap(self, figsize=(12, 6)):
        """
        Monthly returns heatmap
        
        Shows seasonality and consistency
        Popular with hedge fund investors
        """
        # Calculate monthly returns
        monthly_returns = self.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Reshape to year x month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        }).pivot(index='Year', columns='Month', values='Return')
        
        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[m-1] for m in pivot_table.columns]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot_table * 100, annot=True, fmt='.1f',
                   cmap='RdYlGn', center=0, cbar_kws={'label': 'Return (%)'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_sharpe(self, window=252, figsize=(12, 5)):
        """
        Plot rolling Sharpe ratio
        
        Shows strategy stability over time
        """
        # Calculate rolling metrics
        rolling_return = self.returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_sharpe = (rolling_return - RISK_FREE_RATE) / rolling_vol
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                linewidth=2, color='#2E86AB')
        
        # Add horizontal reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 2')
        
        ax.set_title(f'Rolling Sharpe Ratio ({window}-day window)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_return_distribution(self, figsize=(12, 5)):
        """
        Plot return distribution
        
        Shows normality (or fat tails)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(self.returns.dropna() * 100, bins=50,
                alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Daily Return (%)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot (test for normality)
        from scipy import stats
        stats.probplot(self.returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_vs_benchmark(self, window=60, figsize=(12, 5)):
        """
        Plot rolling correlation with benchmark
        
        Key metric for allocators: low correlation = diversification
        """
        if self.benchmark is None:
            print("No benchmark provided")
            return None
        
        # Align returns
        aligned = pd.concat([self.returns, self.benchmark], axis=1, join='inner')
        aligned.columns = ['Strategy', 'Benchmark']
        
        # Calculate rolling correlation
        rolling_corr = aligned['Strategy'].rolling(window).corr(aligned['Benchmark'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(rolling_corr.index, rolling_corr.values,
                linewidth=2, color='#2E86AB')
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1,
                  alpha=0.5, label='Low Correlation (0.3)')
        
        ax.set_title(f'Rolling Correlation with SPY ({window}-day window)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(self, save_path=None):
        """
        Create a comprehensive performance report with all charts
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.cum_returns.index, self.cum_returns.values,
                label='Momentum Strategy', linewidth=2, color='#2E86AB')
        if self.benchmark is not None:
            ax1.plot(self.cum_benchmark.index, self.cum_benchmark.values,
                    label='SPY Benchmark', linewidth=2, color='#A23B72',
                    linestyle='--', alpha=0.7)
        ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1)*100:.0f}%'))
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        running_max = self.cum_returns.expanding().max()
        drawdown = (self.cum_returns - running_max) / running_max
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        
        # 3. Rolling Sharpe
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_return = self.returns.rolling(252).mean() * TRADING_DAYS_PER_YEAR
        rolling_vol = self.returns.rolling(252).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_sharpe = (rolling_return - RISK_FREE_RATE) / rolling_vol
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='#2E86AB')
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling Sharpe Ratio (1Y)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(self.returns.dropna() * 100, bins=50,
                alpha=0.7, color='#2E86AB', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Daily Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlation with Benchmark
        if self.benchmark is not None:
            ax5 = fig.add_subplot(gs[2, 1])
            aligned = pd.concat([self.returns, self.benchmark], axis=1, join='inner')
            aligned.columns = ['Strategy', 'Benchmark']
            rolling_corr = aligned['Strategy'].rolling(60).corr(aligned['Benchmark'])
            ax5.plot(rolling_corr.index, rolling_corr.values, linewidth=2, color='#2E86AB')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax5.set_title('Rolling Correlation with SPY (60d)', fontsize=12, fontweight='bold')
            ax5.set_ylim(-1, 1)
            ax5.grid(True, alpha=0.3)
        
        fig.suptitle('Momentum Factor Strategy - Performance Report',
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Report saved to {save_path}")
        
        return fig
    
    def save_all_charts(self, output_dir=RESULTS_DIR):
        """
        Save all individual charts
        """
        print("\nGenerating charts...")
        
        # Cumulative returns
        fig1 = self.plot_cumulative_returns()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig1.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  ✓ Cumulative returns chart saved")
        
        # Drawdown
        fig2 = self.plot_drawdown()
        fig2.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("  ✓ Drawdown chart saved")
        
        # Monthly heatmap
        fig3 = self.plot_monthly_returns_heatmap()
        fig3.savefig(f'{output_dir}/monthly_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  ✓ Monthly heatmap saved")
        
        # Rolling Sharpe
        fig4 = self.plot_rolling_sharpe()
        fig4.savefig(f'{output_dir}/rolling_sharpe.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  ✓ Rolling Sharpe chart saved")
        
        # Return distribution
        fig5 = self.plot_return_distribution()
        fig5.savefig(f'{output_dir}/return_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("  ✓ Return distribution saved")
        
        # Correlation
        if self.benchmark is not None:
            fig6 = self.plot_correlation_vs_benchmark()
            fig6.savefig(f'{output_dir}/correlation.png', dpi=300, bbox_inches='tight')
            plt.close(fig6)
            print("  ✓ Correlation chart saved")
        
        # Comprehensive report
        fig7 = self.create_comprehensive_report(
            save_path=f'{output_dir}/comprehensive_report.png'
        )
        plt.close(fig7)
        print("  ✓ Comprehensive report saved")
        
        print(f"\nAll charts saved to {output_dir}/")