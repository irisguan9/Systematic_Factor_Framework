import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from config import *


class MLFactorWeighter:
    def __init__(
        self,
        min_train_periods: int = 252,
        refit_frequency: int = 21,
        forward_return_horizon: int = 21,
        ic_window: int = 3,
        alphas = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0), # penalty parameters
        weight_bounds: tuple = (0.05, 0.70),
    ):
        self.min_train_periods    = min_train_periods
        self.refit_frequency      = refit_frequency
        self.forward_horizon      = forward_return_horizon
        self.ic_window            = ic_window
        self.alphas               = alphas
        self.weight_bounds        = weight_bounds

        self.weight_history:   pd.DataFrame | None = None
        self.alpha_history:    pd.Series    | None = None
        self.ic_history:       pd.DataFrame | None = None
        self.composite_factor: pd.DataFrame | None = None

    # =========================================================================
    # STEP 1 — Build monthly IC series for each factor
    # =========================================================================
    def _build_ic_series(
        self,
        factor_dict: dict[str, pd.DataFrame],
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
    
        print("\nBuilding monthly IC series...")

        forward_returns = prices.shift(-self.forward_horizon) / prices - 1
        monthly_dates   = prices.resample('BM').last().index

        records = []
        for date in monthly_dates:
            if date not in prices.index:
                continue

            fwd  = forward_returns.loc[date].dropna()
            row  = {'date': date}
            valid = True

            for name, factor_df in factor_dict.items():
                if date not in factor_df.index:
                    valid = False
                    break
                scores = factor_df.loc[date].dropna()
                common = scores.index.intersection(fwd.index)

                if len(common) < 50:
                    valid = False
                    break

                ic = scores[common].corr(fwd[common], method='spearman')
                row[name] = ic

            if valid:
                records.append(row)

        ic_df = pd.DataFrame(records).set_index('date').dropna()
        self.ic_history = ic_df

        print(f"  IC series built: {len(ic_df)} months")
        print(f"\n  IC summary (mean | std | positive%):")
        for col in ic_df.columns:
            print(f"    {col:<12}: mean={ic_df[col].mean():+.4f} | "
                  f"std={ic_df[col].std():.4f} | "
                  f"positive%={(ic_df[col] > 0).mean():.1%}")

        return ic_df

    # =========================================================================
    # STEP 2 — Build X and y from IC series for one training window
    # =========================================================================
    def _build_ic_features(
        self,
        ic_df: pd.DataFrame,
        train_end_idx: int,
    ) -> tuple:
        
        train_ic = ic_df.iloc[:train_end_idx]

        if len(train_ic) < self.ic_window + 2:
            return None, None

        lag1      = train_ic.shift(1)
        roll_mean = train_ic.rolling(self.ic_window).mean().shift(1)

        X_df = pd.concat([lag1, roll_mean], axis=1)
        X_df.columns = (
            [f"{c}_lag1"              for c in train_ic.columns] +
            [f"{c}_roll{self.ic_window}" for c in train_ic.columns]
        )

        y = train_ic.mean(axis=1)

        combined = pd.concat([X_df, y.rename('y')], axis=1).dropna()

        if len(combined) < 12:
            return None, None

        return combined.drop(columns='y').values, combined['y'].values

    # =========================================================================
    # STEP 3 — Walk-forward training loop
    # =========================================================================
    def fit_walk_forward(
        self,
        factor_dict: dict[str, pd.DataFrame],
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
  
        print("\n" + "="*60)
        print("WALK-FORWARD RIDGE REGRESSION (IC-based)")
        print("="*60)

        ic_df        = self._build_ic_series(factor_dict, prices)
        factor_names = list(factor_dict.keys())

        weight_records = []
        alpha_records  = []
        refit_dates    = []

        min_months        = max(self.min_train_periods // 21, self.ic_window + 12)
        refit_every_n     = max(1, self.refit_frequency // 21)

        for i in range(min_months, len(ic_df)):

            if i % refit_every_n != 0:
                continue

            X_train, y_train = self._build_ic_features(ic_df, train_end_idx=i)

            if X_train is None:
                continue

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            tscv  = TimeSeriesSplit(n_splits=3)
            model = RidgeCV(
                alphas        = self.alphas,
                cv            = tscv,
                scoring       = 'neg_mean_squared_error',
                fit_intercept = True,
            )
            model.fit(X_scaled, y_train)

            # only use lag1 coefficients for weights
            # (first n_factors columns = lag1 features)
            n          = len(factor_names)
            lag1_coefs = model.coef_[:n]

            print(f"  {ic_df.index[i].date()} | "
                  f"alpha={model.alpha_:.2f} | "
                  f"lag1_coefs: " +
                  " ".join(f"{factor_names[j]}={lag1_coefs[j]:+.4f}"
                           for j in range(n)))

            weights = self._coef_to_weights(lag1_coefs, factor_names)
            date    = ic_df.index[i]

            weight_records.append({'date': date, **weights})
            alpha_records.append({'date': date, 'alpha': model.alpha_})
            refit_dates.append(date)

        if not weight_records:
            print("\nWARNING: No weights computed.")
            print(f"  IC series length : {len(ic_df)} months")
            print(f"  Required minimum : {min_months} months")
            raise ValueError("Insufficient data for IC-based Ridge.")

        self.weight_history = pd.DataFrame(weight_records).set_index('date')
        self.alpha_history  = (
            pd.DataFrame(alpha_records).set_index('date')['alpha']
        )

        print(f"\n  Total refits       : {len(refit_dates)}")
        print(f"  Weight history shape: {self.weight_history.shape}")

        return self.weight_history

    # =========================================================================
    # STEP 4 — Softmax coef → interpretable weights
    # =========================================================================
    def _coef_to_weights(
        self,
        raw_coefs: np.ndarray,
        factor_names: list[str],
    ) -> dict[str, float]:
        
        coefs = np.array(raw_coefs, dtype=float)

        coefs_shifted = coefs - coefs.max()
        exp_coefs     = np.exp(coefs_shifted)
        weights       = exp_coefs / exp_coefs.sum()

        lo, hi  = self.weight_bounds
        weights = np.clip(weights, lo, hi)
        weights = weights / weights.sum()

        return dict(zip(factor_names, weights))

    # =========================================================================
    # STEP 5 — Apply dynamic weights → composite factor
    # =========================================================================
    def apply_weights(
        self,
        factor_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        
        if self.weight_history is None:
            raise ValueError("Run fit_walk_forward() first.")

        prices_index = list(factor_dict.values())[0].index
        factor_names = list(factor_dict.keys())

        weights_daily = (
            self.weight_history
            .reindex(prices_index)
            .ffill()
            .shift(1)
        )

        composite = pd.DataFrame(
            0.0,
            index   = prices_index,
            columns = list(factor_dict.values())[0].columns,
        )

        for name in factor_names:
            composite += factor_dict[name].mul(weights_daily[name], axis=0).fillna(0)

        self.composite_factor = composite
        return composite

    # =========================================================================
    # STEP 6 — Diagnostic plots
    # =========================================================================
    def plot_weight_history(self):
        try:
            import matplotlib.pyplot as plt

            n_charts = 3 if self.ic_history is not None else 2
            fig, axes = plt.subplots(n_charts, 1, figsize=(12, 4 * n_charts))

            # Chart 1: weights
            ax1 = axes[0]
            self.weight_history.plot(ax=ax1, linewidth=1.5)
            ax1.axhline(1 / len(self.weight_history.columns),
                        linestyle='--', color='grey', alpha=0.5,
                        label='Equal weight')
            ax1.set_title('Ridge Factor Weights Over Time', fontweight='bold')
            ax1.set_ylabel('Weight')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Chart 2: IC history
            if self.ic_history is not None:
                ax2 = axes[1]
                self.ic_history.plot(ax=ax2, linewidth=1.0, alpha=0.7)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.set_title('Monthly IC per Factor (input to Ridge)',
                              fontweight='bold')
                ax2.set_ylabel('Spearman IC')
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)

            # Chart 3: alpha
            ax3 = axes[-1]
            self.alpha_history.plot(ax=ax3, color='steelblue', linewidth=1.5)
            ax3.set_title('Selected Ridge Alpha (Regularisation Strength)',
                          fontweight='bold')
            ax3.set_ylabel('Alpha')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except ImportError:
            print("matplotlib not available")
            return None

    # =========================================================================
    # STEP 7 — Summary table
    # =========================================================================
    def summary(self) -> pd.DataFrame:
        if self.weight_history is None:
            print("Run fit_walk_forward() first.")
            return

        equal_w = 1 / len(self.weight_history.columns)

        weight_stats = pd.DataFrame({
            'weight_mean': self.weight_history.mean(),
            'weight_std':  self.weight_history.std(),
            'weight_min':  self.weight_history.min(),
            'weight_max':  self.weight_history.max(),
        }).round(3)

        if self.ic_history is not None:
            ic_stats = pd.DataFrame({
                'ic_mean':       self.ic_history.mean(),
                'ic_std':        self.ic_history.std(),
                'ic_positive%':  (self.ic_history > 0).mean(),
            }).round(4)
            stats = pd.concat([weight_stats, ic_stats], axis=1)
        else:
            stats = weight_stats

        print("\n" + "="*60)
        print("RIDGE WEIGHT + IC SUMMARY")
        print(f"Equal-weight baseline : {equal_w:.3f}")
        print("="*60)
        print(stats.to_string())
        print()

        return stats