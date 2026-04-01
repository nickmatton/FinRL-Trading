"""
Dual-Factor Momentum + Mean-Reversion Strategy
===============================================

Combines two well-documented academic factors:

1. Cross-sectional momentum (12-1 month):
   Rank stocks by cumulative return from month t-12 to t-1 (skipping
   the most recent month to avoid short-term reversal). Select the
   top quintile. (Jegadeesh & Titman, 1993)

2. Short-term mean reversion (5-day):
   Within the momentum-selected universe, tilt weight toward stocks
   with the largest recent drawdowns. (Lo & MacKinlay, 1990)

The two scores are blended via configurable weights and converted to
portfolio allocations.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

logger = logging.getLogger(__name__)


@dataclass
class MomentumReversionConfig(StrategyConfig):
    """Configuration for the Momentum + Reversion strategy."""
    momentum_lookback_months: int = 12
    skip_months: int = 1
    reversion_lookback_days: int = 5
    momentum_weight: float = 0.6
    reversion_weight: float = 0.4
    top_pct: float = 0.20
    rebalance_freq: str = "M"


class MomentumReversionStrategy(BaseStrategy):
    """Dual-factor momentum + mean-reversion strategy."""

    def __init__(self, config: MomentumReversionConfig):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                         **kwargs) -> StrategyResult:
        """
        Generate portfolio weights for a single rebalance date.

        Args:
            data: Dictionary with key ``'prices'`` containing a DataFrame.
                  Accepted column layouts:
                  - ``[date, tic, close]``
                  - ``[datadate, tic, adj_close]``
                  - ``[date, gvkey, close]``
            **kwargs:
                as_of_date: Optional[str] — decision date.  Defaults to
                    the last date in the price data.

        Returns:
            StrategyResult with weights DataFrame (columns: tic, weight).
        """
        prices_long = data.get("prices")
        if prices_long is None or prices_long.empty:
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=["tic", "weight"]),
                metadata={"error": "no price data provided"},
            )

        date_col, ticker_col, price_col = self._infer_columns(prices_long)
        prices_wide = self._to_wide(prices_long, date_col, ticker_col, price_col)

        as_of = kwargs.get("as_of_date")
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            prices_wide = prices_wide.loc[:as_of]
        else:
            as_of = prices_wide.index[-1]

        weights_df = self._select_and_weight(prices_wide, as_of)

        if not weights_df.empty:
            weights_df = self.apply_risk_limits(weights_df)

        return StrategyResult(
            strategy_name=self.config.name,
            weights=weights_df,
            metadata={
                "as_of_date": str(as_of.date()),
                "n_positions": len(weights_df),
            },
        )

    def generate_weight_matrix(self, prices_wide: pd.DataFrame,
                               start_date: str, end_date: str) -> pd.DataFrame:
        """
        Produce a wide-format weight matrix for backtesting.

        Args:
            prices_wide: Wide DataFrame (DatetimeIndex, columns = tickers).
            start_date: First eligible rebalance date.
            end_date: Last eligible rebalance date.

        Returns:
            DataFrame — index = rebalance dates, columns = tickers,
            values = portfolio weights.  Compatible with
            ``BacktestEngine.run_backtest()``.
        """
        cfg = self.config
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        rebalance_dates = self._get_rebalance_dates(
            prices_wide.index, start, end, cfg.rebalance_freq
        )

        rows = {}
        for dt in rebalance_dates:
            data_slice = prices_wide.loc[:dt]
            wdf = self._select_and_weight(data_slice, dt)
            if wdf.empty:
                continue
            wdf = self.apply_risk_limits(wdf)
            row = pd.Series(0.0, index=prices_wide.columns)
            row.update(wdf.set_index("tic")["weight"])
            rows[dt] = row

        if not rows:
            return pd.DataFrame(
                index=pd.DatetimeIndex([], name="date"),
                columns=prices_wide.columns,
            )

        matrix = pd.DataFrame(rows).T
        matrix.index.name = "date"
        return matrix

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _compute_momentum_scores(self, prices_wide: pd.DataFrame,
                                 as_of_date: pd.Timestamp) -> pd.Series:
        """
        12-1 month cross-sectional momentum.

        Returns a Series indexed by ticker with momentum return values.
        Tickers with insufficient history are dropped.
        """
        cfg = self.config
        total_months = cfg.momentum_lookback_months
        skip = cfg.skip_months
        lookback_days = total_months * 21
        skip_days = skip * 21

        end_idx = prices_wide.index.get_indexer([as_of_date], method="ffill")[0]
        if end_idx < lookback_days:
            return pd.Series(dtype=float)

        start_idx = end_idx - lookback_days
        skip_idx = end_idx - skip_days

        p_start = prices_wide.iloc[start_idx]
        p_end = prices_wide.iloc[skip_idx]

        momentum = (p_end / p_start) - 1.0
        return momentum.dropna()

    def _compute_reversion_scores(self, prices_wide: pd.DataFrame,
                                  as_of_date: pd.Timestamp) -> pd.Series:
        """
        Short-term (5-day) mean-reversion signal.

        Returns a Series indexed by ticker.  Lower recent return → higher
        reversion score (contrarian tilt).
        """
        days = self.config.reversion_lookback_days

        end_idx = prices_wide.index.get_indexer([as_of_date], method="ffill")[0]
        if end_idx < days:
            return pd.Series(dtype=float)

        start_idx = end_idx - days
        recent_ret = (prices_wide.iloc[end_idx] / prices_wide.iloc[start_idx]) - 1.0
        # Invert: stocks that fell the most get the highest reversion score
        reversion = -recent_ret
        return reversion.dropna()

    def _select_and_weight(self, prices_wide: pd.DataFrame,
                           as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Core logic: score, select, blend, and produce weights.

        Returns DataFrame with columns [tic, weight].
        """
        cfg = self.config

        momentum = self._compute_momentum_scores(prices_wide, as_of_date)
        if momentum.empty:
            return pd.DataFrame(columns=["tic", "weight"])

        # Select top quintile by momentum
        n_select = max(1, int(np.ceil(len(momentum) * cfg.top_pct)))
        top_tickers = momentum.nlargest(n_select).index

        reversion = self._compute_reversion_scores(prices_wide, as_of_date)
        # Restrict reversion to the momentum-selected universe
        reversion = reversion.reindex(top_tickers).fillna(0.0)

        # Rank-based blending (rank → 0-1 scale)
        mom_rank = momentum.reindex(top_tickers).rank(pct=True).fillna(0.5)
        rev_rank = reversion.rank(pct=True).fillna(0.5)

        combined = cfg.momentum_weight * mom_rank + cfg.reversion_weight * rev_rank

        # Convert combined scores to weights
        total = combined.sum()
        if total == 0:
            weights = pd.Series(1.0 / len(combined), index=combined.index)
        else:
            weights = combined / total

        df = pd.DataFrame({"tic": weights.index, "weight": weights.values})
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_columns(df: pd.DataFrame):
        """Detect date, ticker, and price column names."""
        cols = set(df.columns)

        date_col = "date" if "date" in cols else "datadate"
        ticker_col = "tic" if "tic" in cols else "gvkey"
        price_col = (
            "close" if "close" in cols
            else "adj_close" if "adj_close" in cols
            else "prccd"
        )
        return date_col, ticker_col, price_col

    @staticmethod
    def _to_wide(df: pd.DataFrame, date_col: str,
                 ticker_col: str, price_col: str) -> pd.DataFrame:
        """Pivot long-format prices to wide (DatetimeIndex x tickers)."""
        wide = df.pivot_table(
            index=date_col, columns=ticker_col, values=price_col, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        return wide

    @staticmethod
    def _get_rebalance_dates(index: pd.DatetimeIndex,
                             start: pd.Timestamp, end: pd.Timestamp,
                             freq: str) -> pd.DatetimeIndex:
        """Return month-end (or other freq) dates within [start, end] that exist in the index."""
        # pandas >= 2.2 renamed 'M' → 'ME', 'Q' → 'QE', etc.
        freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
        pd_freq = freq_map.get(freq, freq)
        candidates = pd.date_range(start, end, freq=pd_freq)
        # Map each candidate to the nearest prior date in the index
        valid = []
        for dt in candidates:
            mask = index[index <= dt]
            if len(mask) > 0:
                valid.append(mask[-1])
        return pd.DatetimeIndex(sorted(set(valid)))
