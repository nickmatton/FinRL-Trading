"""Tests for MomentumReversionStrategy — unit tests and backtest integration."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import StrategyResult
from src.strategies.momentum_reversion_strategy import (
    MomentumReversionConfig,
    MomentumReversionStrategy,
)


# ======================================================================
# Unit tests
# ======================================================================

class TestMomentumScores:
    def test_returns_series(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        as_of = synthetic_prices_wide.index[-1]
        scores = strategy._compute_momentum_scores(synthetic_prices_wide, as_of)
        assert isinstance(scores, pd.Series)
        assert len(scores) > 0

    def test_ranking_order(self):
        """A ticker with higher 12-1 month return should score higher."""
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        prices = pd.DataFrame({
            "WINNER": np.linspace(100, 200, 300),   # steady uptrend
            "LOSER":  np.linspace(100, 80, 300),    # downtrend
            "FLAT":   np.full(300, 100.0),           # flat
        }, index=dates)

        cfg = MomentumReversionConfig(name="test")
        strategy = MomentumReversionStrategy(cfg)
        scores = strategy._compute_momentum_scores(prices, dates[-1])

        assert scores["WINNER"] > scores["FLAT"]
        assert scores["FLAT"] > scores["LOSER"]

    def test_insufficient_history(self):
        """With less data than the lookback, should return empty."""
        dates = pd.bdate_range("2024-01-01", periods=30, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 110, 30)}, index=dates)

        cfg = MomentumReversionConfig(name="test")
        strategy = MomentumReversionStrategy(cfg)
        scores = strategy._compute_momentum_scores(prices, dates[-1])
        assert scores.empty


class TestReversionScores:
    def test_returns_series(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        as_of = synthetic_prices_wide.index[-1]
        scores = strategy._compute_reversion_scores(synthetic_prices_wide, as_of)
        assert isinstance(scores, pd.Series)
        assert len(scores) > 0

    def test_contrarian_tilt(self):
        """A ticker that dropped recently should have a higher reversion score."""
        dates = pd.bdate_range("2024-01-01", periods=20, freq="B")
        # DROPPER crashes on the last 5 days, RISER rallies
        dropper = np.concatenate([np.full(15, 100.0), np.linspace(100, 80, 5)])
        riser = np.concatenate([np.full(15, 100.0), np.linspace(100, 120, 5)])
        prices = pd.DataFrame({"DROPPER": dropper, "RISER": riser}, index=dates)

        cfg = MomentumReversionConfig(name="test")
        strategy = MomentumReversionStrategy(cfg)
        scores = strategy._compute_reversion_scores(prices, dates[-1])
        assert scores["DROPPER"] > scores["RISER"]


class TestSelectAndWeight:
    def test_weights_sum_to_one(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        as_of = synthetic_prices_wide.index[-1]
        wdf = strategy._select_and_weight(synthetic_prices_wide, as_of)
        assert abs(wdf["weight"].sum() - 1.0) < 1e-9

    def test_number_of_positions(self, synthetic_prices_wide, sample_config):
        """Should select roughly top_pct of the universe."""
        strategy = MomentumReversionStrategy(sample_config)
        as_of = synthetic_prices_wide.index[-1]
        wdf = strategy._select_and_weight(synthetic_prices_wide, as_of)
        expected = max(1, int(np.ceil(30 * sample_config.top_pct)))  # 30 tickers
        assert len(wdf) == expected


class TestGenerateWeights:
    def test_returns_strategy_result(self, synthetic_prices_long, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        result = strategy.generate_weights({"prices": synthetic_prices_long})
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == sample_config.name

    def test_weights_respect_max(self, synthetic_prices_long):
        """With enough positions, weights should respect the cap."""
        cfg = MomentumReversionConfig(
            name="test_cap",
            top_pct=0.50,         # select 50% → ~15 positions from 30 tickers
            max_single_weight=0.10,
        )
        strategy = MomentumReversionStrategy(cfg)
        result = strategy.generate_weights({"prices": synthetic_prices_long})
        if not result.weights.empty:
            assert result.weights["weight"].max() <= cfg.max_single_weight + 1e-9

    def test_empty_data(self, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        result = strategy.generate_weights({"prices": pd.DataFrame()})
        assert result.weights.empty

    def test_single_ticker(self, sample_config):
        dates = pd.bdate_range("2020-01-01", periods=300, freq="B")
        df = pd.DataFrame({
            "date": dates, "tic": "ONLY", "close": np.linspace(100, 150, 300)
        })
        strategy = MomentumReversionStrategy(sample_config)
        result = strategy.generate_weights({"prices": df})
        if not result.weights.empty:
            assert abs(result.weights["weight"].sum() - 1.0) < 1e-9


class TestGenerateWeightMatrix:
    def test_wide_format(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        start = str(synthetic_prices_wide.index[252].date())  # ~1 year in
        end = str(synthetic_prices_wide.index[-1].date())
        matrix = strategy.generate_weight_matrix(synthetic_prices_wide, start, end)

        assert isinstance(matrix.index, pd.DatetimeIndex)
        assert set(matrix.columns).issubset(set(synthetic_prices_wide.columns))
        assert len(matrix) > 0

    def test_monthly_rebalance_dates(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        start = str(synthetic_prices_wide.index[252].date())
        end = str(synthetic_prices_wide.index[-1].date())
        matrix = strategy.generate_weight_matrix(synthetic_prices_wide, start, end)

        # Each row should be a different month
        months = matrix.index.to_period("M")
        assert months.is_unique

    def test_rows_sum_to_one(self, synthetic_prices_wide, sample_config):
        strategy = MomentumReversionStrategy(sample_config)
        start = str(synthetic_prices_wide.index[252].date())
        end = str(synthetic_prices_wide.index[-1].date())
        matrix = strategy.generate_weight_matrix(synthetic_prices_wide, start, end)

        row_sums = matrix.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 0.05  # allow small tolerance from risk limits


# ======================================================================
# Backtest integration test
# ======================================================================

class TestBacktestIntegration:
    def test_end_to_end(self, synthetic_prices_wide, sample_config):
        """Generate weights → run BacktestEngine → get valid metrics."""
        from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

        strategy = MomentumReversionStrategy(sample_config)
        start = str(synthetic_prices_wide.index[252].date())
        end = str(synthetic_prices_wide.index[-1].date())

        weight_matrix = strategy.generate_weight_matrix(
            synthetic_prices_wide, start, end
        )

        bt_cfg = BacktestConfig(
            start_date=start,
            end_date=end,
            rebalance_freq="M",
            transaction_cost=0.001,
            benchmark_tickers=[],  # no benchmarks — avoids API calls
        )
        engine = BacktestEngine(bt_cfg)
        result = engine.run_backtest(
            "MomentumReversion", synthetic_prices_wide, weight_matrix
        )

        assert result.strategy_name == "MomentumReversion"
        assert np.isfinite(result.metrics.get("sharpe_ratio", float("nan")))
        assert np.isfinite(result.metrics.get("max_drawdown", float("nan")))
        assert np.isfinite(result.annualized_return)
