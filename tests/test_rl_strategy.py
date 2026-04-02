"""Tests for RL Portfolio Strategy — env, training, and backtest integration."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.rl_env import PortfolioEnv
from src.strategies.rl_strategy import RLStrategyConfig, RLPortfolioStrategy
from src.strategies.base_strategy import StrategyResult


# ======================================================================
# PortfolioEnv tests
# ======================================================================

class TestPortfolioEnv:
    def test_reset_returns_obs(self, synthetic_prices_wide):
        env = PortfolioEnv(synthetic_prices_wide, window_size=10)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert info["step"] == 10  # window_size

    def test_step_returns_correct_tuple(self, synthetic_prices_wide):
        env = PortfolioEnv(synthetic_prices_wide, window_size=10)
        obs, _ = env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "weights" in info

    def test_weights_sum_to_one(self, synthetic_prices_wide):
        env = PortfolioEnv(synthetic_prices_wide, window_size=10)
        env.reset()
        action = np.random.randn(env.n_assets)
        _, _, _, _, info = env.step(action)
        assert abs(info["weights"].sum() - 1.0) < 1e-9

    def test_episode_terminates(self, synthetic_prices_wide):
        env = PortfolioEnv(synthetic_prices_wide, window_size=10)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        expected_steps = len(synthetic_prices_wide) - 10 - 1
        assert steps == expected_steps

    def test_softmax_normalisation(self):
        """Extreme logits should still produce valid weights."""
        w = PortfolioEnv._softmax(np.array([1000.0, -1000.0, 0.0]))
        assert abs(w.sum() - 1.0) < 1e-9
        assert w[0] > 0.99  # first should dominate

    def test_small_env(self):
        """Env works with minimal data."""
        dates = pd.bdate_range("2024-01-01", periods=30, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 110, 30),
                                "B": np.linspace(100, 90, 30)}, index=dates)
        env = PortfolioEnv(prices, window_size=5)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape


# ======================================================================
# RLPortfolioStrategy tests
# ======================================================================

@pytest.fixture
def rl_config():
    return RLStrategyConfig(
        name="test_rl",
        total_timesteps=500,    # very low for fast tests
        window_size=10,
        train_days=200,
        max_single_weight=0.15,
        max_positions=20,
    )


class TestRLTraining:
    def test_train_creates_model(self, synthetic_prices_wide, rl_config):
        strategy = RLPortfolioStrategy(rl_config)
        assert strategy.model is None
        strategy.train(synthetic_prices_wide, verbose=0)
        assert strategy.model is not None

    def test_predict_weights_shape(self, synthetic_prices_wide, rl_config):
        strategy = RLPortfolioStrategy(rl_config)
        strategy.train(synthetic_prices_wide, verbose=0)
        weights = strategy.predict_weights(synthetic_prices_wide)
        assert len(weights) == len(synthetic_prices_wide.columns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_save_and_load(self, synthetic_prices_wide, rl_config, tmp_path):
        strategy = RLPortfolioStrategy(rl_config)
        strategy.train(synthetic_prices_wide, verbose=0)

        model_dir = str(tmp_path / "model")
        strategy.save_model(model_dir)

        strategy2 = RLPortfolioStrategy(rl_config)
        strategy2.load_model(model_dir)
        w = strategy2.predict_weights(synthetic_prices_wide)
        assert abs(w.sum() - 1.0) < 1e-6


class TestRLGenerateWeights:
    def test_returns_strategy_result(self, synthetic_prices_long, rl_config):
        strategy = RLPortfolioStrategy(rl_config)
        result = strategy.generate_weights({"prices": synthetic_prices_long})
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == rl_config.name
        assert not result.weights.empty

    def test_weights_sum_to_one(self, synthetic_prices_long, rl_config):
        strategy = RLPortfolioStrategy(rl_config)
        result = strategy.generate_weights({"prices": synthetic_prices_long})
        if not result.weights.empty:
            assert abs(result.weights["weight"].sum() - 1.0) < 0.05

    def test_empty_data(self, rl_config):
        strategy = RLPortfolioStrategy(rl_config)
        result = strategy.generate_weights({"prices": pd.DataFrame()})
        assert result.weights.empty

    def test_wide_format_input(self, synthetic_prices_wide, rl_config):
        """generate_weights should also accept wide-format prices."""
        strategy = RLPortfolioStrategy(rl_config)
        result = strategy.generate_weights({"prices": synthetic_prices_wide})
        assert isinstance(result, StrategyResult)
        assert not result.weights.empty


class TestRLWeightMatrix:
    def test_generates_matrix(self, synthetic_prices_wide, rl_config):
        rl_config.retrain_freq = "Q"
        rl_config.train_days = 200
        strategy = RLPortfolioStrategy(rl_config)

        start = str(synthetic_prices_wide.index[300].date())
        end = str(synthetic_prices_wide.index[-1].date())
        matrix = strategy.generate_weight_matrix(
            synthetic_prices_wide, start, end, retrain=True
        )

        assert isinstance(matrix.index, pd.DatetimeIndex)
        assert len(matrix) > 0

    def test_rows_sum_approximately_one(self, synthetic_prices_wide, rl_config):
        rl_config.retrain_freq = "Q"
        rl_config.train_days = 200
        strategy = RLPortfolioStrategy(rl_config)

        start = str(synthetic_prices_wide.index[300].date())
        end = str(synthetic_prices_wide.index[-1].date())
        matrix = strategy.generate_weight_matrix(
            synthetic_prices_wide, start, end, retrain=True
        )

        for s in matrix.sum(axis=1):
            assert abs(s - 1.0) < 0.10  # tolerance for risk-limit rounding


# ======================================================================
# Backtest integration
# ======================================================================

class TestRLBacktestIntegration:
    def test_end_to_end(self, synthetic_prices_wide, rl_config):
        """Train → generate weight matrix → run backtest → valid metrics."""
        from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

        rl_config.retrain_freq = "Q"
        rl_config.train_days = 200
        strategy = RLPortfolioStrategy(rl_config)

        start = str(synthetic_prices_wide.index[300].date())
        end = str(synthetic_prices_wide.index[-1].date())

        weight_matrix = strategy.generate_weight_matrix(
            synthetic_prices_wide, start, end, retrain=True
        )

        bt_cfg = BacktestConfig(
            start_date=start,
            end_date=end,
            rebalance_freq="Q",
            transaction_cost=0.001,
            benchmark_tickers=[],
        )
        engine = BacktestEngine(bt_cfg)
        result = engine.run_backtest(
            "RL_Portfolio", synthetic_prices_wide, weight_matrix
        )

        assert result.strategy_name == "RL_Portfolio"
        assert np.isfinite(result.metrics.get("sharpe_ratio", float("nan")))
        assert np.isfinite(result.metrics.get("max_drawdown", float("nan")))
        assert np.isfinite(result.annualized_return)
