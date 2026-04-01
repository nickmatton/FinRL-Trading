"""Shared pytest fixtures for FinRL-Trading tests."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import StrategyConfig
from src.strategies.momentum_reversion_strategy import MomentumReversionConfig


# ---------------------------------------------------------------------------
# Synthetic price data
# ---------------------------------------------------------------------------

def _generate_gbm_prices(
    n_tickers: int = 30,
    n_days: int = 756,  # ~3 years of trading days
    seed: int = 42,
    start_date: str = "2021-01-04",
) -> pd.DataFrame:
    """
    Generate synthetic daily prices using geometric Brownian motion.

    Returns a wide DataFrame: DatetimeIndex × tickers, values = prices.
    Each ticker has a random annualised drift in [-0.05, 0.25] and vol
    in [0.15, 0.45], so the universe contains both winners and losers.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start_date, periods=n_days, freq="B")
    tickers = [f"TIC{i:02d}" for i in range(n_tickers)]

    # Random parameters per ticker
    annual_drift = rng.uniform(-0.05, 0.25, n_tickers)
    annual_vol = rng.uniform(0.15, 0.45, n_tickers)

    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # GBM paths
    log_returns = (
        daily_drift[None, :]
        + daily_vol[None, :] * rng.standard_normal((n_days, n_tickers))
    )
    log_prices = np.cumsum(log_returns, axis=0)
    # Start all prices at 100
    prices = 100.0 * np.exp(log_prices)

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def synthetic_prices_wide() -> pd.DataFrame:
    """Wide-format synthetic prices: DatetimeIndex × tickers."""
    return _generate_gbm_prices()


@pytest.fixture
def synthetic_prices_long(synthetic_prices_wide) -> pd.DataFrame:
    """Long-format synthetic prices: [date, tic, close]."""
    df = synthetic_prices_wide.stack().reset_index()
    df.columns = ["date", "tic", "close"]
    return df


@pytest.fixture
def sample_config() -> MomentumReversionConfig:
    return MomentumReversionConfig(name="test_momentum_reversion")


@pytest.fixture
def base_config() -> StrategyConfig:
    return StrategyConfig(name="test_base", max_single_weight=0.20,
                          max_positions=5, min_weight=0.02)
