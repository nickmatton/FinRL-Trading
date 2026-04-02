"""
Portfolio Allocation Gymnasium Environment
==========================================

A Gymnasium-compatible environment for training RL agents to allocate
portfolio weights across multiple assets.

Observation:  Rolling window of per-asset features (returns, volatility,
              RSI-like momentum) plus the current portfolio weights.
Action:       Continuous weight vector in [0, 1]^n_assets, softmax-
              normalised so weights sum to 1.
Reward:       Daily portfolio log-return minus a transaction-cost penalty.

Usage:
    env = PortfolioEnv(prices_wide, window_size=20)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Portfolio allocation environment.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format daily prices (DatetimeIndex × tickers).
    window_size : int
        Number of past trading days used to build the observation.
    transaction_cost : float
        Proportional cost applied to absolute weight changes.
    reward_scaling : float
        Multiplier on the raw reward (helps SB3 optimisers).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        window_size: int = 20,
        transaction_cost: float = 0.001,
        reward_scaling: float = 100.0,
    ):
        super().__init__()

        self.prices = prices.values.astype(np.float64)
        self.tickers = list(prices.columns)
        self.dates = prices.index
        self.n_assets = len(self.tickers)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        # Pre-compute daily returns (first row is NaN → 0)
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = self.prices[1:] / self.prices[:-1] - 1.0

        # Action: raw logits for each asset (will be softmax-normalised).
        # Bounded to [-1, 1] as required by SB3; softmax handles the rest.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # Observation: per-asset features over the window + current weights
        # Features per asset per day: return, 5d-vol, 20d-momentum
        self.n_features = 3
        obs_size = self.n_assets * self.n_features * self.window_size + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32,
        )

        # Episode state
        self._start_idx = self.window_size
        self._end_idx = len(self.prices) - 1
        self._current_step = self._start_idx
        self._weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = self._start_idx
        self._weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        # Normalise action to portfolio weights via softmax
        new_weights = self._softmax(action.astype(np.float64))

        # Transaction cost based on turnover
        turnover = np.sum(np.abs(new_weights - self._weights))
        tc = self.transaction_cost * turnover

        # Portfolio return for this step
        day_returns = self.returns[self._current_step]
        portfolio_return = np.dot(new_weights, day_returns)
        log_return = np.log1p(portfolio_return + 1e-10)

        reward = float((log_return - tc) * self.reward_scaling)

        # Update state
        self._weights = new_weights
        self._current_step += 1

        terminated = self._current_step >= self._end_idx
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector."""
        t = self._current_step
        window = slice(t - self.window_size, t)

        # Raw returns over the window: (window_size, n_assets)
        rets = self.returns[window]

        # Rolling 5-day volatility (std of returns)
        vol = np.zeros_like(rets)
        for i in range(rets.shape[0]):
            start = max(0, i - 4)
            vol[i] = rets[start:i + 1].std(axis=0) if i >= 1 else 0.0

        # 20-day cumulative momentum
        cum_ret = np.cumprod(1.0 + rets, axis=0) - 1.0

        # Stack features: (window_size, n_assets, 3)
        features = np.stack([rets, vol, cum_ret], axis=-1)

        # Flatten and append current weights
        flat = features.flatten().astype(np.float32)
        obs = np.concatenate([flat, self._weights.astype(np.float32)])
        return obs

    def _get_info(self) -> dict:
        idx = min(self._current_step, len(self.dates) - 1)
        return {
            "date": str(self.dates[idx]),
            "step": self._current_step,
            "weights": self._weights.copy(),
            "portfolio_value": 1.0,  # placeholder — tracked externally
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()
