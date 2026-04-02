"""
RL Portfolio Allocation Strategy
================================

Uses Stable Baselines3 PPO to learn continuous portfolio weight
allocations.  The agent is trained on historical price data via the
PortfolioEnv Gymnasium environment, then used to generate weight
vectors for backtesting or live trading.

Training modes:
- **Single-period**: Train on one block of history, test on a held-out
  period.
- **Rolling-window**: Walk-forward training where the model is retrained
  at each rebalance date on the most recent ``train_days`` of data.

The strategy inherits from ``BaseStrategy`` so it plugs directly into
the backtest engine and Alpaca execution pipeline.
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from src.strategies.rl_env import PortfolioEnv

logger = logging.getLogger(__name__)


@dataclass
class RLStrategyConfig(StrategyConfig):
    """Configuration for the RL portfolio strategy."""
    # Training
    algorithm: str = "PPO"
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    # Environment
    window_size: int = 20
    transaction_cost: float = 0.001
    reward_scaling: float = 100.0
    # Rolling window
    train_days: int = 504   # ~2 years of trading days
    retrain_freq: str = "Q"  # retrain quarterly
    # Model persistence
    model_dir: str = "./models/rl_portfolio"


class RLPortfolioStrategy(BaseStrategy):
    """RL-based portfolio allocation using Stable Baselines3."""

    def __init__(self, config: RLStrategyConfig):
        super().__init__(config)
        self.model: Optional[PPO] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, prices_wide: pd.DataFrame,
              verbose: int = 0) -> Dict[str, Any]:
        """
        Train the RL agent on the given price data.

        Args:
            prices_wide: Wide-format daily prices (DatetimeIndex × tickers).
            verbose: SB3 verbosity level (0 = silent).

        Returns:
            Dictionary with training metadata (timesteps, env info).
        """
        cfg = self.config

        env = PortfolioEnv(
            prices=prices_wide,
            window_size=cfg.window_size,
            transaction_cost=cfg.transaction_cost,
            reward_scaling=cfg.reward_scaling,
        )
        vec_env = DummyVecEnv([lambda: env])

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            verbose=verbose,
        )

        self.model.learn(total_timesteps=cfg.total_timesteps)

        self.logger.info(
            f"Training complete: {cfg.total_timesteps} timesteps, "
            f"{len(prices_wide.columns)} assets"
        )

        return {
            "total_timesteps": cfg.total_timesteps,
            "n_assets": len(prices_wide.columns),
            "train_days": len(prices_wide),
        }

    def save_model(self, path: Optional[str] = None):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save — call train() first")
        save_path = Path(path or self.config.model_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path / "ppo_portfolio"))
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """Load a previously trained model from disk."""
        load_path = Path(path or self.config.model_dir) / "ppo_portfolio.zip"
        self.model = PPO.load(str(load_path))
        self.logger.info(f"Model loaded from {load_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_weights(self, prices_wide: pd.DataFrame) -> np.ndarray:
        """
        Run the trained agent on price data and return the final weights.

        The agent steps through the entire price history and returns the
        weight vector chosen at the last step.

        Args:
            prices_wide: Wide-format daily prices (must have at least
                ``window_size + 1`` rows).

        Returns:
            1-D numpy array of portfolio weights summing to 1.
        """
        if self.model is None:
            raise ValueError("No model loaded — call train() or load_model() first")

        env = PortfolioEnv(
            prices=prices_wide,
            window_size=self.config.window_size,
            transaction_cost=self.config.transaction_cost,
            reward_scaling=self.config.reward_scaling,
        )

        obs, _ = env.reset()
        weights = None
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            weights = info["weights"]
            done = terminated or truncated

        return weights

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                         **kwargs) -> StrategyResult:
        """
        Generate portfolio weights using the trained RL agent.

        Expects ``data['prices']`` in either:
        - Wide format (DatetimeIndex × tickers) — used directly.
        - Long format with ``[date, tic, close]`` — pivoted automatically.

        If no model is loaded, trains one on the provided data first.

        Kwargs:
            as_of_date: Optional end date to slice data.
            train: bool — if True, (re)train before predicting. Default True
                   if no model is loaded.
        """
        prices = data.get("prices")
        if prices is None or prices.empty:
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=["tic", "weight"]),
                metadata={"error": "no price data"},
            )

        # Auto-detect format
        if "tic" in prices.columns or "gvkey" in prices.columns:
            ticker_col = "tic" if "tic" in prices.columns else "gvkey"
            date_col = "date" if "date" in prices.columns else "datadate"
            price_col = next(
                (c for c in ["close", "adj_close", "prccd"] if c in prices.columns),
                "close",
            )
            prices_wide = prices.pivot_table(
                index=date_col, columns=ticker_col, values=price_col, aggfunc="last"
            )
            prices_wide.index = pd.to_datetime(prices_wide.index)
            prices_wide = prices_wide.sort_index()
        else:
            prices_wide = prices.copy()

        as_of = kwargs.get("as_of_date")
        if as_of is not None:
            prices_wide = prices_wide.loc[:pd.Timestamp(as_of)]

        should_train = kwargs.get("train", self.model is None)
        if should_train:
            self.train(prices_wide, verbose=0)

        weights_arr = self.predict_weights(prices_wide)
        tickers = list(prices_wide.columns)

        wdf = pd.DataFrame({"tic": tickers, "weight": weights_arr})
        wdf = self.apply_risk_limits(wdf)

        return StrategyResult(
            strategy_name=self.config.name,
            weights=wdf,
            metadata={
                "n_positions": len(wdf),
                "as_of_date": str(prices_wide.index[-1].date()),
            },
        )

    def generate_weight_matrix(self, prices_wide: pd.DataFrame,
                               start_date: str, end_date: str,
                               retrain: bool = True) -> pd.DataFrame:
        """
        Walk-forward weight generation for backtesting.

        At each rebalance date the agent is (optionally) retrained on the
        most recent ``train_days`` of data, then predicts weights for that
        date.

        Args:
            prices_wide: Full history of wide-format prices.
            start_date: First eligible rebalance date.
            end_date: Last eligible rebalance date.
            retrain: Whether to retrain at each rebalance. If False, trains
                     once on data up to ``start_date`` and reuses the model.

        Returns:
            Wide-format weight matrix (index = dates, columns = tickers).
        """
        cfg = self.config
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Determine rebalance dates
        freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
        pd_freq = freq_map.get(cfg.retrain_freq, cfg.retrain_freq)
        candidates = pd.date_range(start, end, freq=pd_freq)

        # Map each candidate to nearest prior trading date in the index
        rebalance_dates = []
        for dt in candidates:
            mask = prices_wide.index[prices_wide.index <= dt]
            if len(mask) > 0:
                rebalance_dates.append(mask[-1])
        rebalance_dates = sorted(set(rebalance_dates))

        if not retrain:
            # Train once on data up to start
            train_slice = prices_wide.loc[:start].iloc[-cfg.train_days:]
            if len(train_slice) > cfg.window_size + 10:
                self.train(train_slice, verbose=0)

        rows = {}
        for dt in rebalance_dates:
            train_end = dt
            train_start_idx = max(
                0,
                prices_wide.index.get_indexer([train_end], method="ffill")[0]
                - cfg.train_days
            )
            train_slice = prices_wide.iloc[train_start_idx:
                                           prices_wide.index.get_indexer(
                                               [train_end], method="ffill"
                                           )[0] + 1]

            if len(train_slice) <= cfg.window_size + 10:
                continue

            if retrain:
                self.train(train_slice, verbose=0)

            weights_arr = self.predict_weights(train_slice)
            row = pd.Series(0.0, index=prices_wide.columns)
            row[:] = weights_arr

            # Apply risk limits
            wdf = pd.DataFrame({"tic": prices_wide.columns, "weight": weights_arr})
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
