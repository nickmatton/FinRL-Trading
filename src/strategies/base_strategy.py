"""
Base Strategy Module
====================

Provides the foundational classes for all trading strategies:
- StrategyConfig: Strategy configuration parameters
- StrategyResult: Strategy output (weights + metadata)
- BaseStrategy: Abstract base class with shared risk-limit logic
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    max_single_weight: float = 0.10
    max_positions: int = 20
    min_weight: float = 0.01


@dataclass
class StrategyResult:
    """Output of a strategy's generate_weights method."""
    strategy_name: str
    weights: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    @abstractmethod
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        """Generate portfolio weights from input data."""
        ...

    def apply_risk_limits(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk limits to a weights DataFrame.

        Caps individual weights, drops tiny positions, trims to max
        positions, and renormalizes so weights sum to 1.0.

        The DataFrame must contain a 'weight' column. All other columns
        are preserved unchanged.

        Args:
            weights_df: DataFrame with at least a 'weight' column.

        Returns:
            Cleaned DataFrame with weights summing to 1.0.
        """
        if weights_df.empty or 'weight' not in weights_df.columns:
            return weights_df

        df = weights_df.copy()

        # Drop positions below minimum weight
        df = df[df['weight'] >= self.config.min_weight].copy()

        if df.empty:
            return df

        # Keep only top N positions by weight
        if len(df) > self.config.max_positions:
            df = df.nlargest(self.config.max_positions, 'weight').copy()

        # Iteratively cap weights and redistribute excess to uncapped
        # positions until convergence. When every position would exceed
        # the cap (e.g. 3 positions with cap=0.20), the cap is the
        # binding constraint and we equal-weight among them.
        cap = self.config.max_single_weight
        weights = df['weight'].values.copy().astype(float)
        frozen = np.zeros(len(weights), dtype=bool)

        for _ in range(50):
            total = weights.sum()
            if total <= 0:
                break
            normed = weights / total
            over = (~frozen) & (normed > cap)
            if not over.any():
                weights = normed
                break
            # Freeze capped positions at the cap value
            weights[over] = cap
            frozen |= over
            # Redistribute the remaining budget among unfrozen positions
            remaining = 1.0 - weights[frozen].sum()
            unfrozen = ~frozen
            uf_total = weights[unfrozen].sum()
            if uf_total > 0 and remaining > 0:
                weights[unfrozen] = weights[unfrozen] / uf_total * remaining
            elif unfrozen.any():
                n_uf = unfrozen.sum()
                weights[unfrozen] = remaining / n_uf
        else:
            # All positions frozen at cap — renormalize
            total = weights.sum()
            if total > 0:
                weights = weights / total

        df['weight'] = weights
        return df.reset_index(drop=True)
