"""
Strategies Layer Module
======================

Implements various quantitative trading strategies:
- Machine learning based stock selection
- Portfolio optimization (mean-variance, minimum variance, etc.)
- Dual-factor momentum + mean-reversion
- Backtesting engine
- Performance metrics calculation
"""

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from src.strategies.momentum_reversion_strategy import (
    MomentumReversionConfig,
    MomentumReversionStrategy,
)
