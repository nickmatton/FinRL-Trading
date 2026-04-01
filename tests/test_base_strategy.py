"""Tests for BaseStrategy, StrategyConfig, and StrategyResult."""

import pandas as pd
import pytest

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult


# -- Concrete subclass for testing the non-abstract parts of BaseStrategy --

class _DummyStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs):
        return StrategyResult(strategy_name=self.config.name,
                              weights=pd.DataFrame(), metadata={})


# ---------------------------------------------------------------------------
# StrategyConfig
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    def test_create_with_name_only(self):
        cfg = StrategyConfig(name="my_strat")
        assert cfg.name == "my_strat"
        assert cfg.max_single_weight == 0.10
        assert cfg.max_positions == 20
        assert cfg.min_weight == 0.01

    def test_custom_values(self):
        cfg = StrategyConfig(name="custom", max_single_weight=0.25,
                             max_positions=10, min_weight=0.05)
        assert cfg.max_single_weight == 0.25
        assert cfg.max_positions == 10


# ---------------------------------------------------------------------------
# StrategyResult
# ---------------------------------------------------------------------------

class TestStrategyResult:
    def test_create(self):
        wdf = pd.DataFrame({"gvkey": ["A", "B"], "weight": [0.5, 0.5]})
        result = StrategyResult(strategy_name="s", weights=wdf,
                                metadata={"k": 1})
        assert result.strategy_name == "s"
        assert len(result.weights) == 2
        assert result.metadata["k"] == 1

    def test_default_metadata(self):
        result = StrategyResult(strategy_name="s", weights=pd.DataFrame())
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# BaseStrategy (via _DummyStrategy)
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseStrategy(StrategyConfig(name="x"))

    def test_config_and_logger(self, base_config):
        s = _DummyStrategy(base_config)
        assert s.config is base_config
        assert s.logger is not None

    # -- apply_risk_limits ------------------------------------------------

    def test_caps_weights(self):
        """Weights above max_single_weight should be capped then renormalized."""
        cfg = StrategyConfig(name="cap_test", max_single_weight=0.20,
                             max_positions=20, min_weight=0.01)
        s = _DummyStrategy(cfg)
        # 10 positions — enough room for the 0.20 cap to bind
        df = pd.DataFrame({
            "gvkey": list("ABCDEFGHIJ"),
            "weight": [0.40, 0.25, 0.10, 0.05, 0.05,
                       0.04, 0.03, 0.03, 0.03, 0.02],
        })
        out = s.apply_risk_limits(df)
        assert out["weight"].max() <= cfg.max_single_weight + 1e-9
        assert abs(out["weight"].sum() - 1.0) < 1e-9

    def test_drops_small_weights(self, base_config):
        """Weights below min_weight are removed."""
        s = _DummyStrategy(base_config)  # min_weight=0.02
        df = pd.DataFrame({"gvkey": list("ABCD"),
                           "weight": [0.50, 0.30, 0.19, 0.01]})
        out = s.apply_risk_limits(df)
        assert "D" not in out["gvkey"].values

    def test_max_positions(self, base_config):
        """Only top N positions are retained."""
        s = _DummyStrategy(base_config)  # max_positions=5
        df = pd.DataFrame({
            "gvkey": [f"T{i}" for i in range(10)],
            "weight": [0.10] * 10,
        })
        out = s.apply_risk_limits(df)
        assert len(out) <= base_config.max_positions
        assert abs(out["weight"].sum() - 1.0) < 1e-9

    def test_preserves_extra_columns(self, base_config):
        """Columns besides 'weight' survive the risk-limit pass."""
        s = _DummyStrategy(base_config)
        df = pd.DataFrame({
            "gvkey": ["A", "B"],
            "weight": [0.5, 0.5],
            "sector": ["Tech", "Health"],
            "date": ["2024-01-01", "2024-01-01"],
        })
        out = s.apply_risk_limits(df)
        assert "sector" in out.columns
        assert "date" in out.columns

    def test_empty_df(self, base_config):
        s = _DummyStrategy(base_config)
        df = pd.DataFrame(columns=["gvkey", "weight"])
        out = s.apply_risk_limits(df)
        assert out.empty
