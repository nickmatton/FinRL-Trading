#!/usr/bin/env python3
"""
Runner script for the Momentum + Reversion strategy.

Usage:
    # Backtest
    python run_momentum_reversion.py --config config.yaml --data-dir data/fmp_daily \
        --backtest --start 2021-01-01 --end 2024-12-31

    # Single-date signal
    python run_momentum_reversion.py --config config.yaml --data-dir data/fmp_daily \
        --date 2024-12-31
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

import yaml
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.strategies.momentum_reversion_strategy import (
    MomentumReversionConfig,
    MomentumReversionStrategy,
)


def load_config(config_path: str) -> MomentumReversionConfig:
    """Load strategy config from YAML."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return MomentumReversionConfig(
        name=raw["strategy"]["name"],
        momentum_lookback_months=raw.get("momentum", {}).get("lookback_months", 12),
        skip_months=raw.get("momentum", {}).get("skip_months", 1),
        reversion_lookback_days=raw.get("reversion", {}).get("lookback_days", 5),
        momentum_weight=raw.get("blending", {}).get("momentum_weight", 0.6),
        reversion_weight=raw.get("blending", {}).get("reversion_weight", 0.4),
        top_pct=raw.get("portfolio", {}).get("top_pct", 0.20),
        max_positions=raw.get("portfolio", {}).get("max_positions", 20),
        max_single_weight=raw.get("portfolio", {}).get("max_single_weight", 0.10),
        min_weight=raw.get("portfolio", {}).get("min_weight", 0.01),
        rebalance_freq=raw.get("portfolio", {}).get("rebalance_freq", "M"),
    )


def load_price_data(data_dir: str, symbols: list) -> pd.DataFrame:
    """
    Load per-ticker CSV files into a single wide DataFrame.

    Each file is expected at ``{data_dir}/{SYMBOL}_daily.csv`` with at
    least ``date`` and ``close`` columns.
    """
    data_path = Path(data_dir)
    frames = {}
    for sym in symbols:
        csv = data_path / f"{sym}_daily.csv"
        if not csv.exists():
            print(f"  WARNING: {csv} not found, skipping {sym}")
            continue
        df = pd.read_csv(csv, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        frames[sym] = df["close"]

    if not frames:
        print("ERROR: no price data loaded")
        sys.exit(1)

    wide = pd.DataFrame(frames)
    wide.index = pd.to_datetime(wide.index)
    print(f"  Loaded {len(wide.columns)} tickers, {len(wide)} trading days")
    return wide


def run_backtest(config_path: str, data_dir: str,
                 start_date: str, end_date: str):
    """Run a historical backtest and print metrics."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    symbols = raw.get("symbols", [])
    benchmarks = raw.get("benchmark", {}).get("tickers", ["SPY", "QQQ"])
    symbols_all = list(set(symbols + benchmarks))

    cfg = load_config(config_path)
    strategy = MomentumReversionStrategy(cfg)

    print(f"\n--- Loading price data from {data_dir} ---")
    prices_wide = load_price_data(data_dir, symbols_all)

    print(f"\n--- Generating weight matrix ({start_date} → {end_date}) ---")
    weight_matrix = strategy.generate_weight_matrix(
        prices_wide[symbols].dropna(axis=1, how="all"),
        start_date, end_date,
    )
    print(f"  {len(weight_matrix)} rebalance dates")

    if weight_matrix.empty:
        print("ERROR: no weights generated (insufficient history?)")
        sys.exit(1)

    print("\n--- Running backtest ---")
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

    bt_cfg = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq="M",
        transaction_cost=0.001,
        benchmark_tickers=benchmarks,
    )
    engine = BacktestEngine(bt_cfg)
    result = engine.run_backtest("MomentumReversion", prices_wide, weight_matrix)

    print("\n========== BACKTEST RESULTS ==========")
    print(f"  Period:            {start_date} → {end_date}")
    print(f"  Annualised Return: {result.annualized_return:>8.2%}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:>10.4f}")
    if result.benchmark_annualized:
        print("\n  Benchmarks:")
        for bm, ret in result.benchmark_annualized.items():
            print(f"    {bm:8s}: {ret:>8.2%}")
    print("======================================\n")


def run_single_date(config_path: str, data_dir: str, as_of_date: str):
    """Generate weights for a single decision date and print them."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    symbols = raw.get("symbols", [])
    cfg = load_config(config_path)
    strategy = MomentumReversionStrategy(cfg)

    print(f"\n--- Loading price data from {data_dir} ---")
    prices_wide = load_price_data(data_dir, symbols)

    print(f"\n--- Generating weights for {as_of_date} ---")
    prices_long = prices_wide.stack().reset_index()
    prices_long.columns = ["date", "tic", "close"]

    result = strategy.generate_weights(
        {"prices": prices_long}, as_of_date=as_of_date
    )

    wdf = result.weights
    if wdf.empty:
        print("  No positions generated (insufficient data?)")
        return

    print(f"\n  Target Portfolio ({len(wdf)} positions):")
    print("  " + "-" * 30)
    for _, row in wdf.sort_values("weight", ascending=False).iterrows():
        print(f"    {row['tic']:8s}: {row['weight']:7.2%}")
    cash = 1.0 - wdf["weight"].sum()
    print(f"    {'CASH':8s}: {cash:7.2%}")

    # Save signal
    weights_dir = Path(raw.get("paths", {}).get(
        "weights_dir", "./src/strategies/output/weights/momentum_reversion"
    ))
    weights_dir.mkdir(parents=True, exist_ok=True)
    signal_file = weights_dir / f"signal_{as_of_date}.json"
    signal = {
        "date": as_of_date,
        "weights": dict(zip(wdf["tic"], wdf["weight"].round(6))),
    }
    with open(signal_file, "w") as f:
        json.dump(signal, f, indent=2, default=str)
    print(f"\n  Signal saved to: {signal_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Momentum + Reversion Strategy Runner"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data-dir", required=True, help="Directory with *_daily.csv")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--freq", default="M")
    parser.add_argument("--date", default=None, help="Single decision date")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.config, args.data_dir, args.start, args.end)
    elif args.date:
        run_single_date(args.config, args.data_dir, args.date)
    else:
        parser.error("Specify --backtest or --date")


if __name__ == "__main__":
    main()
