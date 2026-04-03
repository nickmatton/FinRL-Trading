#!/usr/bin/env python3
"""
Runner script for the RL Portfolio strategy.

Usage:
    # Train + backtest
    python run_rl_strategy.py --config rl_strategy_config.yaml \
        --data-dir data/fmp_daily --backtest --start 2022-01-01 --end 2024-12-31

    # Train only (saves model to disk)
    python run_rl_strategy.py --config rl_strategy_config.yaml \
        --data-dir data/fmp_daily --train --end 2023-12-31

    # Single-date signal (loads saved model)
    python run_rl_strategy.py --config rl_strategy_config.yaml \
        --data-dir data/fmp_daily --date 2024-12-31
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date

import yaml
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.strategies.rl_strategy import RLStrategyConfig, RLPortfolioStrategy


def load_config(config_path: str) -> RLStrategyConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    tr = raw.get("training", {})
    env = raw.get("environment", {})
    roll = raw.get("rolling", {})
    pf = raw.get("portfolio", {})

    return RLStrategyConfig(
        name=raw["strategy"]["name"],
        algorithm=tr.get("algorithm", "PPO"),
        total_timesteps=tr.get("total_timesteps", 50_000),
        learning_rate=tr.get("learning_rate", 3e-4),
        batch_size=tr.get("batch_size", 64),
        n_epochs=tr.get("n_epochs", 10),
        gamma=tr.get("gamma", 0.99),
        window_size=env.get("window_size", 20),
        transaction_cost=env.get("transaction_cost", 0.001),
        reward_scaling=env.get("reward_scaling", 100.0),
        train_days=roll.get("train_days", 504),
        retrain_freq=roll.get("retrain_freq", "Q"),
        max_positions=pf.get("max_positions", 20),
        max_single_weight=pf.get("max_single_weight", 0.15),
        min_weight=pf.get("min_weight", 0.01),
        model_dir=raw.get("paths", {}).get("model_dir", "./models/rl_portfolio"),
    )


def load_price_data(data_dir: str, symbols: list) -> pd.DataFrame:
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

    wide = pd.DataFrame(frames).dropna()
    wide.index = pd.to_datetime(wide.index)
    print(f"  Loaded {len(wide.columns)} tickers, {len(wide)} trading days")
    return wide


def run_train(config_path: str, data_dir: str, end_date: str):
    """Train and save the model."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    symbols = raw.get("symbols", [])
    cfg = load_config(config_path)
    strategy = RLPortfolioStrategy(cfg)

    print(f"\n--- Loading price data from {data_dir} ---")
    prices = load_price_data(data_dir, symbols)
    prices = prices.loc[:pd.Timestamp(end_date)]

    print(f"\n--- Training ({len(prices)} days, {cfg.total_timesteps} timesteps) ---")
    meta = strategy.train(prices, verbose=1)
    print(f"  Training metadata: {meta}")

    strategy.save_model()
    print(f"  Model saved to {cfg.model_dir}")


def run_backtest(config_path: str, data_dir: str,
                 start_date: str, end_date: str):
    """Walk-forward train + backtest."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    symbols = raw.get("symbols", [])
    benchmarks = raw.get("benchmark", {}).get("tickers", ["SPY", "QQQ"])
    all_symbols = list(set(symbols + benchmarks))

    cfg = load_config(config_path)
    strategy = RLPortfolioStrategy(cfg)

    print(f"\n--- Loading price data from {data_dir} ---")
    prices_wide = load_price_data(data_dir, all_symbols)

    print(f"\n--- Walk-forward weight generation ({start_date} → {end_date}) ---")
    print(f"  Retrain freq: {cfg.retrain_freq}, Train window: {cfg.train_days} days")
    weight_matrix = strategy.generate_weight_matrix(
        prices_wide[symbols].dropna(axis=1, how="all"),
        start_date, end_date,
        retrain=True,
    )
    print(f"  {len(weight_matrix)} rebalance dates")

    if weight_matrix.empty:
        print("ERROR: no weights generated")
        sys.exit(1)

    print("\n--- Running backtest ---")
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

    bt_cfg = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq="Q",
        transaction_cost=0.001,
        benchmark_tickers=benchmarks,
    )
    engine = BacktestEngine(bt_cfg)
    result = engine.run_backtest("RL_Portfolio", prices_wide, weight_matrix)

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

    # Save model from the last training round
    strategy.save_model()


def run_single_date(config_path: str, data_dir: str, as_of_date: str):
    """Load saved model and generate weights for one date."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    symbols = raw.get("symbols", [])
    cfg = load_config(config_path)
    strategy = RLPortfolioStrategy(cfg)

    print(f"\n--- Loading price data from {data_dir} ---")
    prices_wide = load_price_data(data_dir, symbols)

    # Try to load a saved model; fall back to training
    model_path = Path(cfg.model_dir) / "ppo_portfolio.zip"
    if model_path.exists():
        print(f"  Loading model from {model_path}")
        strategy.load_model()
    else:
        print("  No saved model found — training from scratch")
        train_slice = prices_wide.loc[:pd.Timestamp(as_of_date)]
        strategy.train(train_slice, verbose=0)

    prices_long = prices_wide.loc[:pd.Timestamp(as_of_date)].stack().reset_index()
    prices_long.columns = ["date", "tic", "close"]

    result = strategy.generate_weights(
        {"prices": prices_long}, as_of_date=as_of_date, train=False
    )

    wdf = result.weights
    if wdf.empty:
        print("  No positions generated")
        return

    print(f"\n  Target Portfolio ({len(wdf)} positions):")
    print("  " + "-" * 30)
    for _, row in wdf.sort_values("weight", ascending=False).iterrows():
        print(f"    {row['tic']:8s}: {row['weight']:7.2%}")
    cash = 1.0 - wdf["weight"].sum()
    print(f"    {'CASH':8s}: {cash:7.2%}")

    weights_dir = Path(raw.get("paths", {}).get(
        "weights_dir", "./src/strategies/output/weights/rl_portfolio"
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
    parser = argparse.ArgumentParser(description="RL Portfolio Strategy Runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train", action="store_true", help="Train and save model")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--freq", default="Q", help="Rebalance frequency (ignored, uses config)")
    parser.add_argument("--no-daily-fast-track", action="store_true", help="Ignored (compatibility)")
    parser.add_argument("--date", default=None, help="Single decision date")
    args = parser.parse_args()

    if args.train:
        run_train(args.config, args.data_dir, args.end)
    elif args.backtest:
        run_backtest(args.config, args.data_dir, args.start, args.end)
    elif args.date:
        run_single_date(args.config, args.data_dir, args.date)
    else:
        parser.error("Specify --train, --backtest, or --date")


if __name__ == "__main__":
    main()
