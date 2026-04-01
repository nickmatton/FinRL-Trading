# FinRL-X: Core Architecture & Building Your Own Trading Agent

This guide explains the core pieces of the FinRL-X platform and walks you through building your own trading agent on top of it.

---

## The Big Idea: Weight-Centric Architecture

Everything in FinRL-X revolves around a single concept: **the target portfolio weight vector `w_t`**. Every strategy, no matter how complex, ultimately produces a dictionary of `{symbol: weight}` that sums to 1.0. This weight vector is the universal interface between all layers:

```
Data Pipeline  -->  Strategy (produces weights)  -->  Backtest Engine
                                                 -->  Live Execution (Alpaca)
```

Because backtesting and live trading consume the same weight format, a strategy that works in backtest will execute identically in production with zero code changes.

---

## Repository Structure

```
src/
├── config/settings.py          # Pydantic-based configuration (loads from .env)
├── data/
│   ├── data_fetcher.py         # Multi-source data acquisition (FMP, Yahoo, WRDS)
│   ├── data_processor.py       # Feature engineering & cleaning
│   ├── data_store.py           # SQLite persistence layer
│   └── trading_calendar.py     # NYSE trading day utilities
├── strategies/
│   ├── ml_strategy.py          # ML stock selection (Random Forest / GBM)
│   ├── adaptive_rotation/      # Multi-asset rotation strategy (full sub-package)
│   ├── rl_model.py             # DRL portfolio allocation (PPO/SAC/A2C)
│   ├── base_signal.py          # Base signal computation engine
│   ├── execution_engine.py     # Strategy orchestration
│   └── universe_manager.py     # Dynamic stock universe tracking
├── backtest/
│   └── backtest_engine.py      # bt-powered backtesting with benchmarks
├── trading/
│   ├── alpaca_manager.py       # Alpaca API (multi-account, order management)
│   ├── trade_executor.py       # Risk-aware order execution
│   └── performance_analyzer.py # Live P&L tracking
└── main.py                     # CLI entry point
```

---

## Core Components

### 1. Configuration (`src/config/settings.py`)

All settings are managed through Pydantic models and loaded from a `.env` file:

```bash
cp .env.example .env
# Edit with your API keys:
#   APCA_API_KEY, APCA_API_SECRET  (Alpaca - required for trading)
#   FMP_API_KEY                     (Financial Modeling Prep - data)
#   OPENAI_API_KEY                  (optional - sentiment analysis)
```

Access in code:

```python
from src.config.settings import get_config
config = get_config()
print(config.alpaca.api_key)
print(config.get_data_dir())
```

### 2. Data Pipeline (`src/data/`)

The data layer handles acquisition, caching, and feature engineering.

**Fetching data** (`data_fetcher.py`):
- `fetch_price_data(tickers, start, end)` - OHLCV daily prices
- `fetch_fundamental_data(tickers, start, end)` - P/E, revenue, ROE, etc.
- `fetch_sp500_tickers()` - Current S&P 500 universe
- Auto-fallback priority: FMP > WRDS > Yahoo Finance

**Storage** (`data_store.py`):
- SQLite-backed with tables for `price_data`, `raw_payloads`, `news_articles`
- Deduplication and incremental updates
- Caches raw API responses for replay

**Feature engineering** (`data_processor.py`):
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Rolling volatility
- Lookahead-safe train/test splits

### 3. Strategies (`src/strategies/`)

Three built-in strategies demonstrate different approaches:

#### ML Stock Selection (`ml_strategy.py`)
- Quarterly selection of top-25% stocks using fundamental scoring
- Random Forest or Gradient Boosting ranking model
- Weight methods: equal-weight or minimum-variance optimization

```python
from src.strategies.ml_strategy import MLStockSelectionStrategy

strategy = MLStockSelectionStrategy(config)
result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='min_variance'
)
# result.weights -> DataFrame of {ticker: weight} per rebalance date
```

#### Adaptive Multi-Asset Rotation (`adaptive_rotation/`)
The most sophisticated strategy, built as a modular sub-package:

| Module | Role |
|:-------|:-----|
| `config_loader.py` | YAML-driven strategy parameters |
| `market_regime.py` | Two-layer regime detection (slow weekly + fast daily risk-off) |
| `group_strength.py` | Ranks asset groups by Information Ratio vs QQQ |
| `intra_group_ranking.py` | Selects top assets within each group via residual momentum |
| `exception_framework.py` | Overrides for exceptional momentum assets |
| `portfolio_builder.py` | Assembles final weights from all signals |
| `risk_manager.py` | Stop-loss, trailing stops, cooldown enforcement |
| `walk_forward.py` | Walk-forward backtesting with daily monitoring |

```python
from src.strategies.adaptive_rotation.adaptive_rotation_engine import AdaptiveRotationEngine

engine = AdaptiveRotationEngine(config)
weights, audit_log = engine.run(price_data, as_of_date="2024-12-31")
# weights.weights -> {symbol: weight}
# audit_log -> complete decision trace
```

#### DRL Allocator (`rl_model.py`)
- Stable Baselines3 agents (PPO, SAC, A2C)
- Continuous action space for portfolio weights
- Rolling window training (2+ years train, 1 quarter test)

### 4. Backtesting (`src/backtest/backtest_engine.py`)

Built on the professional `bt` library with multi-benchmark comparison:

```python
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2024-12-31',
    rebalance_freq='Q',           # Q/M/W
    transaction_cost=0.001,       # 10 bps
    benchmark_tickers=['SPY', 'QQQ']
)

engine = BacktestEngine(config)
result = engine.run_backtest("My Strategy", weights_df, prices_df)

# result.metrics includes:
#   annualized_return, sharpe_ratio, max_drawdown, calmar_ratio,
#   volatility, win_rate, information_ratio
```

### 5. Live Trading (`src/trading/`)

**Alpaca integration** (`alpaca_manager.py`):
```python
from src.trading.alpaca_manager import AlpacaManager, AlpacaAccount

account = AlpacaAccount(
    name="Paper1",
    api_key="your_key",
    api_secret="your_secret",
    base_url="https://paper-api.alpaca.markets"
)
alpaca = AlpacaManager([account])
alpaca.set_account("Paper1")

# Check state
alpaca.get_account_info()
alpaca.get_positions()

# Execute
alpaca.execute_portfolio_rebalance(target_weights={'AAPL': 0.3, 'MSFT': 0.7})
```

**Risk-aware execution** (`trade_executor.py`):
- Max order value checks ($100k default)
- Max portfolio turnover (50% default)
- Min order size filtering
- Timeout handling and execution logging

---

## Building Your Own Trading Agent

### Step 1: Define Your Strategy

The simplest path is to produce a weights DataFrame that maps dates to `{ticker: weight}`:

```python
import pandas as pd
import numpy as np

def my_momentum_strategy(prices_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Simple momentum strategy: buy the top N stocks by 6-month return.

    Args:
        prices_df: DataFrame with columns [date, ticker, close]
        top_n: number of stocks to hold

    Returns:
        DataFrame with columns [date, ticker, weight]
    """
    # Pivot to wide format
    wide = prices_df.pivot(index='date', columns='ticker', values='close')

    # Compute 126-day (6-month) returns
    momentum = wide.pct_change(126)

    # Monthly rebalance dates
    rebalance_dates = momentum.resample('M').last().index

    weights_list = []
    for date in rebalance_dates:
        if date not in momentum.index:
            continue
        scores = momentum.loc[date].dropna()
        top = scores.nlargest(top_n).index.tolist()
        weight = 1.0 / len(top)
        for ticker in top:
            weights_list.append({
                'date': date,
                'ticker': ticker,
                'weight': weight
            })

    return pd.DataFrame(weights_list)
```

### Step 2: Fetch Data

```python
from src.data.data_fetcher import fetch_price_data

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
           'JPM', 'V', 'UNH', 'XOM', 'JNJ', 'PG', 'MA', 'HD']

prices = fetch_price_data(tickers, '2019-01-01', '2024-12-31')
```

Or load from CSV files (the format `deploy.sh` downloads):

```python
from pathlib import Path
import pandas as pd

data_dir = Path("data/fmp_daily")
frames = []
for csv in data_dir.glob("*_daily.csv"):
    df = pd.read_csv(csv, parse_dates=['date'])
    df['ticker'] = csv.stem.replace('_daily', '')
    frames.append(df)
prices = pd.concat(frames, ignore_index=True)
```

### Step 3: Backtest

```python
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

weights = my_momentum_strategy(prices, top_n=10)

bt_config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2024-12-31',
    rebalance_freq='M',
    transaction_cost=0.001,
    benchmark_tickers=['SPY', 'QQQ']
)

engine = BacktestEngine(bt_config)
result = engine.run_backtest("Momentum Top-10", weights, prices)

print(f"Annualized Return: {result.metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio:      {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown:      {result.metrics['max_drawdown']:.2%}")
```

### Step 4: Paper Trade

Once your backtest looks good, deploy to Alpaca paper trading:

```python
from src.trading.alpaca_manager import AlpacaManager, AlpacaAccount

account = AlpacaAccount(
    name="Paper",
    api_key="your_paper_key",
    api_secret="your_paper_secret",
    base_url="https://paper-api.alpaca.markets"
)
alpaca = AlpacaManager([account])
alpaca.set_account("Paper")

# Get latest weights from your strategy
latest_weights = weights[weights['date'] == weights['date'].max()]
target = dict(zip(latest_weights['ticker'], latest_weights['weight']))

# Execute rebalance
alpaca.execute_portfolio_rebalance(target_weights=target)
```

Or use the CLI deploy script:

```bash
# Backtest
./deploy.sh --strategy adaptive_rotation --mode backtest --start 2020-01-01 --end 2024-12-31

# Paper trade (preview first)
./deploy.sh --strategy adaptive_rotation --mode paper --dry-run

# Paper trade (execute)
./deploy.sh --strategy adaptive_rotation --mode paper
```

### Step 5: Monitor Performance

```python
from src.trading.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(alpaca)
metrics = analyzer.get_performance_metrics()
# Returns: daily returns, cumulative return, Sharpe, max drawdown, benchmark comparison
```

---

## Adding a Strategy to the Deploy Pipeline

To make your strategy runnable via `deploy.sh`:

1. **Create a runner script** (e.g., `src/strategies/run_my_strategy.py`) that accepts `--config`, `--backtest`, `--start`, `--end`, and `--date` CLI args.

2. **Register it in `deploy.sh`** by adding a line to the `STRATEGIES` variable:

```bash
STRATEGIES="
adaptive_rotation|src/strategies/AdaptiveRotationConf_v1.2.1.yaml|src/strategies/run_adaptive_rotation_strategy.py
my_strategy|src/strategies/my_strategy_config.yaml|src/strategies/run_my_strategy.py
"
```

3. Run it:
```bash
./deploy.sh --strategy my_strategy --mode backtest --start 2020-01-01 --end 2024-12-31
```

---

## Key Design Principles

1. **Weights are the contract.** Every strategy outputs `{ticker: weight}`. Backtesting and execution consume that same format. This makes components freely swappable.

2. **No lookahead.** Data slicing is always `as_of(date)` - only data available on the decision date is used. The adaptive rotation strategy enforces this through `get_data_as_of()` helpers.

3. **Modularity.** The adaptive rotation strategy shows how to decompose a complex strategy into independent modules (regime detection, group ranking, asset selection, risk management) that communicate through well-defined interfaces.

4. **Backtest-to-production consistency.** The same weight vector flows through `bt` backtesting and Alpaca order execution, eliminating simulation-to-live divergence.

---

## Key Dependencies

| Purpose | Library |
|:--------|:--------|
| ML models | `scikit-learn`, `xgboost`, `lightgbm` |
| DRL agents | `stable-baselines3`, `torch` |
| Backtesting | `bt` |
| Data sources | `yfinance`, `requests` (FMP) |
| Broker | `alpaca-py` |
| Config | `pydantic`, `pydantic-settings`, `python-dotenv` |
| Trading calendar | `pandas-market-calendars` |
| Dashboard | `streamlit`, `plotly` |

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Quick Reference

| Task | Command / Code |
|:-----|:---------------|
| Start dashboard | `python src/main.py dashboard` |
| Run backtest | `python src/main.py backtest` |
| Execute trade | `python src/main.py trade` |
| Check config | `python src/main.py config` |
| Deploy strategy | `./deploy.sh --strategy adaptive_rotation --mode backtest` |
| Jupyter tutorial | `jupyter notebook examples/FinRL_Full_Workflow.ipynb` |
