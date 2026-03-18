

# FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading
<p align="center">
  <img src="https://github.com/user-attachments/assets/80fe89bb-fb09-4267-b29a-76030512f8cf" width="500">
</p>

[![Downloads](https://static.pepy.tech/badge/finrl-trading)](https://pepy.tech/project/finrl-trading)
[![Downloads](https://static.pepy.tech/badge/finrl-trading/week)](https://pepy.tech/project/finrl-trading)
[![Join Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/trsr8SXpW5)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/finrl-trading.svg)](https://pypi.org/project/finrl-trading/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/FinRL-Trading.svg?color=brightgreen)
![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/FinRL-Trading?label=Issues)
![](https://img.shields.io/github/issues-closed-raw/AI4Finance-Foundation/FinRL-Trading?label=Closed+Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/FinRL-Trading?label=Open+PRs)
![](https://img.shields.io/github/issues-pr-closed-raw/AI4Finance-Foundation/FinRL-Trading?label=Closed+PRs)


![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL-Trading&countColor=%23B17A)
[![](https://dcbadge.limes.pink/api/server/trsr8SXpW5?cb=1)](https://discord.gg/trsr8SXpW5)


**A modern, modular quantitative trading platform built with Python, featuring machine learning strategies, professional backtesting, and live trading capabilities.**

## 🚀 Key Features

- **🤖 Strategy Framework**: Multiple quantitative strategies including ML-based stock selection
- **📈 Risk Management**: Comprehensive risk controls and position limits
- **💰 Live Trading**: Alpaca integration with paper and live trading support
- **🔧 Modular Design**: Clean, extensible architecture following best practices

## 🏗️ Project Architecture

```
finrl-trading/
├── src/
│   ├── config/           # Centralized configuration management
│   │   └── settings.py   # Pydantic-based settings with environment variables
│   ├── data/            # Data acquisition and processing
│   │   ├── data_fetcher.py     # Multi-source data integration (Yahoo/FMP/WRDS)
│   │   ├── data_processor.py   # Data cleaning and feature engineering
│   │   └── data_store.py       # SQLite-based data persistence
│   ├── backtest/      # Backtesting system
│   │   └── backtest_engine.py  # Professional backtesting engine powered by bt library
│   ├── strategies/      # Trading strategies
│   │   ├── base_strategy.py    # Abstract strategy framework
│   │   └── ml_strategy.py      # ML-based stock selection
│   ├── trading/         # Live trading execution
│   │   ├── alpaca_manager.py     # Alpaca API integration
│   │   ├── trade_executor.py     # Order execution and risk management
│   │   └── performance_analyzer.py  # Performance analysis
│   └── main.py         # CLI entry point
├── examples/           # Examples and tutorials
│   ├── FinRL_Full_Workflow.ipynb  # Complete workflow tutorial (recommended)
│   └── README.md       # Examples documentation
├── data/               # Runtime data storage (gitignored)
├── logs/               # Application logs (gitignored)
├── requirements.txt    # Python dependencies
└── setup.py           # Package installation
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Alpaca Account** (for live trading)
- **Data Source APIs**:
  - FMP API Key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/FinRL-Trading.git
   cd FinRL-Trading
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Copy configuration template
   cp .env.example .env
   
   # Edit .env file with your API keys
   # Windows: notepad .env
   # Linux/Mac: nano .env
   ```

4. **Run example tutorial**
   ```bash
   # Launch Jupyter Notebook (recommended starting point)
   jupyter notebook examples/FinRL_Full_Workflow.ipynb
   ```

### Complete Example Tutorial

The project includes a comprehensive interactive tutorial covering the entire workflow from data acquisition to live trading:

```bash
# View examples documentation
cat examples/README.md

# Run complete workflow tutorial (recommended)
jupyter notebook examples/FinRL_Full_Workflow.ipynb
```

**Tutorial Contents:**
- ✅ S&P 500 components data acquisition
- ✅ Fundamental and historical price data fetching
- ✅ Machine learning stock selection strategy implementation
- ✅ Professional backtesting (with VOO/QQQ benchmark comparison)
- ✅ Alpaca Paper Trading execution


## 📖 Usage Examples

### Data Acquisition

```python
from src.data.data_fetcher import get_data_manager

# Initialize data manager (automatically selects best available source)
manager = get_data_manager()

# Check current data source
info = manager.get_source_info()
print(f"Current data source: {info['current_source']}")
print(f"Available sources: {info['available_sources']}")

# Get S&P 500 components
components = manager.get_sp500_components()

# Fetch fundamental data
tickers = ['AAPL', 'MSFT', 'GOOGL']
fundamentals = manager.get_fundamental_data(
    tickers, '2020-01-01', '2023-12-31'
)

# Fetch historical price data
prices = manager.get_price_data(
    tickers, '2020-01-01', '2023-12-31'
)
```

### Strategy Development

```python
from src.strategies.ml_strategy import MLStockSelectorStrategy
from src.strategies.base_strategy import StrategyConfig

# Create ML-based stock selection strategy
config = StrategyConfig(
    name="ML Stock Selector",
    parameters={
        'model_type': 'random_forest',
        'top_n': 30,
        'sector_neutral': True
    },
    risk_limits={'max_weight': 0.1}
)

strategy = MLStockSelectorStrategy(config)

# Generate portfolio weights
data = {
    'fundamentals': fundamentals,
    'prices': prices
}
result = strategy.generate_weights(data)
print(result.weights.head())
```

### Strategy Backtesting

```python
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

# Configure backtest parameters
backtest_config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1000000,
    rebalance_freq='Q',  # Quarterly rebalancing
    transaction_cost=0.001,  # 0.1% transaction cost
    benchmark_tickers=['VOO', 'QQQ']  # Benchmark comparison
)

# Run backtest
engine = BacktestEngine(backtest_config)
result = engine.run_backtest(
    strategy_name="ML Stock Selector",
    weight_signals=ml_weights,
    price_data=prices
)

# View backtest results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Annualized Return: {result.annualized_return:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# Generate visualization report
engine.plot_results(result)
```

### Live Trading

```python
from src.trading.alpaca_manager import create_alpaca_account_from_env, AlpacaManager
from src.trading.trade_executor import TradeExecutor, ExecutionConfig

# Connect to Alpaca
account = create_alpaca_account_from_env()
alpaca_manager = AlpacaManager([account])

# Configure execution settings
exec_config = ExecutionConfig(
    max_order_value=100000,
    risk_checks_enabled=True
)

executor = TradeExecutor(alpaca_manager, exec_config)

# Execute portfolio rebalance
target_weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}
result = executor.execute_portfolio_rebalance(target_weights)

print(f"Orders placed: {len(result.orders_placed)}")
print(f"Execution success: {result.success}")
```

## 🎯 Core Components

### Data Layer (`src/data/`)
- **Multi-Source Data Manager** (`data_fetcher.py`): Intelligent data source selection and management
  - Yahoo Finance: Free financial data (default)
  - FMP (Financial Modeling Prep): High-quality paid data (requires API Key)
  - WRDS: Academic database (requires credentials)
- **Data Processor** (`data_processor.py`): Feature engineering, data cleaning, and quality checks
- **Data Storage** (`data_store.py`): SQLite-based data persistence with caching and version control

### Strategy Framework (`src/strategies/`)
- **Base Strategy** (`base_strategy.py`): Abstract framework for custom strategies
- **ML Strategy** (`ml_strategy.py`): Random Forest-based stock selection

**Implemented Strategies:**
- Equal Weight Strategy
- Market Cap Weighted Strategy
- ML-based Stock Selection
- Sector Neutral ML Strategy

### Backtesting System (`src/backtest/`)
- **Professional Backtesting Engine** (`backtest_engine.py`): Powered by `bt` library
  - Comprehensive performance and risk analysis
  - Multiple benchmark comparison (SPY, VOO, QQQ, etc.)
  - Transaction cost simulation
  - Visualization report generation

### Trading System (`src/trading/`)
- **Alpaca Integration** (`alpaca_manager.py`): Alpaca API client with multi-account support
- **Trade Executor** (`trade_executor.py`): Order management and risk controls
- **Performance Analyzer** (`performance_analyzer.py`): Real-time position tracking and P&L calculation

### Configuration System (`src/config/`)
- **Pydantic Settings** (`settings.py`): Type-safe configuration with environment variables
- **Multi-environment Support**: Development, testing, production configurations
- **Centralized Management**: All settings in one place


## 🔧 Configuration

The platform uses **Pydantic-based settings** with environment variable support:

### Environment Variables

Create a `.env` file and configure the following variables:

```bash
# Application
ENVIRONMENT=development
APP_NAME="FinRL Trading"

# Alpaca API (Required for live trading)
APCA_API_KEY=your_alpaca_key
APCA_API_SECRET=your_alpaca_secret
APCA_BASE_URL=https://paper-api.alpaca.markets  # Paper Trading

# Data Sources (Optional, prioritized: FMP > WRDS > Yahoo)
FMP_API_KEY=your_fmp_api_key           # Financial Modeling Prep


# Risk Management
TRADING_MAX_ORDER_VALUE=100000         # Maximum order value
TRADING_MAX_PORTFOLIO_TURNOVER=0.5     # Maximum portfolio turnover
STRATEGY_MAX_WEIGHT_PER_STOCK=0.1      # Maximum weight per stock

# Data Management
DATA_CACHE_TTL_HOURS=24                # Cache TTL in hours
DATA_MAX_CACHE_SIZE_MB=1000            # Maximum cache size in MB
```

### Configuration Usage

```python
from src.config.settings import get_config

config = get_config()
print(f"Environment: {config.environment}")
print(f"Database: {config.database.url}")
print(f"Risk Limits: {config.trading.max_order_value}")
```

## 📊 Performance Metrics

The backtesting engine provides comprehensive quantitative analysis:

### Return Metrics
- **Total Return**: Cumulative portfolio performance
- **Annualized Return**: Time-weighted annual performance
- **Alpha**: Excess return over benchmark

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns (Return ÷ Volatility)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: Return ÷ Maximum Drawdown

### Tail Risk Measures
- **Skewness & Kurtosis**: Return distribution characteristics

### Benchmarking
- **Information Ratio**: Active return ÷ Tracking error
- **Beta**: Portfolio sensitivity to market
- **Tracking Error**: Standard deviation of active returns

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```
4. **Make your changes** with proper testing
5. **Commit and push**
   ```bash
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**

### Code Standards

- **Type Hints**: Use modern Python typing
- **Documentation**: Add docstrings to all public functions
- **Testing**: Write tests for new features
- **Style**: Follow PEP 8 with Black formatting

### Adding New Strategies

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyCustomStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # Your strategy logic here
        pass
```

## 📋 Roadmap

### Completed Features ✅
- ✅ Modular strategy framework
- ✅ ML-based stock selection strategies
- ✅ Professional backtesting system (powered by bt library)
- ✅ Alpaca live trading integration
- ✅ Multi-source data support (Yahoo/FMP/WRDS)
- ✅ Comprehensive risk management system
- ✅ Performance analysis and reporting

### Planned Enhancements 🚧
- 🔄 Deep reinforcement learning strategies
- 🔄 Alternative data integration
- 🔄 Multi-asset support (crypto, futures)
- 🔄 Advanced portfolio optimization algorithms
- 🔄 Real-time alerting system
- 🔄 Web visualization interface
- 🔄 Docker containerization

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

**⚠️ NOT FINANCIAL ADVICE**

This software is for **educational and research purposes only**.

**Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📚 References & Acknowledgments

### Academic Papers
- [Machine Learning for Stock Recommendation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3302088) - Machine learning approaches to stock selection
- [FinRL: Deep Reinforcement Learning Framework](https://arxiv.org/abs/2011.09607) - Deep RL framework for quantitative trading
- [Portfolio Allocation with Deep Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) - Portfolio optimization research

### Open Source Projects
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Deep reinforcement learning framework for quantitative trading
- [Alpaca-py](https://github.com/alpacahq/alpaca-py) - Alpaca trading API
- [bt](https://github.com/pmorissette/bt) - Flexible backtesting framework for Python

### Data Sources
- [Yahoo Finance](https://finance.yahoo.com/) - Free financial data
- [Financial Modeling Prep](https://financialmodelingprep.com/) - Professional financial data API
- [WRDS (Wharton Research Data Services)](https://wrds.wharton.upenn.edu/) - Academic financial database
- [Alpaca Markets](https://alpaca.markets/) - Brokerage API and market data

---
<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>
**Built with ❤️ for the quantitative finance community**
