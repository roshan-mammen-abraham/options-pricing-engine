# Options Pricing Engine

A comprehensive Python framework for quantitative finance, featuring advanced options pricing models and equity factor backtesting systems.

## 🏷️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy)
![StatsModels](https://img.shields.io/badge/StatsModels-Statistical%20Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?logo=matplotlib)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-blue?logo=plotly)
![yFinance](https://img.shields.io/badge/yFinance-Market%20Data-green)
![pytest](https://img.shields.io/badge/pytest-Testing-orange?logo=pytest)


## 🎯 Project Features

### 1. Options Pricing Engine
- **Black-Scholes Model**: Closed-form pricing for European options
- **Binomial Tree Model**: Discrete-time options pricing
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, and Rho risk metrics
- **Monte Carlo Simulation**: Path-dependent options pricing with variance reduction techniques

### 2. Equity Factor Model Backtest
- **Data Pipeline**: Automated fetching of stock prices and Fama-French factors
- **Factor Analysis**: Rolling regressions for Fama-French 3-Factor model
- **Signal Generation**: Alpha-based trading signals from factor exposures
- **Portfolio Backtesting**: Complete performance simulation with transaction costs
- **Performance Analytics**: Comprehensive metrics and visualization

## 🗂 Repository Structure
```
options-pricing-engine/
├── pricing/               # Options pricing models
│   ├── black_scholes.py
│   ├── binomial.py
│   └── monte_carlo.py
├── backtest/             # Factor model backtesting
│   ├── factors.py
│   ├── portfolio.py
│   ├── engine.py
│   └── performance.py
├── data/                 # Data management
│   ├── fetch.py
│   └── cache/           # Cached financial data
├── notebooks/            # Jupyter notebooks for analysis
│   └── factor_model_backtest.ipynb
├── scripts/              # Execution scripts
│   ├── fetch_data.py
│   └── run_backtest.py
├── tests/               # Test suite
├── utils/               # Utility functions
│   └── logging.py
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── .gitignore
```

## ⚙️ Tech Stack

- **Python 3.10+** - Core programming language
- **pandas & numpy** - Data manipulation and numerical computing
- **scipy & statsmodels** - Statistical analysis and optimization
- **yfinance** - Market data acquisition
- **matplotlib & plotly** - Data visualization
- **pytest** - Testing framework

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/timothykimutai/options-pricing-engine.git
cd options-pricing-engine
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Fetch Market Data

```bash
python scripts/fetch_data.py
```

### 5. Run Factor Model Backtest

```bash
python scripts/run_backtest.py
```

### 6. Explore Analysis Notebooks

```bash
jupyter lab notebooks/
```

## 📊 Usage Examples

### Options Pricing

```python
from pricing.black_scholes import BlackScholes

bs = BlackScholes()
call_price = bs.price_call(S=100, K=105, T=1, r=0.05, sigma=0.2)
print(f"Call option price: ${call_price:.2f}")
```

### Factor Backtesting

```python
from backtest.engine import run_backtest
from data.fetch import fetch_stock_data, fetch_fama_french_factors

# Load data
stock_prices = fetch_stock_data(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2023-12-31')
factors = fetch_fama_french_factors('2020-01-01', '2023-12-31')

# Run backtest
results = run_backtest(stock_prices, factors)
```

## 📈 Project Highlights

- **Production-Ready Architecture**: Modular design with clear separation of concerns
- **Comprehensive Testing**: Unit tests for all major components
- **Real-World Data**: Integration with live market data sources
- **Academic Rigor**: Implementation of proven financial models
- **Extensible Framework**: Easy to add new models or strategies

## 🧪 Testing

Run the test suite to verify everything works correctly:

```bash
pytest tests/ -v
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- **Code Documentation**: All major functions include docstrings
- **Example Notebooks**: See `notebooks/` for detailed usage examples
- **Inline Comments**: Code is thoroughly commented for clarity

## 🐛 Bug Reports

If you encounter any bugs or have suggestions, please open an issue on GitHub.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Black, Scholes, and Merton for foundational options pricing theory
- Fama and French for factor modeling framework
- Yahoo Finance for providing free financial data
- The Python quant finance community for excellent educational resources

---
