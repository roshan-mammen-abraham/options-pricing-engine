```
# Quant Finance Milestone Project

This repository contains a production-grade Python project for quantitative finance, combining an advanced options pricing engine and a comprehensive equity factor model backtesting system.

## 🎯 Project Features

### 1. Options Pricing Engine
- **Black-Scholes Model**: Closed-form pricing for European call and put options.
- **Greeks**: Calculation of Delta, Gamma, Vega, Theta, and Rho.
- **Monte Carlo Pricer**:
    - Geometric Brownian Motion (GBM) path simulation.
    - **Variance Reduction**: Implements Antithetic Variates and Control Variates (using Black-Scholes as the control).
    - **Monte Carlo Greeks**: Calculated via the finite difference (bump-and-reprice) method.
    - **Convergence Diagnostics**: Includes standard error and confidence intervals.

### 2. Equity Factor Model Backtest
- **Data Ingestion**: Fetches asset prices from Yahoo Finance and Fama-French/Carhart factors from the Ken French Data Library.
- **Factor Modeling**: Performs rolling regressions for Fama-French 3-Factor and Carhart 4-Factor models.
- **Signal Generation**: Generates trading signals based on factor exposures (e.g., momentum, alpha).
- **Portfolio Construction**:
    - Equal Weight
    - Mean-Variance Optimization
    - Risk Parity
- **Backtest Engine**:
    - Simulates portfolio performance with rebalancing logic.
    - Accounts for transaction costs and turnover.
    - Calculates key performance metrics (Annualized Return, Volatility, Sharpe Ratio, Max Drawdown).
- **Performance Analytics**: Generates plots for rolling factor exposures and portfolio performance attribution.

## 🗂 Repository Structure

```
project-root/
├─ pricing/                 # Options pricing engine modules
├─ backtest/               # Factor model and backtesting modules
├─ data/                   # Data fetching and caching
├─ notebooks/              # Jupyter notebooks for demonstration and analysis
├─ scripts/                # Standalone scripts for running tasks
├─ tests/                  # Unit and integration tests
├─ requirements.txt        # Project dependencies
├─ pyproject.toml          # Project configuration (linting, formatting)
├─ README.md               # This file
└─ .github/workflows/      # GitHub Actions CI workflow
```

## ⚙️ Tech Stack

- **Core**: Python 3.10+
- **Numerical & Data**: `numpy`, `pandas`, `scipy`, `statsmodels`
- **Data Sources**: `yfinance`, `pandas-datareader`
- **Plotting**: `matplotlib`, `plotly`
- **Testing**: `pytest`
- **Code Quality**: `black`, `isort`, `pre-commit`
- **CI/CD**: GitHub Actions

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/timothykimutai/options-pricing-engine.git
cd options-pricing-engine
```

### 2. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Pre-Commit Hooks (Optional but Recommended)

This will automatically format your code before each commit.

```bash
pip install pre-commit
pre-commit install
```

### 5. Run a Workflow

#### Fetching Data

First, you need to download the required financial data. The script saves it to the `data/cache/` directory.

```bash
python scripts/fetch_data.py
```

#### Running the Backtest

Execute the factor model backtest using the cached data.

```bash
python scripts/run_backtest.py
```

#### Running Tests

To ensure everything is working correctly, run the test suite.

```bash
pytest
```

#### Exploring Notebooks

Launch Jupyter and navigate to the `notebooks/` directory to explore the demonstrations.

```bash
jupyter lab
```

## ✅ Continuous Integration

This project uses GitHub Actions for CI. The workflow automatically runs on every push and pull request to:

- **Lint**: Check code formatting with black and isort.
- **Test**: Run the entire test suite with pytest.

This ensures code quality and correctness are maintained.

## 📋 Dependencies

The project's Python dependencies are listed in `requirements.txt`:

```
# Core numerical and data analysis libraries
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
statsmodels>=0.13.0

# Data fetching
yfinance>=0.2.0
requests>=2.28.0

# Plotting
matplotlib>=3.6.0
plotly>=5.10.0

# Testing and code quality
pytest>=7.0.0
black>=22.10.0
isort>=5.10.0
pre-commit>=2.20.0

# Jupyter environment
jupyterlab>=3.5.0
ipykernel>=6.15.0
```

## 📊 Project Highlights

- **Production-Ready Code**: Modular architecture with comprehensive testing
- **Real Financial Data**: Integration with Yahoo Finance and academic factor data
- **Advanced Modeling**: Implements both theoretical and numerical pricing methods
- **Professional Backtesting**: Complete pipeline from data to performance analytics
- **Extensible Design**: Easy to add new models, factors, or trading strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Black, Scholes, and Merton for the foundational options pricing model
- Fama and French for the factor modeling framework
- Yahoo Finance for providing free financial data
- The Python quant finance community for excellent libraries and resources
```

