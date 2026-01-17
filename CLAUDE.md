# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests with coverage
pytest tests/ -v --cov=./ --cov-report=xml

# Run a single test file
pytest tests/test_bs.py -v

# Code formatting (check)
black . --check --diff
isort . --check-only --diff

# Code formatting (apply)
black .
isort .

# Security check
pip install safety && safety check --full-report
```

## Architecture Overview

This is a quantitative finance library with two main capabilities: **options pricing** and **equity factor backtesting**.

### Module Structure

**pricing/** - Options pricing models (independent module)
- `black_scholes.py`: Closed-form European option pricing (`bs_price()`, `bs_greeks()`)
- `mc_pricer.py`: Monte Carlo simulation with antithetic variates and control variate variance reduction

**backtest/** - Portfolio backtesting and factor analysis
- `factors.py`: Rolling Fama-French factor regressions (FF3/Carhart4, 36-month window)
- `portfolio.py`: Portfolio construction (equal weight, mean-variance optimization, risk parity)
- `engine.py`: Main backtest runner with transaction costs and rebalancing

**data/** - Data fetching and caching
- `fetch.py`: Retrieves stock prices (yfinance) and Fama-French factors (Ken French data library)
- `cache/`: Local cache for fetched data

**RMA/** - Dashboard utilities with error handling wrappers

### Data Flow (Backtesting)

```
yfinance/Ken French → fetch.py → factors.py (rolling regression)
    → Generate signals from alpha exposures
    → portfolio.py (construct weights)
    → engine.py (execute backtest)
    → Performance metrics
```

## Key Technical Details

- Python 3.10+ required (CI tests on 3.10 and 3.11)
- Black formatter with 88-character line length, isort with "black" profile
- Monte Carlo pricer implements variance reduction techniques for convergence
- Factor models support both Fama-French 3-factor and Carhart 4-factor specifications
- Tests validate financial properties (e.g., put-call parity in `test_bs.py`)
