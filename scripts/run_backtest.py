import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtest.factors import (prepare_data_for_regression, run_rolling_factor_regression,
                            get_available_factor_model, calculate_expected_returns)
from data.fetch import fetch_stock_data, fetch_fama_french_factors

def main():
    print("--- Starting Factor Model Backtest ---\n")
    
    # Define parameters
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    START_DATE = '2010-01-01'
    END_DATE = '2024-12-31'
    
    try:
        # [1/4] Load data
        print("[1/4] Loading data...")
        stock_prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
        ff_factors = fetch_fama_french_factors(START_DATE, END_DATE)
        
        print(f"Available factors: {list(ff_factors.columns)}")
        
        # Determine which model to use based on available factors
        model_type = get_available_factor_model(ff_factors)
        print(f"Using {model_type} model based on available factors")
        
        # [2/4] Prepare data
        print("\n[2/4] Preparing data...")
        regression_data = prepare_data_for_regression(stock_prices, ff_factors)
        print(f"Regression data shape: {regression_data.shape}")
        
        # [3/4] Run rolling regressions
        print("\n[3/4] Running rolling factor regressions...")
        all_exposures = {}
        
        for ticker in TICKERS:
            if ticker in stock_prices.columns:
                print(f"  Processing {ticker}...")
                exposures = run_rolling_factor_regression(regression_data, ticker, model_type, window=36)
                all_exposures[ticker] = exposures
                print(f"    {ticker}: {len(exposures)} monthly exposures")
        
        # [4/4] Calculate expected returns
        print("\n[4/4] Calculating expected returns...")
        
        # Calculate historical factor returns (simple average)
        factor_returns = {}
        if 'Mkt-RF' in ff_factors.columns:
            factor_returns['Mkt-RF'] = ff_factors['Mkt-RF'].mean()
        if 'SMB' in ff_factors.columns:
            factor_returns['SMB'] = ff_factors['SMB'].mean()
        if 'HML' in ff_factors.columns:
            factor_returns['HML'] = ff_factors['HML'].mean()
        if 'Mom' in ff_factors.columns:
            factor_returns['Mom'] = ff_factors['Mom'].mean()
        
        factor_returns_series = pd.Series(factor_returns)
        print(f"Historical factor returns:\n{factor_returns_series}")
        
        # Calculate expected returns for each stock
        expected_returns_summary = {}
        for ticker, exposures in all_exposures.items():
            exp_returns = calculate_expected_returns(exposures, factor_returns_series)
            expected_returns_summary[ticker] = exp_returns
            
            if len(exp_returns) > 0:
                print(f"  {ticker}: Avg expected return = {exp_returns.mean():.4f}")
        
        print("\n--- Backtest Complete ---")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        raise

if __name__ == '__main__':
    main()