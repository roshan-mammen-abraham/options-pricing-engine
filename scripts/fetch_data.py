import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.fetch import fetch_stock_data, fetch_fama_french_factors

if __name__ == "__main__":
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    START_DATE = '2010-01-01'
    END_DATE = '2024-12-31'

    print("--- Starting Data Fetching Process ---")
    
    # 1. Fetch Stock Prices
    print("\nFetching stock prices...")
    stock_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    if not stock_data.empty:
        print(f"Successfully fetched stock data. Shape: {stock_data.shape}")
    else:
        print("Failed to fetch stock data.")
        
    # 2. Fetch Fama-French Factors
    print("\nFetching Fama-French factors...")
    ff_factors = fetch_fama_french_factors(START_DATE, END_DATE)
    if not ff_factors.empty:
        print(f"Successfully fetched Fama-French factors. Shape: {ff_factors.shape}")
    else:
        print("Failed to fetch Fama-French factors.")

    print("\n--- Data Fetching Complete ---")