import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define cache directory
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- Fetch daily adjusted close prices for a list of tickers from Yahoo Finance. Save the data to a CSV file in the cache directory. ---  
def fetch_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    
    filepath = CACHE_DIR / "stock_prices.csv"
    if filepath.exists():
        logging.info("Loading stock data from cache.")
        return pd.read_csv(filepath, index_col=0, parse_dates=True)

    logging.info(f"Fetching stock data for {tickers} from {start_date} to {end_date}.")
    try:
        # Download data with auto_adjust=True (default behavior in newer yfinance)
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Handle the column naming change in yfinance
        if 'Adj Close' in data.columns:
            adj_close = data['Adj Close']
        else:
            adj_close = data['Close']
        
        adj_close.to_csv(filepath)
        logging.info(f"Stock data saved to {filepath}")
        return adj_close
        
    except Exception as e:
        logging.error(f"Failed to fetch stock data: {e}")
        return pd.DataFrame()

# --- Fetch Fama-French 3 Factors and Momentum Factor directly from Ken French's website ---
def fetch_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    filepath = CACHE_DIR / "fama_french_factors.csv"
    if filepath.exists():
        logging.info("Loading Fama-French factors from cache.")
        factors_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        # Filter by date range in case cached data has different range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        factors_df = factors_df[(factors_df.index >= start) & (factors_df.index <= end)]
        return factors_df

    logging.info("Fetching Fama-French factors from Ken French's data library...")
    try:
        # Use the simpler alternative method that worked
        return fetch_fama_french_factors_alternative(start_date, end_date)
        
    except Exception as e:
        logging.error(f"Failed to fetch Fama-French factors: {e}")
        logging.info("Attempting alternative method...")
        return fetch_fama_french_factors_alternative(start_date, end_date)

# --- Improved alternative method ---
def fetch_fama_french_factors_alternative(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        # Direct CSV download for 3 factors (non-zipped version)
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily.CSV"
        
        response = requests.get(url)
        if response.status_code == 200:
            # Read the data, skipping the header lines
            lines = response.text.split('\n')
            data_lines = []
            skip_lines = 4  # Skip copyright and header lines
            
            for i, line in enumerate(lines):
                if i >= skip_lines and line.strip():
                    # Check if line contains data (starts with a digit)
                    if line.strip() and line.strip()[0].isdigit():
                        data_lines.append(line)
            
            # Create DataFrame from data lines
            if data_lines:
                df = pd.read_csv(StringIO('\n'.join(data_lines)), 
                               header=None, names=['Date', 'Mkt-RF', 'SMB', 'HML', 'RF'],
                               sep=',', engine='python')
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                df = df.dropna(subset=['Date'])  # Remove invalid dates
                df.set_index('Date', inplace=True)
                
                # Convert percentages to decimals and ensure numeric data
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                
                # Filter by date range
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                factors_df = df[(df.index >= start) & (df.index <= end)]
                
                # Save to cache
                factors_df.to_csv(CACHE_DIR / "fama_french_factors.csv")
                logging.info(f"Fama-French factors saved to {CACHE_DIR / 'fama_french_factors.csv'} (alternative method)")
                logging.info(f"Factors data range: {factors_df.index.min()} to {factors_df.index.max()}")
                return factors_df
            else:
                logging.error("No data lines found in the response")
                
    except Exception as e:
        logging.error(f"Alternative method failed: {e}")
    
    # Return empty DataFrame if all methods fail
    logging.error("All methods to fetch Fama-French factors failed")
    return pd.DataFrame()

def main():
    print("--- Starting Data Fetching Process ---\n")
    
    # Define parameters
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    START_DATE = '2010-01-01'
    END_DATE = '2024-12-31'
    
    # Fetch stock data
    print("Fetching stock prices...")
    stock_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    
    if not stock_data.empty:
        print(f"Successfully fetched stock data. Shape: {stock_data.shape}")
        print(f"Data range: {stock_data.index.min()} to {stock_data.index.max()}")
    else:
        print("Failed to fetch stock data.")
    
    print("\nFetching Fama-French factors...")
    factors_data = fetch_fama_french_factors(START_DATE, END_DATE)
    
    if not factors_data.empty:
        print(f"Successfully fetched Fama-French factors. Shape: {factors_data.shape}")
        print(f"Available factors: {list(factors_data.columns)}")
        print(f"Factors data range: {factors_data.index.min()} to {factors_data.index.max()}")
    else:
        print("Failed to fetch Fama-French factors.")
    
    print("\n--- Data Fetching Complete ---")

if __name__ == '__main__':
    main()