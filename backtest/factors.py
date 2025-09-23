import pandas as pd
import statsmodels.api as sm
from typing import Literal

FactorModel = Literal["FF3", "Carhart4"]

# --- Prepare data by resampling to monthly, calculating excess returns --
def prepare_data_for_regression(
    stock_prices: pd.DataFrame, ff_factors: pd.DataFrame
) -> pd.DataFrame:
    # Resample stock prices to monthly, using the last price of the month
    monthly_prices = stock_prices.resample('ME').last()  # Changed 'M' to 'ME'
    monthly_returns = monthly_prices.pct_change().dropna()

    # Resample FF factors to monthly by summing up daily factors
    monthly_factors = ff_factors.resample('ME').sum()  # Changed 'M' to 'ME'
    
    # Align data
    data = monthly_returns.join(monthly_factors).dropna()
    
    # Calculate excess returns for each stock
    for stock in monthly_returns.columns:
        data[f'{stock}_excess'] = data[stock] - data['RF']
        
    return data

def run_rolling_factor_regression(
    data: pd.DataFrame,
    ticker: str,
    model: FactorModel,
    window: int = 36  
) -> pd.DataFrame:
    # Perform a rolling OLS regression for a given ticker and factor model.

    y_col = f'{ticker}_excess'
    
    # Check which factors are available in the data
    available_factors = []
    for factor in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
        if factor in data.columns:
            available_factors.append(factor)
    
    # Determine which model to use based on available factors
    if model == "Carhart4" and 'Mom' not in data.columns:
        print(f"Warning: Momentum factor not available. Using FF3 model instead of Carhart4.")
        model = "FF3"
    
    if model == "FF3":
        X_cols = ['Mkt-RF', 'SMB', 'HML']
    elif model == "Carhart4":
        X_cols = ['Mkt-RF', 'SMB', 'HML', 'Mom']
    else:
        raise ValueError("Model must be 'FF3' or 'Carhart4'.")
    
    # Ensure all requested factors are available
    missing_factors = [col for col in X_cols if col not in data.columns]
    if missing_factors:
        raise ValueError(f"Missing factors in data: {missing_factors}. Available factors: {list(data.columns)}")
    
    if y_col not in data.columns:
        raise ValueError(f"Excess returns column '{y_col}' not found in data")

    y = data[y_col]
    X = sm.add_constant(data[X_cols])

    rolling_results = []

    for i in range(window, len(data)):
        window_X = X.iloc[i-window:i]
        window_y = y.iloc[i-window:i]
        
        try:
            model_fit = sm.OLS(window_y, window_X).fit()
            
            result = {
                'date': data.index[i],
                'alpha': model_fit.params['const'],
                'beta_mkt': model_fit.params['Mkt-RF'],
                'beta_smb': model_fit.params['SMB'],
                'beta_hml': model_fit.params['HML'],
                'r_squared': model_fit.rsquared
            }
            
            if model == "Carhart4":
                result['beta_mom'] = model_fit.params['Mom']
                
            rolling_results.append(result)
            
        except Exception as e:
            # If regression fails, use NaN values
            result = {
                'date': data.index[i],
                'alpha': float('nan'),
                'beta_mkt': float('nan'),
                'beta_smb': float('nan'),
                'beta_hml': float('nan'),
                'r_squared': float('nan')
            }
            if model == "Carhart4":
                result['beta_mom'] = float('nan')
                
            rolling_results.append(result)
            print(f"Warning: Regression failed for {ticker} at {data.index[i]}: {e}")

    return pd.DataFrame(rolling_results).set_index('date')

# --- Additional utility function to check available factors ---
def get_available_factor_model(ff_factors: pd.DataFrame) -> FactorModel:
    """
    Determine which factor model is available based on the data.
    """
    if 'Mom' in ff_factors.columns:
        return "Carhart4"
    else:
        return "FF3"

# --- Calculate expected returns from factor exposures ---
def calculate_expected_returns(factor_exposures: pd.DataFrame, 
                             factor_returns: pd.Series) -> pd.Series:
    expected_returns = pd.Series(index=factor_exposures.index, dtype=float)
    
    for date, exposures in factor_exposures.iterrows():
        exp_return = 0
        if 'beta_mkt' in exposures and 'Mkt-RF' in factor_returns:
            exp_return += exposures['beta_mkt'] * factor_returns['Mkt-RF']
        if 'beta_smb' in exposures and 'SMB' in factor_returns:
            exp_return += exposures['beta_smb'] * factor_returns['SMB']
        if 'beta_hml' in exposures and 'HML' in factor_returns:
            exp_return += exposures['beta_hml'] * factor_returns['HML']
        if 'beta_mom' in exposures and 'Mom' in factor_returns:
            exp_return += exposures['beta_mom'] * factor_returns['Mom']
            
        expected_returns[date] = exp_return
    
    return expected_returns