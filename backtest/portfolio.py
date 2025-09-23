
import pandas as pd
import numpy as np
from scipy.optimize import minimize
# --- Assign equal weight to all assets in the portfolio ---
def equal_weight(signals: pd.Series) -> pd.Series:
    return pd.Series(1.0 / len(signals), index=signals.index)

def mean_variance_optimization(
    expected_returns: pd.Series, cov_matrix: pd.DataFrame
) -> pd.Series:
    """
    Calculates portfolio weights using mean-variance optimization.
    Maximizes Sharpe ratio.
    """
    num_assets = len(expected_returns)
    args = (expected_returns.values, cov_matrix.values)

    def portfolio_volatility(weights, expected_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def sharpe_ratio(weights, expected_returns, cov_matrix):
        p_return = np.sum(expected_returns * weights)
        p_vol = portfolio_volatility(weights, expected_returns, cov_matrix)
        return -p_return / p_vol 

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(
        sharpe_ratio, initial_guess, args=args,
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    return pd.Series(result.x, index=expected_returns.index)

def risk_parity(cov_matrix: pd.DataFrame) -> pd.Series:
    num_assets = len(cov_matrix)
    
    def total_risk_contribution(weights, cov_matrix):
        # minimized: sum of squared differences in risk contributions
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_risk = (cov_matrix @ weights) / portfolio_vol
        risk_contributions = weights * marginal_risk
        
        # Target contribution is equal for all
        target_contribution = portfolio_vol / num_assets
        return np.sum((risk_contributions - target_contribution)**2)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        total_risk_contribution, initial_guess, args=(cov_matrix.values,),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    
    # Normalize weights to sum to 1
    weights = result.x / np.sum(result.x)
    return pd.Series(weights, index=cov_matrix.columns)