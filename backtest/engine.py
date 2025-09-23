import pandas as pd
import numpy as np

def run_backtest(
    monthly_returns: pd.DataFrame,
    signals: pd.DataFrame,
    portfolio_constructor,
    rebalance_freq: str = 'M',
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    tickers = monthly_returns.columns
    dates = signals.index
    
    portfolio_values = []
    turnovers = []
    current_weights = pd.Series(0, index=tickers)
    last_rebalance_value = 1.0
    pnl = pd.Series(index=monthly_returns.index)
    
    # Align returns with signals
    aligned_returns = monthly_returns.loc[signals.index[0]:]
    
    for i in range(len(aligned_returns)):
        date = aligned_returns.index[i]
        
        # Check if it's a rebalancing date
        if date in dates:
            # Generate target weights from signals
            current_signal = signals.loc[date]
            
            # Simple signal: long top quartile based on alpha signal
            long_stocks = current_signal.nlargest(int(len(current_signal) * 0.25)).index
            signal_series = pd.Series(1, index=long_stocks) # Simplified signal
            
            # For MVO/Risk Parity, we need expected returns and covariance
            if portfolio_constructor.__name__ != 'equal_weight':
                hist_returns = monthly_returns.loc[:date].tail(36)
                exp_returns = hist_returns.mean()
                cov_matrix = hist_returns.cov()
                
                # Filter for stocks in the signal
                exp_returns_filtered = exp_returns[long_stocks]
                cov_matrix_filtered = cov_matrix.loc[long_stocks, long_stocks]
                
                if not cov_matrix_filtered.empty:
                    target_weights_subset = portfolio_constructor(exp_returns_filtered, cov_matrix_filtered)
                    target_weights = pd.Series(0.0, index=tickers)
                    target_weights.update(target_weights_subset)
                else:
                    target_weights = pd.Series(0.0, index=tickers)

            else: 
                if not signal_series.empty:
                    target_weights = equal_weight(signal_series)
                    # Expand to all tickers
                    target_weights = target_weights.reindex(tickers, fill_value=0.0)
                else:
                    target_weights = pd.Series(0.0, index=tickers)
            
            # Calculate turnover and transaction costs
            turnover = np.sum(np.abs(target_weights - current_weights))
            turnovers.append(turnover)
            cost = turnover * transaction_cost
            
            # Update weights and apply costs
            current_weights = target_weights
            pnl.iloc[i] = (1 + aligned_returns.iloc[i] @ current_weights) * (1 - cost) - 1
            
        else:
            pnl.iloc[i] = aligned_returns.iloc[i] @ current_weights

    # Calculate portfolio value series
    portfolio_series = (1 + pnl).cumprod()
    
    return pd.DataFrame({
        'portfolio_value': portfolio_series,
        'pnl': pnl,
    })
# --- Calculate key performance metrics for a portfolio ---
def calculate_performance_metrics(pnl_series: pd.Series) -> dict:
    cumulative_return = (1 + pnl_series).prod() - 1
    annualized_return = (1 + pnl_series.mean())**12 - 1
    annualized_vol = pnl_series.std() * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_vol
    
    # Max drawdown
    cum_returns = (1 + pnl_series).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns/peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }