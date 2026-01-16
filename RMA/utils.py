"""
Utility functions for options pricing and risk management analysis.

This module contains helper functions for various calculations and operations
related to options pricing, including statistical calculations, validation,
and data transformation utilities.
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_positive(value: float, param_name: str) -> None:
    """
    Validate that a parameter is positive.

    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def validate_probability(value: float, param_name: str) -> None:
    """
    Validate that a value is between 0 and 1 (inclusive).

    Args:
        value: The value to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not in [0, 1]
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{param_name} must be between 0 and 1, got {value}")


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate logarithmic returns from price series.

    Args:
        prices: Array of prices

    Returns:
        Array of log returns
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to calculate returns")

    return np.diff(np.log(prices))


def calculate_simple_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate simple returns from price series.

    Args:
        prices: Array of prices

    Returns:
        Array of simple returns
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to calculate returns")

    return np.diff(prices) / prices[:-1]


def annualize_volatility(volatility: float, periods_per_year: int = 252) -> float:
    """
    Annualize volatility from a different time period.

    Args:
        volatility: Volatility in the given period
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)

    Returns:
        Annualized volatility
    """
    return volatility * np.sqrt(periods_per_year)


def calculate_historical_volatility(
    prices: np.ndarray,
    window: Optional[int] = None,
    annualize: bool = True,
    periods_per_year: int = 252
) -> Union[float, np.ndarray]:
    """
    Calculate historical volatility from price series.

    Args:
        prices: Array of prices
        window: Rolling window size (if None, uses all data)
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year for annualization

    Returns:
        Historical volatility (scalar or array if window is specified)
    """
    log_returns = calculate_log_returns(prices)

    if window is None:
        vol = np.std(log_returns, ddof=1)
        if annualize:
            vol = annualize_volatility(vol, periods_per_year)
        return vol
    else:
        # Rolling volatility
        rolling_vol = np.array([
            np.std(log_returns[max(0, i-window):i], ddof=1)
            for i in range(window, len(log_returns) + 1)
        ])
        if annualize:
            rolling_vol = annualize_volatility(rolling_vol, periods_per_year)
        return rolling_vol


def time_to_maturity(days: int) -> float:
    """
    Convert days to years (for option pricing formulas).

    Args:
        days: Number of days to expiration

    Returns:
        Time in years
    """
    return days / 365.0


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_option_payoff(
    spot_prices: np.ndarray,
    strike: float,
    option_type: str = "call"
) -> np.ndarray:
    """
    Calculate option payoff at expiration.

    Args:
        spot_prices: Array of spot prices at expiration
        strike: Strike price
        option_type: Either "call" or "put"

    Returns:
        Array of payoffs
    """
    option_type = option_type.lower()

    if option_type == "call":
        return np.maximum(spot_prices - strike, 0)
    elif option_type == "put":
        return np.maximum(strike - spot_prices, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def calculate_profit_loss(
    payoffs: np.ndarray,
    premium: float,
    position: str = "long"
) -> np.ndarray:
    """
    Calculate profit/loss including premium paid/received.

    Args:
        payoffs: Array of option payoffs
        premium: Option premium
        position: Either "long" or "short"

    Returns:
        Array of profit/loss values
    """
    position = position.lower()

    if position == "long":
        return payoffs - premium
    elif position == "short":
        return premium - payoffs
    else:
        raise ValueError(f"position must be 'long' or 'short', got {position}")


def calculate_breakeven(
    strike: float,
    premium: float,
    option_type: str = "call"
) -> float:
    """
    Calculate breakeven price for an option position.

    Args:
        strike: Strike price
        premium: Option premium
        option_type: Either "call" or "put"

    Returns:
        Breakeven price
    """
    option_type = option_type.lower()

    if option_type == "call":
        return strike + premium
    elif option_type == "put":
        return strike - premium
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def interpolate_volatility(
    strikes: np.ndarray,
    volatilities: np.ndarray,
    target_strike: float
) -> float:
    """
    Interpolate implied volatility for a given strike.

    Args:
        strikes: Array of strike prices
        volatilities: Array of implied volatilities
        target_strike: Strike price to interpolate for

    Returns:
        Interpolated volatility
    """
    return np.interp(target_strike, strikes, volatilities)


def generate_price_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stock price paths using geometric Brownian motion.

    Args:
        S0: Initial stock price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon in years
        steps: Number of time steps
        paths: Number of paths to simulate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (paths, steps+1) containing price paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    prices = np.zeros((paths, steps + 1))
    prices[:, 0] = S0

    for t in range(1, steps + 1):
        z = np.random.standard_normal(paths)
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return prices


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(excess_returns, ddof=1) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)


def calculate_max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from price series.

    Args:
        prices: Array of prices

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    cumulative_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_max) / cumulative_max

    max_drawdown_idx = np.argmin(drawdowns)
    max_drawdown = drawdowns[max_drawdown_idx]

    peak_idx = np.argmax(prices[:max_drawdown_idx + 1])

    return abs(max_drawdown), peak_idx, max_drawdown_idx


if __name__ == "__main__":
    # Example usage
    logger.info("RMA Utils module loaded successfully")

    # Example: Calculate historical volatility
    sample_prices = np.array([100, 102, 101, 103, 105, 104, 106])
    vol = calculate_historical_volatility(sample_prices)
    logger.info(f"Sample historical volatility: {format_percentage(vol)}")
