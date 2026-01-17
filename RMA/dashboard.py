"""
Options Pricing Dashboard

This module provides a dashboard interface for options pricing analysis,
importing and utilizing functions from utils.py with comprehensive error handling.
"""

import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Error handling for imports from utils.py
try:
    from utils import (
        validate_positive,
        validate_probability,
        calculate_log_returns,
        calculate_simple_returns,
        annualize_volatility,
        calculate_historical_volatility,
        time_to_maturity,
        format_percentage,
        calculate_option_payoff,
        calculate_profit_loss,
        calculate_breakeven,
        interpolate_volatility,
        generate_price_paths,
        calculate_sharpe_ratio,
        calculate_max_drawdown
    )
    logger.info("Successfully imported all functions from utils.py")
    UTILS_AVAILABLE = True
except ModuleNotFoundError as e:
    logger.error(f"utils.py module not found: {e}")
    logger.error("Please ensure utils.py is in the same directory as dashboard.py")
    UTILS_AVAILABLE = False
except ImportError as e:
    logger.error(f"Error importing specific functions from utils.py: {e}")
    logger.error("Some functions may not be available in utils.py")
    UTILS_AVAILABLE = False
except Exception as e:
    logger.error(f"Unexpected error while importing from utils.py: {e}")
    UTILS_AVAILABLE = False

# Try importing numpy with error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy imported successfully")
except ImportError as e:
    logger.error(f"NumPy is not available: {e}")
    logger.error("Install numpy using: pip install numpy")
    NUMPY_AVAILABLE = False
    np = None


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.

    Returns:
        True if all dependencies are available, False otherwise
    """
    if not UTILS_AVAILABLE:
        logger.error("Utils module is not available. Dashboard functionality is limited.")
        return False

    if not NUMPY_AVAILABLE:
        logger.error("NumPy is not available. Dashboard functionality is limited.")
        return False

    logger.info("All dependencies are available")
    return True


def safe_calculate_volatility(prices, **kwargs):
    """
    Safely calculate historical volatility with error handling.

    Args:
        prices: Array of prices
        **kwargs: Additional arguments for calculate_historical_volatility

    Returns:
        Volatility value or None if calculation fails
    """
    if not UTILS_AVAILABLE:
        logger.error("Cannot calculate volatility: utils module not available")
        return None

    if not NUMPY_AVAILABLE:
        logger.error("Cannot calculate volatility: NumPy not available")
        return None

    try:
        volatility = calculate_historical_volatility(prices, **kwargs)
        logger.info(f"Successfully calculated volatility: {volatility}")
        return volatility
    except ValueError as e:
        logger.error(f"ValueError in volatility calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in volatility calculation: {e}")
        return None


def safe_calculate_option_value(spot_prices, strike, option_type="call"):
    """
    Safely calculate option payoff with error handling.

    Args:
        spot_prices: Array of spot prices
        strike: Strike price
        option_type: "call" or "put"

    Returns:
        Option payoff array or None if calculation fails
    """
    if not UTILS_AVAILABLE:
        logger.error("Cannot calculate option payoff: utils module not available")
        return None

    try:
        payoff = calculate_option_payoff(spot_prices, strike, option_type)
        logger.info(f"Successfully calculated {option_type} option payoff")
        return payoff
    except ValueError as e:
        logger.error(f"ValueError in option payoff calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in option payoff calculation: {e}")
        return None


def safe_generate_price_paths(S0, mu, sigma, T, steps, paths, seed=None):
    """
    Safely generate price paths with error handling.

    Args:
        S0: Initial stock price
        mu: Drift
        sigma: Volatility
        T: Time horizon
        steps: Number of steps
        paths: Number of paths
        seed: Random seed

    Returns:
        Price paths array or None if generation fails
    """
    if not UTILS_AVAILABLE:
        logger.error("Cannot generate price paths: utils module not available")
        return None

    try:
        price_paths = generate_price_paths(S0, mu, sigma, T, steps, paths, seed)
        logger.info(f"Successfully generated {paths} price paths with {steps} steps")
        return price_paths
    except ValueError as e:
        logger.error(f"ValueError in price path generation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in price path generation: {e}")
        return None


def safe_calculate_metrics(prices, risk_free_rate=0.0):
    """
    Safely calculate various risk metrics with error handling.

    Args:
        prices: Array of prices
        risk_free_rate: Risk-free rate for Sharpe ratio

    Returns:
        Dictionary of metrics or None if calculation fails
    """
    if not UTILS_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("Cannot calculate metrics: required modules not available")
        return None

    metrics = {}

    try:
        # Calculate returns
        log_returns = calculate_log_returns(prices)
        simple_returns = calculate_simple_returns(prices)

        # Calculate Sharpe ratio
        sharpe = calculate_sharpe_ratio(log_returns, risk_free_rate)

        # Calculate max drawdown
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(prices)

        metrics = {
            'log_returns_mean': np.mean(log_returns),
            'log_returns_std': np.std(log_returns),
            'simple_returns_mean': np.mean(simple_returns),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_peak_idx': peak_idx,
            'max_drawdown_trough_idx': trough_idx
        }

        logger.info("Successfully calculated all risk metrics")
        return metrics

    except ValueError as e:
        logger.error(f"ValueError in metrics calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in metrics calculation: {e}")
        return None


def main():
    """
    Main function to run the dashboard with comprehensive error handling.
    """
    logger.info("Starting Options Pricing Dashboard")

    # Check dependencies
    if not check_dependencies():
        logger.error("Cannot start dashboard: missing dependencies")
        sys.exit(1)

    # Example usage with error handling
    try:
        logger.info("Running example calculations...")

        # Example data
        sample_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 110])

        # Calculate volatility
        vol = safe_calculate_volatility(sample_prices)
        if vol is not None:
            logger.info(f"Historical Volatility: {format_percentage(vol)}")

        # Calculate metrics
        metrics = safe_calculate_metrics(sample_prices)
        if metrics is not None:
            logger.info(f"Risk Metrics: {metrics}")

        # Generate price paths
        paths = safe_generate_price_paths(
            S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252, paths=1000, seed=42
        )
        if paths is not None:
            logger.info(f"Generated price paths with shape: {paths.shape}")

        # Calculate option payoff
        spot_range = np.linspace(80, 120, 41)
        call_payoff = safe_calculate_option_value(spot_range, strike=100, option_type="call")
        if call_payoff is not None:
            logger.info(f"Call option payoff calculated for {len(spot_range)} prices")

        logger.info("Dashboard examples completed successfully")

    except Exception as e:
        logger.error(f"Error in main dashboard execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
