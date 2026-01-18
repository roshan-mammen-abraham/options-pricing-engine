"""Stock analyzer class for fetching and analyzing stock data."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RMA.utils import (
    calculate_historical_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_log_returns,
)
from pricing.black_scholes import bs_price, bs_greeks


class StockAnalyzer:
    """Wrapper class for fetching and analyzing stock data."""

    def __init__(self, ticker: str, period: str = "1y"):
        """
        Initialize the stock analyzer.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            period: Time period for historical data (e.g., "1y", "6mo", "3mo")
        """
        self.ticker = ticker.upper()
        self.period = period
        self._stock = yf.Ticker(self.ticker)
        self._history: Optional[pd.DataFrame] = None
        self._info: Optional[Dict] = None

    def fetch_data(self) -> bool:
        """
        Fetch stock data from yfinance.

        Returns:
            True if data was fetched successfully, False otherwise.
        """
        try:
            self._history = self._stock.history(period=self.period)
            if self._history.empty:
                return False
            self._info = self._stock.info
            return True
        except Exception:
            return False

    @property
    def history(self) -> Optional[pd.DataFrame]:
        """Get the historical price data."""
        return self._history

    @property
    def info(self) -> Optional[Dict]:
        """Get the stock info."""
        return self._info

    def get_current_price(self) -> Optional[float]:
        """Get the current/latest stock price."""
        if self._history is None or self._history.empty:
            return None
        return float(self._history["Close"].iloc[-1])

    def get_previous_close(self) -> Optional[float]:
        """Get the previous day's closing price."""
        if self._history is None or len(self._history) < 2:
            return None
        return float(self._history["Close"].iloc[-2])

    def get_price_change(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the daily price change.

        Returns:
            Tuple of (dollar_change, percent_change)
        """
        current = self.get_current_price()
        previous = self.get_previous_close()
        if current is None or previous is None:
            return None, None
        dollar_change = current - previous
        percent_change = (dollar_change / previous) * 100
        return dollar_change, percent_change

    def get_52_week_range(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the 52-week high and low.

        Returns:
            Tuple of (52_week_low, 52_week_high)
        """
        if self._info is None:
            return None, None
        low = self._info.get("fiftyTwoWeekLow")
        high = self._info.get("fiftyTwoWeekHigh")
        return low, high

    def get_volume_info(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get current volume and average volume.

        Returns:
            Tuple of (current_volume, average_volume)
        """
        if self._history is None or self._history.empty:
            return None, None
        current_volume = int(self._history["Volume"].iloc[-1])
        avg_volume = self._info.get("averageVolume") if self._info else None
        return current_volume, avg_volume

    def get_volatility(self, annualize: bool = True) -> Optional[float]:
        """
        Get historical volatility using RMA utils.

        Args:
            annualize: Whether to annualize the volatility

        Returns:
            Historical volatility
        """
        if self._history is None or len(self._history) < 2:
            return None
        prices = self._history["Close"].values
        return calculate_historical_volatility(prices, annualize=annualize)

    def get_rolling_volatility(
        self, windows: list = None
    ) -> Dict[int, np.ndarray]:
        """
        Get rolling volatility for multiple window sizes.

        Args:
            windows: List of window sizes (default: [20, 50, 100])

        Returns:
            Dictionary mapping window size to rolling volatility array
        """
        if windows is None:
            windows = [20, 50, 100]

        if self._history is None or len(self._history) < max(windows):
            return {}

        prices = self._history["Close"].values
        result = {}
        for window in windows:
            if len(prices) >= window + 1:
                vol = calculate_historical_volatility(
                    prices, window=window, annualize=True
                )
                result[window] = vol
        return result

    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> Optional[float]:
        """
        Get the Sharpe ratio using RMA utils.

        Args:
            risk_free_rate: Annualized risk-free rate (default: 5%)

        Returns:
            Sharpe ratio
        """
        if self._history is None or len(self._history) < 2:
            return None
        prices = self._history["Close"].values
        returns = calculate_log_returns(prices)
        return calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate)

    def get_max_drawdown(self) -> Optional[Tuple[float, int, int]]:
        """
        Get maximum drawdown using RMA utils.

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if self._history is None or len(self._history) < 2:
            return None
        prices = self._history["Close"].values
        return calculate_max_drawdown(prices)

    def get_total_return(self) -> Optional[float]:
        """
        Get total return for the period.

        Returns:
            Total return as a decimal (e.g., 0.15 for 15%)
        """
        if self._history is None or len(self._history) < 2:
            return None
        start_price = self._history["Close"].iloc[0]
        end_price = self._history["Close"].iloc[-1]
        return (end_price - start_price) / start_price

    def get_option_price(
        self,
        strike: float,
        days_to_expiry: int,
        option_type: str = "call",
        risk_free_rate: float = 0.05,
    ) -> Optional[float]:
        """
        Get theoretical option price using Black-Scholes.

        Args:
            strike: Strike price
            days_to_expiry: Days until expiration
            option_type: "call" or "put"
            risk_free_rate: Risk-free rate

        Returns:
            Theoretical option price
        """
        current_price = self.get_current_price()
        volatility = self.get_volatility()
        if current_price is None or volatility is None:
            return None

        T = days_to_expiry / 365.0
        return bs_price(
            S=current_price,
            K=strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type,
        )

    def get_option_greeks(
        self,
        strike: float,
        days_to_expiry: int,
        option_type: str = "call",
        risk_free_rate: float = 0.05,
    ) -> Optional[Dict[str, float]]:
        """
        Get option Greeks using Black-Scholes.

        Args:
            strike: Strike price
            days_to_expiry: Days until expiration
            option_type: "call" or "put"
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with delta, gamma, vega, theta, rho
        """
        current_price = self.get_current_price()
        volatility = self.get_volatility()
        if current_price is None or volatility is None:
            return None

        T = days_to_expiry / 365.0
        return bs_greeks(
            S=current_price,
            K=strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type,
        )

    def get_company_name(self) -> str:
        """Get the company name."""
        if self._info is None:
            return self.ticker
        return self._info.get("longName", self._info.get("shortName", self.ticker))

    def get_market_cap(self) -> Optional[float]:
        """Get market capitalization."""
        if self._info is None:
            return None
        return self._info.get("marketCap")

    def get_sector(self) -> Optional[str]:
        """Get the company sector."""
        if self._info is None:
            return None
        return self._info.get("sector")

    def get_industry(self) -> Optional[str]:
        """Get the company industry."""
        if self._info is None:
            return None
        return self._info.get("industry")
