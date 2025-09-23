# Expose functions from black_scholes at the package level
from .black_scholes import bs_price, bs_greeks

__all__ = ["bs_price", "bs_greeks"]
__version__ = "0.1.0"