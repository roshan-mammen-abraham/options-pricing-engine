import numpy as np
from scipy.stats import norm
from typing import Dict, Literal

OptionType = Literal["call", "put"]

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    if T == 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> Dict[str, float]:
    if T == 0:
        return {'delta': 1 if S > K else 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    nd1 = norm.pdf(d1)

    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega = S * nd1 * np.sqrt(T) / 100  # Per 1% change in vol
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (- (S * nd1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }