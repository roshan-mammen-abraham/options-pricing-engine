import numpy as np
from typing import Dict, Tuple, Literal
from .black_scholes import bs_price, OptionType

def gbm_paths(S0: float, T: float, r: float, sigma: float, n_sims: int, n_steps: int) -> np.ndarray:
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate random shocks for all paths and steps at once for efficiency
    Z = np.random.standard_normal((n_sims, n_steps))
    
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    return paths

def mc_price_european(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType,
    n_sims: int = 10000, use_antithetic: bool = True, use_control_variate: bool = True
) -> Dict[str, float]:
    """
    Prices a European option using Monte Carlo simulation with variance reduction.
    """
    dt = T
    
    # 1. Standard Monte Carlo
    Z1 = np.random.standard_normal(n_sims)
    ST1 = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z1)

    if option_type == 'call':
        payoff1 = np.maximum(ST1 - K, 0)
    else: # put
        payoff1 = np.maximum(K - ST1, 0)
    
    # Apply variance reduction techniques
    if use_antithetic:
        # 2. Antithetic Variates
        Z2 = -Z1
        ST2 = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z2)
        if option_type == 'call':
            payoff2 = np.maximum(ST2 - K, 0)
        else: # put
            payoff2 = np.maximum(K - ST2, 0)
        
        # Average the two payoffs
        payoffs = (payoff1 + payoff2) / 2.0
        # Also average the stock prices for control variate
        ST_avg = (ST1 + ST2) / 2.0
    else:
        payoffs = payoff1
        ST_avg = ST1

    # Discount the payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs

    # 3. Control Variate (FIXED)
    if use_control_variate:
        # Use the stock price as control variate
        # We know E[ST] = S * exp(r*T), so E[discounted_ST] = S
        control_variate = np.exp(-r * T) * ST_avg  # Discounted stock price
        expected_control = S  # E[discounted_ST] = S
        
        # Calculate optimal beta using covariance (this is the key fix)
        covariance = np.cov(discounted_payoffs, control_variate)[0, 1]
        variance_control = np.var(control_variate)
        
        # Avoid division by zero
        if variance_control > 1e-10:
            beta = covariance / variance_control
        else:
            beta = 0
            
        # Apply control variate adjustment
        adjusted_payoffs = discounted_payoffs - beta * (control_variate - expected_control)
        price = np.mean(adjusted_payoffs)
        
        # Use adjusted payoffs for error calculation
        final_payoffs = adjusted_payoffs
    else:
        price = np.mean(discounted_payoffs)
        final_payoffs = discounted_payoffs

    # Calculate convergence diagnostics (USE FINAL_PAYOFFS)
    std_err = np.std(final_payoffs) / np.sqrt(n_sims)
    confidence_interval = (price - 1.96 * std_err, price + 1.96 * std_err)

    return {
        "price": price,
        "std_error": std_err,
        "confidence_interval_95": confidence_interval
    }

def mc_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType,
    n_sims: int = 10000, bump: float = 0.01
) -> Dict[str, float]:
    """
    Calculates option Greeks using Monte Carlo with finite differences.
    """
    # Base price
    base_price = mc_price_european(S, K, T, r, sigma, option_type, n_sims, use_antithetic=True, use_control_variate=False)['price']

    # Delta
    price_up = mc_price_european(S * (1 + bump), K, T, r, sigma, option_type, n_sims)['price']
    price_down = mc_price_european(S * (1 - bump), K, T, r, sigma, option_type, n_sims)['price']
    delta = (price_up - price_down) / (2 * S * bump)

    # Gamma
    gamma = (price_up - 2 * base_price + price_down) / ((S * bump) ** 2)

    # Vega
    price_vega_up = mc_price_european(S, K, T, r, sigma + bump, option_type, n_sims)['price']
    vega = (price_vega_up - base_price) / (bump * 100) # Per 1%

    # Theta
    price_theta_down = mc_price_european(S, K, T - bump, r, sigma, option_type, n_sims)['price']
    theta = (price_theta_down - base_price) / (bump * 365) # Per day
    
    # Rho
    price_rho_up = mc_price_european(S, K, T, r + bump, sigma, option_type, n_sims)['price']
    rho = (price_rho_up - base_price) / (bump * 100) # Per 1%

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }