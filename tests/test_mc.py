import pytest
import numpy as np
from pricing.mc_pricer import mc_price_european
from pricing.black_scholes import bs_price

# Test parameters
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
N_SIMS = 50000 

@pytest.fixture(scope="module")
# --- Calculate the BS price once for all tests in this module ---
def bs_call_price_fixture():
    return bs_price(S, K, T, r, sigma, 'call')
# --- Test if MC price converges to BS price for a large number of simulations. ---
def test_mc_convergence(bs_call_price_fixture):
    np.random.seed(42) # for reproducibility
    mc_result = mc_price_european(S, K, T, r, sigma, 'call', n_sims=N_SIMS, use_antithetic=True)
    
    mc_price = mc_result['price']
    
    # Check if MC price is within 1% of the BS price
    assert abs(mc_price - bs_call_price_fixture) / bs_call_price_fixture < 0.01
# --- Test if variance reduction techniques (antithetic variates) reduce standard error. ---
def test_variance_reduction():
    np.random.seed(42)
    mc_standard = mc_price_european(S, K, T, r, sigma, 'call', n_sims=10000, use_antithetic=False)
    
    np.random.seed(42)
    mc_antithetic = mc_price_european(S, K, T, r, sigma, 'call', n_sims=10000, use_antithetic=True)
    
    assert mc_antithetic['std_error'] < mc_standard['std_error']
# --- Test if the true BS price falls within the 95% confidence interval. ---
def test_confidence_interval(bs_call_price_fixture):
    np.random.seed(123)
    mc_result = mc_price_european(S, K, T, r, sigma, 'call', n_sims=N_SIMS)
    
    ci_lower, ci_upper = mc_result['confidence_interval_95']
    
    assert ci_lower <= bs_call_price_fixture <= ci_upper