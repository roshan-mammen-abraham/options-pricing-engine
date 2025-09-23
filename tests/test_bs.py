import pytest
from pricing.black_scholes import bs_price, bs_greeks

# Test parameters
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

# --- Test BS price against known values from online calculators. ---
def test_bs_known_values():
    expected_call_price = 10.4506
    expected_put_price = 5.5735
    
    call_price = bs_price(S, K, T, r, sigma, 'call')
    put_price = bs_price(S, K, T, r, sigma, 'put')
    
    assert abs(call_price - expected_call_price) < 0.001
    assert abs(put_price - expected_put_price) < 0.001

def test_put_call_parity():
    # C - P = S - K * exp(-r*T) 
    call_price = bs_price(S, K, T, r, sigma, 'call')
    put_price = bs_price(S, K, T, r, sigma, 'put')
    
    parity_lhs = call_price - put_price
    parity_rhs = S - K * (2.71828 ** (-r * T))  
    
    # Compare with tolerance
    assert abs(parity_lhs - parity_rhs) < 0.01

# --- Test Greeks against known values ---
def test_bs_greeks_values():
    greeks = bs_greeks(S, K, T, r, sigma, 'call')
    
    # Expected values
    expected_delta = 0.6368
    expected_gamma = 0.0187
    expected_vega = 0.3752 # per 1%
    expected_theta = -0.0171 # per day
    
    assert abs(greeks['delta'] - expected_delta) < 0.001
    assert abs(greeks['gamma'] - expected_gamma) < 0.001
    assert abs(greeks['vega'] - expected_vega) < 0.01
    assert abs(greeks['theta'] - expected_theta) < 0.001
# --- Test behavior at expiry T=0.---
def test_zero_time_to_maturity():
    assert bs_price(110, 100, 0, r, sigma, 'call') == 10
    assert bs_price(90, 100, 0, r, sigma, 'call') == 0
    assert bs_price(110, 100, 0, r, sigma, 'put') == 0
    assert bs_price(90, 100, 0, r, sigma, 'put') == 10