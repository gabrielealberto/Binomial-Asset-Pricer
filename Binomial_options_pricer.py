# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 23:51:32 2026
@author: Gabriele Alberto
"""

"""
Binomial Option Pricing - CRR model (European & American)

Prices call and put options via a recombining binomial tree.
Supports European and American exercise styles with continuous dividend yield.
Vectorized implementation, no full price matrix stored.
"""

# -------------------- LIBRARIES --------------------

import yfinance as yf
import numpy as np
import pandas as pd

# -------------------- DATA DOWNLOAD --------------------

ticker = 'NVDA'

# 5 years of adjusted closes to get a reliable vol estimate
df = yf.download(ticker, period='5y', interval='1d', auto_adjust=True)["Close"]

# -------------------- VOLATILITY ESTIMATION --------------------

log_returns = np.log(df / df.shift(1)).dropna()

# Annualized historical vol — 252 trading days
sigma = log_returns.std().iloc[0] * np.sqrt(252)

# -------------------- DIVIDEND YIELD --------------------

ticker_obj = yf.Ticker(ticker)

q_raw = ticker_obj.info.get('dividendYield', 0)
q = (q_raw / 100) if q_raw is not None else 0   # yfinance returns it as percentage

# -------------------- CURRENCY --------------------

currency = ticker_obj.info.get('currency', 'USD')

# -------------------- SPOT PRICE --------------------

# Re-download without adjustment to get the actual traded price
df = yf.download(ticker, period='5d', interval='1d', auto_adjust=False)["Close"]
s0 = df.iloc[-1].iloc[0]

# -------------------- MODEL PARAMETERS --------------------

k = float(s0 * 1.03)   # strike 3% OTM
n = 500                  # tree steps
t = 90                  # days to expiry
r = 0.04227            # risk-free rate
opttype = 'c'           # 'c' = call, 'p' = put
option_style = "eu"     # 'eu' = European, 'am' = American

# -------------------- BINOMIAL TREE FUNCTION --------------------

def tree_engine(s0, sigma, k, n, t, r, q, opttype='c', option_style="eu"):
    """
    CRR binomial pricer - vectorized.
    No (n+1)x(n+1) matrix is ever built: terminal prices are computed
    directly, and intermediate stock prices (needed only for American
    early exercise) are regenerated on the fly at each backward step.
    """

    dt = (t / 365) / n

    # CRR parametrization: u*d = 1 ensures the tree recombines
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1 / u

    # Risk-neutral probability
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p_inverse = 1 - p

    disc = np.exp(-r * dt)

    # --- Terminal stock prices ---
    # Instead of storing the whole tree, we only need the n+1 terminal nodes.
    # Node j at maturity has made j down-moves: S_T(j) = S0 * u^(n-j) * d^j
    j = np.arange(n + 1)
    sT = s0 * (u ** (n - j)) * (d ** j)

    # --- Terminal payoff ---
    if opttype == 'c':
        v = np.maximum(sT - k, 0.0)
    else:
        v = np.maximum(k - sT, 0.0)

    # --- Backward induction ---
    # At each step the value vector shrinks by one element.
    # The slice v[:i+1] and v[1:i+2] picks up- and down-branches
    # for all nodes simultaneously — no inner loop over j needed.
    style = option_style.lower().strip()

    for i in range(n - 1, -1, -1):

        v = disc * (p * v[:i + 1] + p_inverse * v[1:i + 2])

        if style == "am":
            # For American options we compare continuation vs immediate exercise.
            # Stock price at node (i, j): S0 * u^(i-j) * d^j — computed on the fly
            # to avoid storing the full price matrix.
            jj = np.arange(i + 1)
            s_i = s0 * (u ** (i - jj)) * (d ** jj)
            exercise = np.maximum(s_i - k, 0.0) if opttype == 'c' else np.maximum(k - s_i, 0.0)
            v = np.maximum(v, exercise)

    return None, v.reshape(1, 1)

# -------------------- PRICING --------------------

discounted_tree_c = tree_engine(
    s0, sigma, k, n, t, r, q,
    opttype='c',
    option_style=option_style
)[1]
option_price_c = discounted_tree_c[0, 0]

discounted_tree_p = tree_engine(
    s0, sigma, k, n, t, r, q,
    opttype='p',
    option_style=option_style
)[1]
option_price_p = discounted_tree_p[0, 0]

# -------------------- PUT-CALL PARITY CHECK --------------------

def check_parity(s0, k, r, q, t, call_price, put_price, tol=1e-6):
    # C - P = S*e^(-qT) - K*e^(-rT)
    T = t / 365
    lhs = call_price - put_price
    rhs = (s0 * np.exp(-q * T)) - (k * np.exp(-r * T))
    return abs(lhs - rhs) < tol

# -------------------- OUTPUT --------------------

if option_style.lower() == "eu":
    print('\nOption Style: European')
else:
    print('\nOption Style: American')

print("-------------------------------------------")

print("\nActual underlying price:", round(s0, 4), currency)
print("Dividend Yield used: {:.2%}".format(q))

if opttype.lower() == 'c':
    print("Option type: Call")
    print("Strike price:", round(k, 4), currency)
    print("Binomial Call option price: $", round(option_price_c, 4), currency)
else:
    print("Option type: Put")
    print("Strike price:", round(k, 4), currency)
    print("Binomial Put option price:", round(option_price_p, 4), currency)

parity_check = check_parity(s0, k, r, q, t, option_price_c, option_price_p)

if option_style.lower() == "eu":
    if parity_check:
        print("Call-put parity verified")
    else:
        print("Call-put parity NOT verified")
else:
    print("AMERICAN OPTION! --> Call-put parity NOT verified")

print("\n-------------------------------------------")