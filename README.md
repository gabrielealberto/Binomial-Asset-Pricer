# Binomial-Option-Pricer
This notebook implements the Cox-Ross-Rubinstein (CRR) binomial tree method to value equity options. It supports both European and American exercise styles (Calls &amp; Puts) and includes tools to visualize the backward induction process through an interactive value tree.

# Binomial Option Pricer 📈

A Python implementation of the **Cox-Ross-Rubinstein (CRR) Binomial Model** for pricing equity options. This notebook supports both **European** and **American** style options (Calls and Puts) and includes visualization tools to display the option value propagation through the binomial tree.

## 🚀 Features

* **Valuation Model:** Uses the CRR Binomial Tree method to calculate option premiums.
* **Option Styles:**
    * 🇪🇺 **European:** Exercisable only at maturity.
    * 🇺🇸 **American:** Exercisable at any step (checks for early exercise optimality).
* **Option Types:** Call and Put.
* **Visualization:** Generates a visual tree diagram of option values to trace backward induction.
* **Validation:** Includes **Put-Call Parity** check for European options to verify theoretical consistency.

## 🧮 Mathematical Background

The model discretizes the time to maturity $T$ into $N$ steps of length $\Delta t$.
The asset price moves up by a factor $u$ or down by a factor $d$ at each step:

$$u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u}$$

The risk-neutral probability $p$ is calculated as:

$$p = \frac{e^{r \Delta t} - d}{u - d}$$

For **American options**, at each node $t < T$, the algorithm checks for early exercise:
$$V_{t} = \max(\text{Continuation Value}, \text{Intrinsic Value})$$

## 📦 Requirements

To run this notebook, you need Python installed along with the following libraries:

* `numpy` (for math operations)
* `matplotlib` (for tree visualization)
* `yfinance` (optional, for fetching real market data)

You can install them via pip:

```bash
pip install numpy matplotlib yfinance
💻 UsageOpen Binomial_Asset_Pricer.ipynb in Jupyter Notebook or Google Colab.Configure the input parameters in the setup cell:Python# Example Parameters
s0 = 100       # Current asset price
k = 100        # Strike price
t = 1          # Time to maturity (years)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
n = 10         # Number of binomial steps

opttype = 'c'        # 'c' for Call, 'p' for Put
option_style = 'am'  # 'am' for American, 'eu' for European
Run the cells to calculate the price and visualize the tree.Example OutputPlaintextOption Type         Put
Strike Price        $100.0000
Binomial Price      $5.4321

Put-Call Parity: Verified
🛠 Project Structuretree_engine(): The core logic that builds the Stock Price tree and performs backward induction for the Option Value.
check_parity(): Verifies $C - P = S_0 - K e^{-rT}$ (valid for European options).plot_option_tree(): Visualizes the nodes and values using Matplotlib.
📝 LicenseThis project is open-source and available for educational purposes.
