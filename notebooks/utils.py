import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf

DATA_FOLDER = "../data"

def download_or_load(ticker, start, end):
    """
    Downloads the CSV for a ticker only if not already saved.
    Returns the loaded DataFrame.
    """

    file_path = os.path.join(DATA_FOLDER, f"{ticker}.csv")

    # If CSV already exists it will not be downloaded again
    if os.path.exists(file_path):
        print(f"[LOAD DATA] {ticker} already saved. Loading from CSV...")
        return pd.read_csv(file_path, skiprows=2, parse_dates=["Date"], index_col="Date")

    # Otherwise we download and save the CSV
    print(f"[DOWNLOAD DATA] Getting data for {ticker}...")
    df = yf.download(ticker, start=start, end=end)

    df.index.name = "Date"

    # Prefer Adj Close if available, else fall back to Close
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        raise ValueError(f"No usable price column found for {ticker}")

    df.to_csv(file_path)
    print(f"[SAVED DATA] {ticker}.csv created.")

    return df

def clear_all_csv():
    """
    Deletes ALL CSV files inside /data folder.
    Use to force re-download of data.
    """

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            os.remove(os.path.join(DATA_FOLDER, file))
            print(f"[DELETED DATA] {file}")
    print("[DATA CLEARED] All CSV files deleted.")

def max_drawdown(cum_return_series):
    """
    Compute max drawdown from cumulative returns.
    """
    peak = cum_return_series.cummax()
    drawdown = (cum_return_series - peak) / peak
    return drawdown.min()

def performance_metrics(cumreturn_series, var_conf=0.95):
    """
    Compute metrics for one return series.
    We use the folowing metrics to evaluate performance:
    - Sharpe Ratio
    - Max Drawdown
    - Kurtosis
    - Skewness
    - Annualized Volatility
    """
    # Daily returns from cumulative return series
    daily_returns = cumreturn_series.pct_change().dropna()
    ann_factor = np.sqrt(252)

    sharpe = (daily_returns.mean() / daily_returns.std()) * ann_factor
    vol = daily_returns.std() * np.sqrt(252)
    mdd = max_drawdown(cumreturn_series)
    kurt = daily_returns.kurtosis()
    skew = daily_returns.skew()

    # Historical Value at Risk
    var_level = 1 - var_conf
    var = daily_returns.quantile(var_level)

    # Conditional VaR (Expected Shortfall)
    cvar = daily_returns[daily_returns <= var].mean()

    return pd.Series({
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Kurtosis": kurt,
        "Skewness": skew,
        "Annualized Volatility": vol,
        f"VaR {int(var_conf*100)}%": var,
        f"CVaR {int(var_conf*100)}%": cvar
    })

# CPPI simulation function with params
def run_cppi(returns, floor=0.8, m=3, rebalance_freq='W'):
    portfolio_value = 1.0
    cppi_vals = []
    weight_risky = []
    
    rebalance_dates = set(returns.resample(rebalance_freq).first().index)
    w = 0.0  # initial risky weight
    
    for date, row in returns.iterrows():
        if date in rebalance_dates:
            cushion = max(portfolio_value - floor, 0) / portfolio_value
            w = np.clip(m * cushion, 0, 1)
            w = float(w)
        
        spy = float(row["SPY"])
        shy = float(row["SHY"])
        r = w * spy + (1 - w) * shy
        
        portfolio_value *= (1 + r)
        cppi_vals.append(portfolio_value)
        weight_risky.append(w)
        
    strategy_cum = pd.Series(cppi_vals, index=returns.index)
    return strategy_cum, weight_risky

# Colors for floors
floor_colors = {
    0.5: "blue",
    0.55: "cyan",
    0.6: "green",
    0.65: "lime",
    0.7: "yellowgreen",
    0.75: "gold",
    0.8: "orange",
    0.85: "darkorange",
    0.9: "red",
    0.95: "darkred"
}

# Line styles for multipliers
multiplier_styles = {
    2: "-",
    2.5: "--",
    3: "-.",
    3.5: ":",
    4: (0, (3, 1, 1, 1)),  
    4.5: (0, (5, 5)),      
    5: (0, (1, 1))          
}

# alpha values for rebalance frequencies
rebalance_alpha = {
    'D': 1.0,
    'W': 0.7,
    'ME': 0.4
}

def check_frozen(strategy_weights, threshold=0.5):
    """
    Check if CPPI strategy is frozen (risky allocation mostly 0)
    
    strategy_weights : pd.Series or list of risky weights
    threshold : fraction of time risky weight = 0 to be considered frozen
    """
    w_array = pd.Series(strategy_weights)
    frozen_fraction = (w_array == 0).mean()
    return frozen_fraction >= threshold, frozen_fraction

def run_cppi_dynamic_floor(returns, floor=0.8, m=3, rebalance_freq='ME'):
    """
    CPPI with dynamic floor:
    - Once risky allocation hits 100%, the floor is updated to the current portfolio value.
    - On next rebalance, the weight is recalculated based on new floor.
    - This prevents the strategy from staying at 100% SPY forever.
    
    Parameters:
        returns : pd.DataFrame with columns ["SPY", "SHY"]
        floor : initial floor as fraction of initial portfolio
        m : CPPI multiplier
        rebalance_freq : 'D', 'W', 'ME'
    
    Returns:
        portfolio_cum : pd.Series of portfolio value
        weights : list of risky allocations
    """
    
    portfolio_value = 1.0
    values = []
    weights = []
    
    # Initialize dynamic floor
    dynamic_floor = floor

    # Scheduled rebalance dates
    rebalance_dates = set(returns.resample(rebalance_freq).first().index)

    w = 0.0

    for date, row in returns.iterrows():
        # At rebalance:
        if date in rebalance_dates:
            cushion = max(portfolio_value - dynamic_floor, 0) / portfolio_value
            w = np.clip(m * cushion, 0, 1)

        # If full allocation reached, update floor accordingly
        if w == 1.0:
            dynamic_floor = floor * portfolio_value

        # Calculate portfolio return with current weight allocation
        r = w * row["SPY"] + (1 - w) * row["SHY"]
        portfolio_value *= (1 + r)

        values.append(portfolio_value)
        weights.append(w)

    return pd.Series(values, index=returns.index), weights