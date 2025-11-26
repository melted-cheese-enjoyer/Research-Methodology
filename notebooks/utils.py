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

def max_drawdown(return_series):
    """
    Compute max drawdown from cumulative returns.
    """
    cum_returns = (1 + return_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def performance_metrics(return_series):
    """
    Compute metrics for one return series.
    We use the folowing metrics to evaluate performance:
    - Sharpe Ratio
    - Max Drawdown
    - Kurtosis
    - Skewness
    - Annualized Volatility
    """
    # Annualization factor (daily data)
    ann_factor = np.sqrt(252)

    sharpe = (return_series.mean() / return_series.std()) * ann_factor
    vol = return_series.std() * np.sqrt(252)
    mdd = max_drawdown(return_series)
    kurt = return_series.kurtosis()
    skew = return_series.skew()

    return pd.Series({
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Kurtosis": kurt,
        "Skewness": skew,
        "Annualized Volatility": vol
    })