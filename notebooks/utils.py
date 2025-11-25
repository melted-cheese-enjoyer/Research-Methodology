import os
import pandas as pd
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
        return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

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

