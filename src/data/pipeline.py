import os
import pandas as pd

from src.data.downloader import download_stock
from src.data.cleaner import clean_stock
from src.data.storage import save_to_parquet


def download_universe(
    universe_file="data/raw/nifty500_universe.csv",
    period="20y"
):
    universe = pd.read_csv(universe_file)

    all_data = []
    failed = []

    for i, ticker in enumerate(universe["yf_ticker"], start=1):

        print(f"[{i}/{len(universe)}] Downloading {ticker}")

        data = download_stock(
            ticker,
            period=period
        )

        if data.empty:
            failed.append(ticker)
            continue

        data = clean_stock(data)

        if data.empty:
            failed.append(ticker)
            continue

        all_data.append(data)

    if not all_data:
        raise ValueError("No stock data downloaded.")

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    save_to_parquet(combined)

    # Save failed tickers
    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame({
        "yf_ticker": failed
    }).to_csv(
        "data/processed/failed_tickers.csv",
        index=False
    )

    return combined, failed