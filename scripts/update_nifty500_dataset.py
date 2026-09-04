from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.downloader import download_stocks
from src.data.storage import load_market_data, save_market_data


UNIVERSE_FILE = Path("data/raw/nifty500_universe.csv")
MARKET_DATA_FILE = Path("data/market/nifty500.parquet")

REQUIRED_COLUMNS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE_FILE}"
        )

    universe = pd.read_csv(UNIVERSE_FILE)

    required = {"sector", "symbol", "yf_ticker"}
    missing = required - set(universe.columns)

    if missing:
        raise ValueError(
            f"Universe missing columns: {sorted(missing)}"
        )

    universe["yf_ticker"] = (
        universe["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    universe = universe[
        universe["yf_ticker"].notna()
        & (universe["yf_ticker"] != "")
        & (universe["yf_ticker"].str.lower() != "nan")
    ]

    return universe.drop_duplicates(
        subset=["yf_ticker"]
    ).reset_index(drop=True)


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    data = data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    ).dt.tz_localize(None)

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    )

    data = data[
        (data["Open"] > 0)
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["Volume"] >= 0)
    ]

    data = data.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    )

    return data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


def main() -> None:
    start_time = time.perf_counter()

    print("=" * 70)
    print("NIFTY 500 INCREMENTAL DATA UPDATE")
    print("=" * 70)

    if not MARKET_DATA_FILE.exists():
        raise FileNotFoundError(
            "Master Parquet file does not exist. "
            "Run build_nifty500_dataset.py first."
        )

    existing = load_market_data()

    if existing.empty:
        raise ValueError(
            "Master Parquet exists but contains no data."
        )

    universe = load_universe()

    tickers = universe["yf_ticker"].tolist()

    existing_tickers = set(
        existing["Ticker"].unique()
    )

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in existing_tickers
    ]

    print(
        f"Universe tickers : {len(tickers):,}"
    )

    print(
        f"Stored tickers   : {len(existing_tickers):,}"
    )

    print(
        f"Missing tickers  : {len(missing_tickers):,}"
    )

    print(
        f"Stored rows      : {len(existing):,}"
    )

    print()

    # ---------------------------------------------------------
    # Determine the latest stored date for every ticker
    # ---------------------------------------------------------

    latest_dates = (
        existing
        .groupby("Ticker")["Date"]
        .max()
    )

    print("Checking latest stored date for each ticker...")

    # ---------------------------------------------------------
    # Download updates ticker-by-ticker.
    #
    # This deliberately avoids assuming that every ticker has
    # the same latest available date.
    # ---------------------------------------------------------

    updated_frames = []

    today = pd.Timestamp.today().normalize()

    for index, ticker in enumerate(tickers, start=1):

        latest_date = latest_dates.get(ticker)

        if pd.isna(latest_date):
            download_start = "2000-01-01"
        else:
            download_start = (
                latest_date + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")

        if pd.Timestamp(download_start) > today:
            continue

        print(
            f"[{index:3}/{len(tickers)}] "
            f"{ticker:<20} "
            f"from {download_start}"
        )

        try:
            new_data = download_stocks(
                [ticker],
                start=download_start,
            )

            new_data = validate_data(new_data)

            if not new_data.empty:
                updated_frames.append(new_data)

        except Exception as exc:
            print(
                f"    UPDATE FAILED: {exc}"
            )

    # ---------------------------------------------------------
    # Add newly discovered tickers
    # ---------------------------------------------------------

    if missing_tickers:
        print()
        print(
            f"Downloading {len(missing_tickers)} "
            f"missing ticker(s)..."
        )

        try:
            new_ticker_data = download_stocks(
                missing_tickers,
                start="2000-01-01",
            )

            new_ticker_data = validate_data(
                new_ticker_data
            )

            if not new_ticker_data.empty:
                updated_frames.append(
                    new_ticker_data
                )

        except Exception as exc:
            print(
                f"Missing ticker download failed: {exc}"
            )

    # ---------------------------------------------------------
    # Nothing new
    # ---------------------------------------------------------

    if not updated_frames:
        elapsed = time.perf_counter() - start_time

        print()
        print("=" * 70)
        print("NO NEW MARKET DATA")
        print("=" * 70)
        print(
            f"Existing rows : {len(existing):,}"
        )
        print(
            f"Elapsed time  : {elapsed:.2f} seconds"
        )
        print("=" * 70)

        return

    # ---------------------------------------------------------
    # Merge existing + new data
    # ---------------------------------------------------------

    new_data = pd.concat(
        updated_frames,
        ignore_index=True,
    )

    new_data = validate_data(new_data)

    combined = pd.concat(
        [existing, new_data],
        ignore_index=True,
    )

    combined = validate_data(combined)

    # ---------------------------------------------------------
    # Save updated master dataset
    # ---------------------------------------------------------

    save_market_data(
        combined,
        path=MARKET_DATA_FILE,
    )

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("NIFTY 500 UPDATE COMPLETE")
    print("=" * 70)

    print(
        f"Previous rows   : {len(existing):,}"
    )

    print(
        f"New rows         : {len(new_data):,}"
    )

    print(
        f"Final rows       : {len(combined):,}"
    )

    print(
        f"Tickers          : "
        f"{combined['Ticker'].nunique():,}"
    )

    print(
        f"Latest date      : "
        f"{combined['Date'].max().date()}"
    )

    print(
        f"Elapsed time     : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"File             : "
        f"{MARKET_DATA_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()