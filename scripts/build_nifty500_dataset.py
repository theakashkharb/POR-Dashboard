from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.downloader import download_stocks
from src.data.storage import save_market_data


UNIVERSE_FILE = Path("data/raw/nifty500_universe.csv")
OUTPUT_FILE = Path("data/market/nifty500.parquet")

REQUIRED_UNIVERSE_COLUMNS = {
    "sector",
    "symbol",
    "yf_ticker",
}

REQUIRED_DATA_COLUMNS = {
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    universe = pd.read_csv(path)

    missing = REQUIRED_UNIVERSE_COLUMNS - set(universe.columns)
    if missing:
        raise ValueError(
            f"Universe file is missing required columns: {sorted(missing)}"
        )

    universe = universe.copy()

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

    universe = universe.drop_duplicates(
        subset=["yf_ticker"]
    ).reset_index(drop=True)

    return universe


def validate_market_data(
    data: pd.DataFrame,
    requested_tickers: list[str],
) -> tuple[pd.DataFrame, list[str]]:

    if data.empty:
        raise ValueError("Downloaded market data is empty.")

    missing = REQUIRED_DATA_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            f"Downloaded data is missing required columns: {sorted(missing)}"
        )

    data = data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    ).dt.tz_localize(None)

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    before = len(data)

    data = data.dropna(
        subset=[
            "Ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
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

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    downloaded_tickers = set(data["Ticker"].unique())
    requested_set = set(requested_tickers)

    failed_tickers = sorted(
        requested_set - downloaded_tickers
    )

    removed_rows = before - len(data)

    print(f"Rows removed during validation: {removed_rows:,}")

    return data, failed_tickers


def print_report(
    universe: pd.DataFrame,
    data: pd.DataFrame,
    failed_tickers: list[str],
    elapsed: float,
) -> None:

    requested = len(universe)
    successful = data["Ticker"].nunique()
    failed = len(failed_tickers)

    file_size_mb = 0.0

    if OUTPUT_FILE.exists():
        file_size_mb = (
            OUTPUT_FILE.stat().st_size
            / (1024 * 1024)
        )

    print()
    print("=" * 70)
    print("NIFTY 500 DATASET BUILD COMPLETE")
    print("=" * 70)

    print(f"Universe file       : {UNIVERSE_FILE}")
    print(f"Output file         : {OUTPUT_FILE}")
    print()

    print(f"Requested tickers   : {requested:,}")
    print(f"Successful tickers  : {successful:,}")
    print(f"Failed tickers      : {failed:,}")
    print(f"Total rows          : {len(data):,}")
    print()

    if not data.empty:
        print(
            f"Minimum date        : "
            f"{data['Date'].min().date()}"
        )

        print(
            f"Maximum date        : "
            f"{data['Date'].max().date()}"
        )

    print(
        f"Parquet size        : "
        f"{file_size_mb:.2f} MB"
    )

    print(
        f"Elapsed time        : "
        f"{elapsed:.2f} seconds"
    )

    print("=" * 70)

    if failed_tickers:
        print()
        print("FAILED TICKERS")
        print("-" * 70)

        for ticker in failed_tickers:
            print(ticker)

        print()
        print(
            "Failed tickers can be investigated and retried later."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build the local NIFTY 500 historical market "
            "data Parquet dataset."
        )
    )

    parser.add_argument(
        "--start",
        default="2000-01-01",
        help="Historical start date. Default: 2000-01-01",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Historical end date. Default: latest available.",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    print("=" * 70)
    print("BUILDING NIFTY 500 MARKET DATASET")
    print("=" * 70)

    print(f"Start date: {args.start}")

    print(
        f"End date  : "
        f"{args.end if args.end else 'latest available'}"
    )

    print()

    universe = load_universe(UNIVERSE_FILE)

    tickers = universe["yf_ticker"].tolist()

    print(
        f"Universe loaded: {len(tickers):,} unique tickers"
    )

    print()
    print("Starting batch download...")
    print()

    data = download_stocks(
        tickers,
        start=args.start,
        end=args.end,
    )

    if data.empty:
        raise RuntimeError(
            "No data was downloaded. Dataset was not created."
        )

    data, failed_tickers = validate_market_data(
        data,
        tickers,
    )

    if data.empty:
        raise RuntimeError(
            "All downloaded data failed validation."
        )

    save_market_data(
        data,
        path=OUTPUT_FILE,
    )

    elapsed = time.perf_counter() - start_time

    print_report(
        universe,
        data,
        failed_tickers,
        elapsed,
    )


if __name__ == "__main__":
    main()
