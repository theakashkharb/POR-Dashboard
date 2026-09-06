from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NIFTY500_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "nifty500.parquet"
)

NIFTY50_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "nifty50.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytics"
    / "stock_daily_returns.parquet"
)


def prepare_stock_returns(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily returns for every NIFTY 500 stock."""

    data = market_data[
        ["Ticker", "Date", "Close"]
    ].copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data = (
        data.dropna(
            subset=["Ticker", "Date", "Close"]
        )
        .drop_duplicates(
            subset=["Ticker", "Date"],
            keep="last",
        )
        .sort_values(
            ["Ticker", "Date"]
        )
    )

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    return data[
        ["Ticker", "Date", "Return"]
    ].reset_index(drop=True)


def prepare_nifty50_returns(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily NIFTY 50 returns."""

    data = market_data[
        ["Ticker", "Date", "Close"]
    ].copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data = (
        data.dropna(
            subset=["Date", "Close"]
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
    )

    data["Return"] = data["Close"].pct_change()

    return data[
        ["Ticker", "Date", "Return"]
    ].reset_index(drop=True)


def main() -> None:
    print("Loading NIFTY 500 data...")
    nifty500 = pd.read_parquet(NIFTY500_FILE)

    print("Calculating NIFTY 500 daily returns...")
    stock_returns = prepare_stock_returns(nifty500)

    print("Loading NIFTY 50 data...")
    nifty50 = pd.read_parquet(NIFTY50_FILE)

    print("Calculating NIFTY 50 daily returns...")
    nifty50_returns = prepare_nifty50_returns(nifty50)

    result = pd.concat(
        [
            stock_returns,
            nifty50_returns,
        ],
        ignore_index=True,
    )

    result = (
        result
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Precalculation complete.")
    print(f"Rows: {len(result):,}")
    print(f"Tickers: {result['Ticker'].nunique():,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()