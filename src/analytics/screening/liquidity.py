from __future__ import annotations

import pandas as pd


def calculate_liquidity(
    market_data: pd.DataFrame,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Calculate liquidity characteristics for each stock.

    Metrics:
    - Average Daily Volume
    - Average Daily Traded Value
    - Median Daily Traded Value
    - Trading Days
    """

    required_columns = {
        "Ticker",
        "Date",
        "Close",
        "Volume",
    }

    missing_columns = (
        required_columns
        - set(market_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Market data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = market_data.copy()

    data["Ticker"] = (
        data["Ticker"]
        .astype(str)
        .str.strip()
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce",
    )

    data = data[
        data["Close"].notna()
        & (data["Close"] > 0)
        & data["Volume"].notna()
        & (data["Volume"] >= 0)
    ].copy()

    if data.empty:
        raise ValueError(
            "No valid price and volume data available."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    data = (
        data
        .drop_duplicates(
            subset=["Ticker", "Date"],
            keep="last",
        )
    )

    # Keep the most recent observation window
    # available for each stock.
    data["Rank"] = (
        data.groupby("Ticker")["Date"]
        .rank(
            method="first",
            ascending=False,
        )
    )

    data = data[
        data["Rank"] <= lookback_days
    ].copy()

    data["Traded Value"] = (
        data["Close"]
        * data["Volume"]
    )

    result = (
        data
        .groupby("Ticker")
        .agg(
            Average_Daily_Volume=(
                "Volume",
                "mean",
            ),
            Average_Daily_Traded_Value=(
                "Traded Value",
                "mean",
            ),
            Median_Daily_Traded_Value=(
                "Traded Value",
                "median",
            ),
            Trading_Days=(
                "Date",
                "nunique",
            ),
        )
        .reset_index()
    )

    result["Average_Daily_Volume"] = (
        result["Average_Daily_Volume"]
        .astype(float)
    )

    result["Average_Daily_Traded_Value"] = (
        result["Average_Daily_Traded_Value"]
        .astype(float)
    )

    result["Median_Daily_Traded_Value"] = (
        result["Median_Daily_Traded_Value"]
        .astype(float)
    )

    result["Trading_Days"] = (
        result["Trading_Days"]
        .astype(int)
    )

    return (
        result
        .sort_values(
            "Average_Daily_Traded_Value",
            ascending=False,
        )
        .reset_index(drop=True)
    )