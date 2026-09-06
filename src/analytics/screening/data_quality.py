from __future__ import annotations

import pandas as pd


def calculate_data_quality(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate basic market-data availability for the universe.

    Returns one row per universe stock with:
    - Data Available
    - First Date
    - Last Date
    - Trading Days
    """

    required_market_columns = {
        "Ticker",
        "Date",
        "Close",
        "Volume",
    }

    missing_market_columns = (
        required_market_columns
        - set(market_data.columns)
    )

    if missing_market_columns:
        raise ValueError(
            "Market data is missing columns: "
            f"{sorted(missing_market_columns)}"
        )

    required_universe_columns = {
        "symbol",
        "yf_ticker",
        "sector",
    }

    missing_universe_columns = (
        required_universe_columns
        - set(universe.columns)
    )

    if missing_universe_columns:
        raise ValueError(
            "Universe is missing columns: "
            f"{sorted(missing_universe_columns)}"
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

    valid_data = data[
        data["Close"].notna()
        & (data["Close"] > 0)
    ].copy()

    summary = (
        valid_data
        .groupby("Ticker")
        .agg(
            First_Date=("Date", "min"),
            Last_Date=("Date", "max"),
            Trading_Days=("Date", "nunique"),
        )
        .reset_index()
    )

    result = universe[
        [
            "symbol",
            "yf_ticker",
            "sector",
        ]
    ].copy()

    result["yf_ticker"] = (
        result["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    result = result.merge(
        summary,
        left_on="yf_ticker",
        right_on="Ticker",
        how="left",
    )

    result["Data Available"] = (
        result["Trading_Days"]
        .fillna(0)
        .gt(0)
    )

    result["Trading_Days"] = (
        result["Trading_Days"]
        .fillna(0)
        .astype(int)
    )

    result = result.drop(
        columns=["Ticker"]
    )

    return result[
        [
            "symbol",
            "yf_ticker",
            "sector",
            "Data Available",
            "First_Date",
            "Last_Date",
            "Trading_Days",
        ]
    ].sort_values(
        "symbol"
    ).reset_index(
        drop=True
    )