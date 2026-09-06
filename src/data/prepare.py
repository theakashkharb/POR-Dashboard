from __future__ import annotations

import pandas as pd


def prepare_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare the NIFTY 500 universe.
    """

    required_columns = {
        "sector",
        "symbol",
        "yf_ticker",
    }

    missing_columns = required_columns - set(universe.columns)

    if missing_columns:
        raise ValueError(
            f"Universe is missing columns: {sorted(missing_columns)}"
        )

    result = universe.copy()

    for column in required_columns:
        result[column] = (
            result[column]
            .astype(str)
            .str.strip()
        )

    result = result.drop_duplicates(
        subset=["symbol"]
    )

    return result.reset_index(drop=True)


def prepare_market_data(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and prepare historical market data.
    """

    required_columns = {
        "Ticker",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing_columns = required_columns - set(
        market_data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Market data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    result = market_data.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="raise",
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result["Ticker"] = (
        result["Ticker"]
        .astype(str)
        .str.strip()
    )

    result = result.sort_values(
        ["Ticker", "Date"]
    )

    result = result.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    )

    return result.reset_index(drop=True)