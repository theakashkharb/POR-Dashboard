"""
POR-Dashboard
Data Cleaner
============

Responsible only for validating and cleaning downloaded market data.

This module:
    - Validates required columns
    - Converts dates
    - Converts numeric market data
    - Removes invalid rows
    - Checks missing data
    - Sorts data chronologically
    - Forward-fills remaining missing values
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_columns(
    data: pd.DataFrame,
) -> None:
    """
    Validate that all required market-data columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def clean_stock(
    data: pd.DataFrame,
    max_missing_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Clean historical stock data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw downloaded stock data.

    max_missing_pct : float, default=0.05
        Maximum allowed missing percentage in any
        required column.

    Returns
    -------
    pd.DataFrame
        Cleaned stock data.

    Returns an empty DataFrame when the input is empty
    or when missing data remains excessive.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame"
        )

    if data.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_columns(data)

    data = data.copy()

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Remove rows with missing critical values
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Remove invalid market values
    # --------------------------------------------------------

    data = data[
        (data["Open"] > 0)
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["Volume"] >= 0)
    ]

    # --------------------------------------------------------
    # Validate OHLC relationships
    # --------------------------------------------------------

    data = data[
        (data["High"] >= data["Open"])
        & (data["High"] >= data["Close"])
        & (data["High"] >= data["Low"])
        & (data["Low"] <= data["Open"])
        & (data["Low"] <= data["Close"])
    ]

    # --------------------------------------------------------
    # Nothing left after cleaning
    # --------------------------------------------------------

    if data.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Sort by ticker and date
    # --------------------------------------------------------

    data = (
        data
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check remaining missing data
    # --------------------------------------------------------

    missing_pct = (
        data[REQUIRED_COLUMNS]
        .isna()
        .mean()
    )

    if missing_pct.max() > max_missing_pct:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Forward-fill within each ticker
    #
    # Important:
    # Keep Ticker as a normal column.
    # --------------------------------------------------------

    data[numeric_columns] = (
        data
        .groupby("Ticker")[numeric_columns]
        .ffill()
    )

    # --------------------------------------------------------
    # Remove anything still missing
    # --------------------------------------------------------

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    return (
        data
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )