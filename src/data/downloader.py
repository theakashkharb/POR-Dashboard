"""
POR-Dashboard
Data Downloader
==============

Responsible only for downloading raw market data.

This module:
    - Downloads OHLCV data
    - Normalizes the output structure
    - Adds the Ticker column

It does NOT:
    - Clean missing data
    - Screen assets
    - Select assets
    - Calculate returns
    - Calculate risk
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_stock(
    ticker: str,
    period: str = "max",
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single stock.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    period : str, default="max"
        Yahoo Finance download period.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:

        Ticker
        Date
        Open
        High
        Low
        Close
        Volume

    Notes
    -----
    An empty DataFrame is returned when no data is available.
    """

    if not isinstance(ticker, str):
        raise TypeError("ticker must be a string")

    ticker = ticker.strip()

    if not ticker:
        raise ValueError("ticker cannot be empty")

    try:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download data for {ticker}: {exc}"
        ) from exc

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    # --------------------------------------------------------
    # Handle Yahoo Finance MultiIndex columns
    # --------------------------------------------------------

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # --------------------------------------------------------
    # Required OHLCV columns
    # --------------------------------------------------------

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Downloaded data for {ticker} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Reset date index
    # --------------------------------------------------------

    data = data.reset_index()

    # Yahoo may call this column Date
    # after reset_index(). Validate it explicitly.
    if "Date" not in data.columns:
        raise ValueError(
            f"Downloaded data for {ticker} does not contain Date column"
        )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    data = data[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].copy()

    # --------------------------------------------------------
    # Add ticker identifier
    # --------------------------------------------------------

    data["Ticker"] = ticker

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    data = data[
        [
            "Ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    return data