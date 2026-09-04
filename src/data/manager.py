"""
POR-Dashboard
Data Manager
============

Coordinates the market-data workflow:

    Download → Clean → Filter → Combine

This module should NOT contain:
    - Feature engineering
    - Stock selection logic
    - Portfolio calculations
    - Risk calculations
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.downloader import download_stock
from src.data.cleaner import clean_stock


@st.cache_data
def get_data(
    tickers: list[str],
    start=None,
    end=None,
    period: str = "max",
) -> pd.DataFrame:
    """
    Download, clean, filter, and combine stock data.

    Parameters
    ----------
    tickers : list[str]
        Stock ticker symbols.

    start : optional
        Start date for filtering.

    end : optional
        End date for filtering.

    period : str, default="max"
        Period passed to the downloader.

    Returns
    -------
    pd.DataFrame
        Combined cleaned market data.

    Raises
    ------
    ValueError
        If no valid data is available.
    """

    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    all_data: list[pd.DataFrame] = []

    for ticker in tickers:

        # 1. Download
        data = download_stock(
            ticker,
            period=period,
        )

        if data.empty:
            continue

        # 2. Clean
        data = clean_stock(data)

        if data.empty:
            continue

        # 3. Date filtering
        if start is not None:
            data = data[
                data["Date"] >= pd.to_datetime(start)
            ]

        if end is not None:
            data = data[
                data["Date"] <= pd.to_datetime(end)
            ]

        if data.empty:
            continue

        all_data.append(data)

    # 4. Combine all valid tickers
    if not all_data:
        raise ValueError(
            "No valid data available."
        )

    return pd.concat(
        all_data,
        ignore_index=True,
    )
