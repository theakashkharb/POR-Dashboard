from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.cleaner import clean_stock
from src.data.downloader import download_stocks
from src.data.storage import load_market_data


@st.cache_data(show_spinner=False)
def get_data(
    tickers,
    start=None,
    end=None,
    period="max",
):

    tickers = tuple(
        dict.fromkeys(
            str(ticker).strip()
            for ticker in tickers
            if ticker
        )
    )

    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    # Read local Parquet first.
    stored = load_market_data()

    if not stored.empty:

        stored = stored[
            stored["Ticker"].isin(tickers)
        ].copy()

        if start is not None:
            stored = stored[
                stored["Date"] >= pd.to_datetime(start)
            ]

        if end is not None:
            stored = stored[
                stored["Date"] <= pd.to_datetime(end)
            ]

        if not stored.empty:
            return stored.sort_values(
                ["Ticker", "Date"]
            ).reset_index(drop=True)

    # Fallback: batch download.
    data = download_stocks(
        list(tickers),
        start=start,
        end=end,
        period=period,
    )

    if data.empty:
        raise ValueError("No valid market data available.")

    # Clean each ticker independently.
    cleaned = []

    for ticker, ticker_data in data.groupby(
        "Ticker",
        sort=False,
    ):

        try:
            result = clean_stock(
                ticker_data.copy()
            )
        except Exception as e:
            print(
                f"CLEANING FAILED: {ticker} | {e}"
            )
            continue

        if not result.empty:
            cleaned.append(result)

    if not cleaned:
        raise ValueError(
            "No valid market data remained after cleaning."
        )

    combined = pd.concat(
        cleaned,
        ignore_index=True,
    )

    if start is not None:
        combined = combined[
            combined["Date"] >= pd.to_datetime(start)
        ]

    if end is not None:
        combined = combined[
            combined["Date"] <= pd.to_datetime(end)
        ]

    return combined.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)