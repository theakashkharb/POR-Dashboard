import pandas as pd
import streamlit as st

from src.data.downloader import download_stock
from src.data.cleaner import clean_stock

@st.cache_data
def get_data(
    tickers,
    start=None,
    end=None,
    period="max"
):
    all_data = []

    for ticker in tickers:

        data = download_stock(
            ticker,
            period=period
        )

        if data.empty:
            continue

        data = clean_stock(data)

        if data.empty:
            continue

        if start is not None:
            data = data[
                data["Date"] >= pd.to_datetime(start)
            ]

        if end is not None:
            data = data[
                data["Date"] <= pd.to_datetime(end)
            ]

        all_data.append(data)

    if not all_data:
        raise ValueError("No valid data available.")

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined