from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


MARKET_DATA_FILE = Path("data/market/nifty500.parquet")
UNIVERSE_FILE = Path("data/raw/nifty500_universe.csv")


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    """
    Load the complete local NIFTY 500 market dataset.

    No internet request is made.
    """

    if not MARKET_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Market data file not found: {MARKET_DATA_FILE}"
        )

    data = pd.read_parquet(
        MARKET_DATA_FILE,
        engine="pyarrow",
    )

    data["Date"] = pd.to_datetime(data["Date"])

    return data


@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    """
    Load NIFTY 500 universe metadata.
    """

    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE_FILE}"
        )

    universe = pd.read_csv(UNIVERSE_FILE)

    universe["yf_ticker"] = (
        universe["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    return universe


@st.cache_data(show_spinner=False)
def get_stock_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return historical data for one stock.
    """

    data = load_market_data()

    result = data[
        data["Ticker"] == ticker
    ].copy()

    if start is not None:
        result = result[
            result["Date"] >= pd.to_datetime(start)
        ]

    if end is not None:
        result = result[
            result["Date"] <= pd.to_datetime(end)
        ]

    return result.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_stocks_data(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return historical data for multiple stocks.
    """

    data = load_market_data()

    result = data[
        data["Ticker"].isin(tickers)
    ].copy()

    if start is not None:
        result = result[
            result["Date"] >= pd.to_datetime(start)
        ]

    if end is not None:
        result = result[
            result["Date"] <= pd.to_datetime(end)
        ]

    return result.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_sector_stocks(
    sector: str,
) -> list[str]:
    """
    Return Yahoo tickers belonging to a sector.
    """

    universe = load_universe()

    result = universe[
        universe["sector"] == sector
    ]

    return result["yf_ticker"].tolist()


@st.cache_data(show_spinner=False)
def get_sector_data(
    sector: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return all historical stock data belonging to a sector.
    """

    tickers = get_sector_stocks(sector)

    if not tickers:
        return pd.DataFrame()

    return get_stocks_data(
        tickers,
        start=start,
        end=end,
    )


@st.cache_data(show_spinner=False)
def get_available_sectors() -> list[str]:
    """
    Return all sectors available in the NIFTY 500 universe.
    """

    universe = load_universe()

    return sorted(
        universe["sector"]
        .dropna()
        .unique()
        .tolist()
    )


@st.cache_data(show_spinner=False)
def get_available_tickers() -> list[str]:
    """
    Return all tickers available in the market dataset.
    """

    data = load_market_data()

    return sorted(
        data["Ticker"]
        .dropna()
        .unique()
        .tolist()
    )


@st.cache_data(show_spinner=False)
def get_universe_data(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return the complete NIFTY 500 market dataset
    within the requested date range.
    """

    data = load_market_data()

    result = data.copy()

    if start is not None:
        result = result[
            result["Date"] >= pd.to_datetime(start)
        ]

    if end is not None:
        result = result[
            result["Date"] <= pd.to_datetime(end)
        ]

    return result.reset_index(drop=True)