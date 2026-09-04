"""
POR-Dashboard
Stock Screening
==============

Filters the investment universe based on basic data-quality,
liquidity, price, and volatility requirements.

Screening pipeline:

    Data Availability
        ↓
    Missing Data
        ↓
    ADTV
        ↓
    Price Floor
        ↓
    Volatility
        ↓
    Top Stocks
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def check_data_availability(
    data: pd.DataFrame,
    min_observations: int = 252,
) -> pd.DataFrame:
    """Return tickers with enough historical observations."""

    observations = (
        data.groupby("Ticker")["Close"]
        .count()
        .reset_index(name="observations")
    )

    return observations[
        observations["observations"] >= min_observations
    ].copy()


def check_missing_data(
    data: pd.DataFrame,
    max_missing_pct: float = 0.05,
) -> pd.DataFrame:
    """Return tickers within the allowed missing-data threshold."""

    missing = (
        data.groupby("Ticker")["Close"]
        .apply(lambda x: x.isna().mean())
        .reset_index(name="missing_pct")
    )

    return missing[
        missing["missing_pct"] <= max_missing_pct
    ].copy()


def calculate_adtv(
    data: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate Average Daily Traded Value.

    ADTV = Close × Volume averaged over the window.
    """

    data = data.copy()

    data["Traded_Value"] = (
        data["Close"] * data["Volume"]
    )

    adtv = (
        data.sort_values(["Ticker", "Date"])
        .groupby("Ticker")["Traded_Value"]
        .rolling(window)
        .mean()
        .groupby(level=0)
        .last()
        .reset_index(name="ADTV")
    )

    return adtv


def apply_adtv_filter(
    data: pd.DataFrame,
    min_adtv: float = 1e7,
    window: int = 20,
) -> pd.DataFrame:
    """Return tickers satisfying the minimum ADTV requirement."""

    adtv = calculate_adtv(
        data,
        window=window,
    )

    return adtv[
        adtv["ADTV"] >= min_adtv
    ].copy()


def apply_price_floor(
    data: pd.DataFrame,
    min_price: float = 20,
) -> pd.DataFrame:
    """Return tickers whose latest price exceeds the minimum."""

    latest_price = (
        data.sort_values("Date")
        .groupby("Ticker")
        .tail(1)[["Ticker", "Close"]]
        .rename(columns={"Close": "latest_price"})
    )

    return latest_price[
        latest_price["latest_price"] >= min_price
    ].copy()


def calculate_volatility(
    data: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Calculate annualized historical volatility.

    Volatility is calculated from daily percentage returns.
    """

    data = (
        data.sort_values(["Ticker", "Date"])
        .copy()
    )

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    volatility = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .groupby(level=0)
        .last()
        .mul(np.sqrt(TRADING_DAYS))
        .reset_index(name="annualized_volatility")
    )

    return volatility


def apply_volatility_filter(
    data: pd.DataFrame,
    min_vol: float = 0.05,
    max_vol: float = 0.80,
    window: int = 252,
) -> pd.DataFrame:
    """Return tickers within the permitted volatility range."""

    volatility = calculate_volatility(
        data,
        window=window,
    )

    return volatility[
        (volatility["annualized_volatility"] >= min_vol)
        & (volatility["annualized_volatility"] <= max_vol)
    ].copy()


def select_top_stocks(
    data: pd.DataFrame,
    max_stocks: int = 25,
) -> list[str]:
    """Return up to max_stocks eligible tickers."""

    return (
        data.sort_values("Ticker")["Ticker"]
        .drop_duplicates()
        .head(max_stocks)
        .tolist()
    )


def run_screening(
    data: pd.DataFrame,
    min_observations: int = 252,
    max_missing_pct: float = 0.05,
    min_adtv: float = 1e7,
    min_price: float = 20,
    min_vol: float = 0.05,
    max_vol: float = 0.80,
    max_stocks: int = 25,
) -> list[str]:
    """
    Run the complete stock-screening pipeline.

    Returns
    -------
    list[str]
        Final list of eligible stocks.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame"
        )

    if data.empty:
        return []

    required_columns = {
        "Ticker",
        "Date",
        "Close",
        "Volume",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    eligible = set(
        data["Ticker"].dropna().unique()
    )

    # 1. Data availability
    availability = check_data_availability(
        data,
        min_observations,
    )
    eligible &= set(
        availability["Ticker"]
    )

    # 2. Missing data
    missing = check_missing_data(
        data,
        max_missing_pct,
    )
    eligible &= set(
        missing["Ticker"]
    )

    # 3. Liquidity
    adtv = apply_adtv_filter(
        data,
        min_adtv,
    )
    eligible &= set(
        adtv["Ticker"]
    )

    # 4. Price
    price = apply_price_floor(
        data,
        min_price,
    )
    eligible &= set(
        price["Ticker"]
    )

    # 5. Volatility
    volatility = apply_volatility_filter(
        data,
        min_vol,
        max_vol,
    )
    eligible &= set(
        volatility["Ticker"]
    )

    # 6. Final selection
    screened_data = data[
        data["Ticker"].isin(eligible)
    ].copy()

    return select_top_stocks(
        screened_data,
        max_stocks,
    )
