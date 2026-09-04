import numpy as np
import pandas as pd


# ============================================================
# HISTORICAL VOLATILITY — LATEST VALUE
# ============================================================

def calculate_historical_volatility(
    data,
    window=90,
    annualization=252
):
    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    volatility = (
        data
        .groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .groupby(level=0)
        .last()
        * np.sqrt(annualization)
    )

    return volatility.dropna()


# ============================================================
# HISTORICAL VOLATILITY — ROLLING SERIES
# ============================================================

def calculate_rolling_historical_volatility(
    data,
    window=90,
    annualization=252
):
    """
    Calculate rolling annualized historical volatility
    for every ticker through time.
    """

    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    data["Historical_Volatility"] = (
        data
        .groupby("Ticker")["Return"]
        .transform(
            lambda x: (
                x.rolling(window).std()
                * np.sqrt(annualization)
            )
        )
    )

    return data[
        [
            "Date",
            "Ticker",
            "Historical_Volatility",
        ]
    ].dropna()


# ============================================================
# EWMA VOLATILITY — LATEST VALUE
# ============================================================

def calculate_ewma_volatility(
    data,
    span=90,
    annualization=252
):
    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    ewma_variance = (
        data
        .groupby("Ticker")["Return"]
        .ewm(span=span)
        .var()
        .groupby(level=0)
        .last()
    )

    ewma_volatility = (
        np.sqrt(ewma_variance)
        * np.sqrt(annualization)
    )

    return ewma_volatility.dropna()


# ============================================================
# EWMA VOLATILITY — ROLLING SERIES
# ============================================================

def calculate_rolling_ewma_volatility(
    data,
    span=90,
    annualization=252
):
    """
    Calculate rolling annualized EWMA volatility
    for every ticker through time.
    """

    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    data["EWMA_Volatility"] = (
        data
        .groupby("Ticker")["Return"]
        .transform(
            lambda x: (
                x.ewm(span=span).std()
                * np.sqrt(annualization)
            )
        )
    )

    return data[
        [
            "Date",
            "Ticker",
            "EWMA_Volatility",
        ]
    ].dropna()