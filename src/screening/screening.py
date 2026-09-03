import pandas as pd
import numpy as np


def check_data_availability(data, min_observations=252):
    observations = (
        data.groupby("Ticker")["Close"]
        .count()
        .reset_index(name="observations")
    )

    return observations[
        observations["observations"] >= min_observations
    ].copy()


def check_missing_data(data, max_missing_pct=0.05):
    missing = (
        data.groupby("Ticker")["Close"]
        .apply(lambda x: x.isna().mean())
        .reset_index(name="missing_pct")
    )

    return missing[
        missing["missing_pct"] <= max_missing_pct
    ].copy()


def calculate_adtv(data, window=20):
    data = data.copy()

    data["Traded_Value"] = data["Close"] * data["Volume"]

    adtv = (
        data.groupby("Ticker")["Traded_Value"]
        .rolling(window)
        .mean()
        .groupby(level=0)
        .last()
        .reset_index(name="ADTV")
    )

    return adtv


def apply_adtv_filter(data, min_adtv=1e7):
    adtv = calculate_adtv(data)

    return adtv[
        adtv["ADTV"] >= min_adtv
    ].copy()


def apply_price_floor(data, min_price=20):
    latest_price = (
        data.sort_values("Date")
        .groupby("Ticker")
        .tail(1)[["Ticker", "Close"]]
        .rename(columns={"Close": "latest_price"})
    )

    return latest_price[
        latest_price["latest_price"] >= min_price
    ].copy()


def calculate_volatility(data, window=252):
    returns = (
        data.sort_values("Date")
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    data = data.copy()
    data["Return"] = returns

    volatility = (
        data.groupby("Ticker")["Return"]
        .std()
        .mul(np.sqrt(252))
        .reset_index(name="annualized_volatility")
    )

    return volatility


def apply_volatility_filter(
    data,
    min_vol=0.05,
    max_vol=0.80
):
    volatility = calculate_volatility(data)

    return volatility[
        (volatility["annualized_volatility"] >= min_vol)
        & (volatility["annualized_volatility"] <= max_vol)
    ].copy()


def select_top_stocks(data, max_stocks=25):
    stocks = (
        data.sort_values("Ticker")
        ["Ticker"]
        .drop_duplicates()
        .head(max_stocks)
        .tolist()
    )

    return stocks


def run_screening(
    data,
    min_observations=252,
    max_missing_pct=0.05,
    min_adtv=1e7,
    min_price=20,
    min_vol=0.05,
    max_vol=0.80,
    max_stocks=25
):
    eligible = set(data["Ticker"].unique())

    availability = check_data_availability(
        data,
        min_observations
    )
    eligible &= set(availability["Ticker"])

    missing = check_missing_data(
        data,
        max_missing_pct
    )
    eligible &= set(missing["Ticker"])

    adtv = apply_adtv_filter(
        data,
        min_adtv
    )
    eligible &= set(adtv["Ticker"])

    price = apply_price_floor(
        data,
        min_price
    )
    eligible &= set(price["Ticker"])

    volatility = apply_volatility_filter(
        data,
        min_vol,
        max_vol
    )
    eligible &= set(volatility["Ticker"])

    screened_data = data[
        data["Ticker"].isin(eligible)
    ].copy()

    selected_stocks = select_top_stocks(
        screened_data,
        max_stocks
    )

    return selected_stocks