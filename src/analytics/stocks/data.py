from __future__ import annotations

import pandas as pd


def _prepare_stock_data(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    required_columns = {
        "Ticker",
        "Date",
        "Close",
    }

    missing_columns = (
        required_columns - set(market_data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Market data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = market_data.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    data = data[
        data["Ticker"].astype(str).str.strip()
        == str(ticker).strip()
    ].copy()

    if start_date is not None:
        data = data[
            data["Date"] >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        data = data[
            data["Date"] <= pd.Timestamp(end_date)
        ]

    if data.empty:
        raise ValueError(
            f"No market data available for {ticker}."
        )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    if "Volume" in data.columns:
        data["Volume"] = pd.to_numeric(
            data["Volume"],
            errors="coerce",
        )

    data = (
        data
        .dropna(subset=["Close"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            f"No valid closing prices available for {ticker}."
        )

    return data


def _stock_price_series(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    data = _prepare_stock_data(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    prices = data.set_index("Date")["Close"]
    prices.name = ticker

    return prices


def _stock_return_series(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    prices = _stock_price_series(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    returns = prices.pct_change().dropna()
    returns.name = ticker

    return returns