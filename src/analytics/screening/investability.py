from __future__ import annotations

import pandas as pd


def calculate_investability(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    minimum_trading_days: int = 200,
) -> pd.DataFrame:
    """
    Calculate basic investability characteristics.

    Metrics:
    - Trading Days
    - Positive Close Days
    - Zero Volume Days
    - Investable

    A stock is considered investable when:
    - it has at least minimum_trading_days observations
    - closing prices are valid
    - it has at least one day with positive trading volume
    """

    required_market_columns = {
        "Ticker",
        "Date",
        "Close",
        "Volume",
    }

    missing_market_columns = (
        required_market_columns
        - set(market_data.columns)
    )

    if missing_market_columns:
        raise ValueError(
            "Market data is missing columns: "
            f"{sorted(missing_market_columns)}"
        )

    required_universe_columns = {
        "symbol",
        "yf_ticker",
        "sector",
    }

    missing_universe_columns = (
        required_universe_columns
        - set(universe.columns)
    )

    if missing_universe_columns:
        raise ValueError(
            "Universe is missing columns: "
            f"{sorted(missing_universe_columns)}"
        )

    data = market_data.copy()

    data["Ticker"] = (
        data["Ticker"]
        .astype(str)
        .str.strip()
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce",
    )

    data = (
        data
        .drop_duplicates(
            subset=["Ticker", "Date"],
            keep="last",
        )
        .sort_values(
            ["Ticker", "Date"]
        )
    )

    if data.empty:
        raise ValueError(
            "No market data available."
        )

    summary = (
        data
        .groupby("Ticker")
        .agg(
            Trading_Days=(
                "Date",
                "nunique",
            ),
            Positive_Close_Days=(
                "Close",
                lambda x: int(
                    (
                        x.notna()
                        & (x > 0)
                    ).sum()
                ),
            ),
            Zero_Volume_Days=(
                "Volume",
                lambda x: int(
                    (
                        x.fillna(0)
                        <= 0
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    result = universe[
        [
            "symbol",
            "yf_ticker",
            "sector",
        ]
    ].copy()

    result["yf_ticker"] = (
        result["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    result = result.merge(
        summary,
        left_on="yf_ticker",
        right_on="Ticker",
        how="left",
    )

    result["Trading_Days"] = (
        result["Trading_Days"]
        .fillna(0)
        .astype(int)
    )

    result["Positive_Close_Days"] = (
        result["Positive_Close_Days"]
        .fillna(0)
        .astype(int)
    )

    result["Zero_Volume_Days"] = (
        result["Zero_Volume_Days"]
        .fillna(0)
        .astype(int)
    )

    result["Investable"] = (
        result["Trading_Days"]
        >= minimum_trading_days
    ) & (
        result["Positive_Close_Days"]
        > 0
    )

    result = result.drop(
        columns=["Ticker"]
    )

    return (
        result[
            [
                "symbol",
                "yf_ticker",
                "sector",
                "Trading_Days",
                "Positive_Close_Days",
                "Zero_Volume_Days",
                "Investable",
            ]
        ]
        .sort_values("symbol")
        .reset_index(drop=True)
    )