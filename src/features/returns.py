from __future__ import annotations

import numpy as np
import pandas as pd


def create_price_matrix(
    market_data: pd.DataFrame,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Convert long-format market data into a price matrix.

    Rows    = dates
    Columns = tickers
    Values  = selected price
    """

    required_columns = {
        "Ticker",
        "Date",
        price_column,
    }

    missing_columns = (
        required_columns - set(market_data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Market data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prices = market_data.pivot(
        index="Date",
        columns="Ticker",
        values=price_column,
    )

    return prices.sort_index()


def calculate_simple_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate simple percentage returns.
    """

    if prices.empty:
        raise ValueError(
            "Price matrix is empty."
        )

    return prices.pct_change(
        fill_method=None
    )


def calculate_log_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate continuously compounded log returns.
    """

    if prices.empty:
        raise ValueError(
            "Price matrix is empty."
        )

    if (prices <= 0).any().any():
        raise ValueError(
            "Prices must be positive "
            "for log returns."
        )

    return np.log(
        prices / prices.shift(1)
    )