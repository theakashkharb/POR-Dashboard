from __future__ import annotations

import pandas as pd


def calculate_stock_period_returns(
    market_data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Calculate stock returns over a selected period.

    A stock receives a period return only when it has
    a valid closing price at both the first and last
    available trading dates of the selected period.
    """

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
        (data["Date"] >= pd.Timestamp(start_date))
        & (data["Date"] <= pd.Timestamp(end_date))
    ].copy()

    if data.empty:
        raise ValueError(
            "No market data available "
            "for selected period."
        )

    data = data.dropna(
        subset=["Close"]
    )

    prices = (
        data
        .pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )
        .sort_index()
    )

    if prices.empty:
        raise ValueError(
            "No valid closing prices available "
            "for selected period."
        )

    start_prices = prices.iloc[0]
    end_prices = prices.iloc[-1]

    valid_stocks = (
        start_prices.notna()
        & end_prices.notna()
        & (start_prices > 0)
        & (end_prices > 0)
    )

    returns = (
        end_prices[valid_stocks]
        / start_prices[valid_stocks]
    ) - 1.0

    result = (
        returns
        .rename("Returns")
        .reset_index()
    )

    return result.dropna(
        subset=["Returns"]
    )


def build_market_map_data(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build Market Map data for all stocks that have
    a valid period return.
    """

    required_columns = {
        "sector",
        "symbol",
        "yf_ticker",
    }

    missing_columns = (
        required_columns - set(universe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Universe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    stock_returns = calculate_stock_period_returns(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )

    mapping = universe[
        [
            "sector",
            "symbol",
            "yf_ticker",
        ]
    ].copy()

    mapping["yf_ticker"] = (
        mapping["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    result = stock_returns.merge(
        mapping,
        left_on="Ticker",
        right_on="yf_ticker",
        how="inner",
    )

    if result.empty:
        raise ValueError(
            "No stocks could be matched between "
            "market data and universe."
        )

    return (
        result[
            [
                "sector",
                "symbol",
                "Ticker",
                "Returns",
            ]
        ]
        .sort_values(
            ["sector", "Returns"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def find_best_stock(
    stock_returns: pd.DataFrame,
) -> pd.Series:
    """
    Return the best-performing stock.
    """

    if stock_returns.empty:
        raise ValueError(
            "Stock return data is empty."
        )

    required_columns = {
        "Ticker",
        "Returns",
    }

    missing_columns = (
        required_columns - set(stock_returns.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Stock return data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return stock_returns.loc[
        stock_returns["Returns"].idxmax()
    ]


def find_worst_stock(
    stock_returns: pd.DataFrame,
) -> pd.Series:
    """
    Return the worst-performing stock.
    """

    if stock_returns.empty:
        raise ValueError(
            "Stock return data is empty."
        )

    required_columns = {
        "Ticker",
        "Returns",
    }

    missing_columns = (
        required_columns - set(stock_returns.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Stock return data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return stock_returns.loc[
        stock_returns["Returns"].idxmin()
    ]


def calculate_market_breadth(
    stock_returns: pd.DataFrame,
) -> dict[str, int]:
    """
    Calculate the number of advancing and declining
    stocks for the selected period.
    """

    if stock_returns.empty:
        raise ValueError(
            "Stock return data is empty."
        )

    if "Returns" not in stock_returns.columns:
        raise ValueError(
            "Stock return data must contain "
            "'Returns'."
        )

    returns = stock_returns["Returns"].dropna()

    return {
        "Advancing Stocks": int(
            (returns > 0).sum()
        ),
        "Declining Stocks": int(
            (returns < 0).sum()
        ),
        "Unchanged Stocks": int(
            (returns == 0).sum()
        ),
        "Stocks Tracked": int(
            returns.count()
        ),
    }