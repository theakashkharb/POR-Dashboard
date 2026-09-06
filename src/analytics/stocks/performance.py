from __future__ import annotations

import pandas as pd


TRADING_DAYS = 252


def _prepare_stock_prices(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
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

    data["Close"] = pd.to_numeric(
        data["Close"],
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
    )

    if data.empty:
        raise ValueError(
            f"No valid price data available for {ticker}."
        )

    prices = data.set_index("Date")["Close"]

    if len(prices) < 2:
        raise ValueError(
            "At least two prices are required."
        )

    return prices


def calculate_period_return(
    prices: pd.Series,
    months: int,
) -> float:
    if prices.empty:
        return float("nan")

    end_date = prices.index[-1]

    target_date = (
        end_date
        - pd.DateOffset(months=months)
    )

    eligible_prices = prices[
        prices.index <= target_date
    ]

    if eligible_prices.empty:
        return float("nan")

    start_price = eligible_prices.iloc[-1]
    end_price = prices.iloc[-1]

    if start_price <= 0:
        return float("nan")

    return float(
        end_price / start_price - 1.0
    )


def calculate_cagr(
    prices: pd.Series,
) -> float:
    if len(prices) < 2:
        return float("nan")

    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]

    if start_price <= 0 or end_price <= 0:
        return float("nan")

    elapsed_days = (
        prices.index[-1] - prices.index[0]
    ).days

    years = elapsed_days / 365.25

    if years <= 0:
        return float("nan")

    return float(
        (end_price / start_price)
        ** (1.0 / years)
        - 1.0
    )


def calculate_stock_performance(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    prices = _prepare_stock_prices(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "1M Return":
            calculate_period_return(
                prices,
                1,
            ),
        "3M Return":
            calculate_period_return(
                prices,
                3,
            ),
        "6M Return":
            calculate_period_return(
                prices,
                6,
            ),
        "1Y Return":
            calculate_period_return(
                prices,
                12,
            ),
        "3Y Return":
            calculate_period_return(
                prices,
                36,
            ),
        "CAGR":
            calculate_cagr(prices),
    }


def calculate_rolling_1y_return(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    prices = _prepare_stock_prices(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    rolling_return = (
        prices
        / prices.shift(TRADING_DAYS)
        - 1.0
    )

    rolling_return.name = (
        "Rolling 1Y Return"
    )

    return rolling_return.dropna()


def calculate_win_rate(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    prices = _prepare_stock_prices(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    daily_returns = (
        prices
        .pct_change()
        .dropna()
    )

    daily_win_rate = float(
        (daily_returns > 0).mean()
    )

    monthly_returns = (
        (
            1.0 + daily_returns
        )
        .resample("ME")
        .prod()
        - 1.0
    ).dropna()

    if monthly_returns.empty:
        monthly_win_rate = float("nan")
    else:
        monthly_win_rate = float(
            (monthly_returns > 0).mean()
        )

    return {
        "Daily Win Rate":
            daily_win_rate,
        "Monthly Win Rate":
            monthly_win_rate,
    }