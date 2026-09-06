from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.market import (
    calculate_annualized_volatility,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
)
from src.analytics.stocks import calculate_stock_period_returns
from src.features.returns import calculate_simple_returns, create_price_matrix


TOP_STOCK_COUNT = 10


def build_top_stocks(
    market_data: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    stock_returns = calculate_stock_period_returns(
        market_data=market_data,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    if stock_returns.empty:
        raise ValueError(
            "No stock return data available for the selected period."
        )

    top_stocks = (
        stock_returns
        .sort_values(
            "Returns",
            ascending=False,
        )
        .head(TOP_STOCK_COUNT)
        .copy()
    )

    selected_tickers = top_stocks["Ticker"].tolist()

    selected_market_data = market_data[
        market_data["Ticker"].isin(selected_tickers)
        & (
            market_data["Date"]
            >= pd.Timestamp(start_date)
        )
        & (
            market_data["Date"]
            <= pd.Timestamp(end_date)
        )
    ].copy()

    if selected_market_data.empty:
        raise ValueError(
            "No price data available for the selected stocks."
        )

    prices = create_price_matrix(
        selected_market_data
    )

    daily_returns = calculate_simple_returns(
        prices
    )

    metrics = []

    for _, stock in top_stocks.iterrows():
        ticker = stock["Ticker"]

        if ticker not in daily_returns.columns:
            continue

        stock_daily_returns = daily_returns[
            ticker
        ].dropna()

        if stock_daily_returns.empty:
            continue

        metrics.append(
            {
                "Ticker": ticker,
                "Returns": float(stock["Returns"]),
                "Volatility": calculate_annualized_volatility(
                    stock_daily_returns
                ),
                "Sharpe": calculate_sharpe_ratio(
                    stock_daily_returns
                ),
                "Max Drawdown": calculate_maximum_drawdown(
                    stock_daily_returns
                ),
                "Positive Days": float(
                    (stock_daily_returns > 0).mean()
                ),
            }
        )

    if not metrics:
        raise ValueError(
            "No valid risk metrics could be calculated "
            "for the selected stocks."
        )

    result = pd.DataFrame(metrics)

    result = (
        result
        .sort_values(
            "Returns",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "Rank",
        range(1, len(result) + 1),
    )

    return result


def render_top_stocks(
    market_data: pd.DataFrame,
    start_date,
    end_date,
) -> None:
    st.subheader("Top Performing Stocks")

    top_stocks = build_top_stocks(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )

    if top_stocks.empty:
        st.info(
            "No Top Performing Stocks data available "
            "for the selected period."
        )
        return

    display_data = top_stocks.copy()

    display_data["Returns"] = display_data[
        "Returns"
    ].map(
        lambda value: f"{value:.2%}"
    )

    display_data["Volatility"] = display_data[
        "Volatility"
    ].map(
        lambda value: f"{value:.2%}"
    )

    display_data["Sharpe"] = display_data[
        "Sharpe"
    ].map(
        lambda value: f"{value:.2f}"
    )

    display_data["Max Drawdown"] = display_data[
        "Max Drawdown"
    ].map(
        lambda value: f"{value:.2%}"
    )

    display_data["Positive Days"] = display_data[
        "Positive Days"
    ].map(
        lambda value: f"{value:.2%}"
    )

    display_data = display_data.rename(
        columns={
            "Ticker": "Stock",
        }
    )

    st.dataframe(
        display_data[
            [
                "Rank",
                "Stock",
                "Returns",
                "Volatility",
                "Sharpe",
                "Max Drawdown",
                "Positive Days",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )