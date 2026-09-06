from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.stocks.market import calculate_stock_period_returns


def build_stock_snapshot(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    selected_ticker: str,
    start_date,
    end_date,
) -> dict:
    required_universe_columns = {
        "sector",
        "symbol",
        "yf_ticker",
    }

    missing_columns = (
        required_universe_columns
        - set(universe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Universe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    stock_data = market_data[
        market_data["Ticker"] == selected_ticker
    ].copy()

    stock_data = (
        stock_data[
            ["Date", "Close", "Volume"]
        ]
        .dropna(subset=["Close"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
    )

    if stock_data.empty:
        raise ValueError(
            "No price data available "
            "for the selected stock."
        )

    selected_period_data = stock_data[
        (stock_data["Date"] >= pd.Timestamp(start_date))
        & (stock_data["Date"] <= pd.Timestamp(end_date))
    ].copy()

    if selected_period_data.empty:
        raise ValueError(
            "No price data available "
            "for the selected period."
        )

    stock_returns = calculate_stock_period_returns(
        market_data=market_data,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    if stock_returns.empty:
        raise ValueError(
            "No stock returns available "
            "for the selected period."
        )

    selected_return = stock_returns[
        stock_returns["Ticker"] == selected_ticker
    ]

    if selected_return.empty:
        raise ValueError(
            "Selected stock does not have "
            "a valid period return."
        )

    period_return = float(
        selected_return.iloc[0]["Returns"]
    )

    ranked_returns = (
        stock_returns
        .dropna(subset=["Returns"])
        .sort_values(
            "Returns",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    stock_rank = int(
        ranked_returns.index[
            ranked_returns["Ticker"]
            == selected_ticker
        ][0]
        + 1
    )

    stocks_tracked = len(ranked_returns)

    stock_info = universe[
        universe["yf_ticker"] == selected_ticker
    ]

    if stock_info.empty:
        raise ValueError(
            "Selected stock could not be matched "
            "to the universe."
        )

    stock_info = stock_info.iloc[0]

    sector = stock_info["sector"]

    sector_returns = stock_returns.merge(
        universe[
            ["yf_ticker", "sector"]
        ],
        left_on="Ticker",
        right_on="yf_ticker",
        how="inner",
    )

    sector_returns = sector_returns[
        sector_returns["sector"] == sector
    ].copy()

    sector_returns = (
        sector_returns
        .sort_values(
            "Returns",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    sector_rank = int(
        sector_returns.index[
            sector_returns["Ticker"]
            == selected_ticker
        ][0]
        + 1
    )

    sector_stocks_tracked = len(
        sector_returns
    )

    latest_price = float(
        stock_data.iloc[-1]["Close"]
    )

    last_252 = stock_data.tail(252)

    high_52_week = float(
        last_252["Close"].max()
    )

    low_52_week = float(
        last_252["Close"].min()
    )

    distance_from_high = (
        latest_price / high_52_week
    ) - 1.0

    if (
        not last_252["Volume"]
        .dropna()
        .empty
    ):
        average_volume = float(
            last_252["Volume"]
            .dropna()
            .mean()
        )
    else:
        average_volume = float("nan")

    return {
        "Stock": stock_info["symbol"],
        "Ticker": selected_ticker,
        "Sector": sector,
        "Latest Price": latest_price,
        "Period Return": period_return,
        "52W High": high_52_week,
        "52W Low": low_52_week,
        "Distance From 52W High": distance_from_high,
        "Average Volume": average_volume,
        "Market Rank": stock_rank,
        "Stocks Tracked": stocks_tracked,
        "Sector Rank": sector_rank,
        "Sector Stocks Tracked": sector_stocks_tracked,
    }


def render_stock_snapshot(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    selected_ticker: str,
    start_date,
    end_date,
) -> None:
    st.subheader("Stock Overview")

    data = build_stock_snapshot(
        market_data=market_data,
        universe=universe,
        selected_ticker=selected_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Price",
        f"{data['Latest Price']:,.2f}",
    )

    col2.metric(
        "Period Return",
        f"{data['Period Return']:.2%}",
    )

    col3.metric(
        "52W High",
        f"{data['52W High']:,.2f}",
    )

    col4.metric(
        "52W Low",
        f"{data['52W Low']:,.2f}",
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "From 52W High",
        f"{data['Distance From 52W High']:.2%}",
    )

    col2.metric(
        "Market Rank",
        f"{data['Market Rank']} / "
        f"{data['Stocks Tracked']}",
    )

    col3.metric(
        "Sector Rank",
        f"{data['Sector Rank']} / "
        f"{data['Sector Stocks Tracked']}",
    )

    col4.metric(
        "Average Volume",
        (
            f"{data['Average Volume']:,.0f}"
            if pd.notna(data["Average Volume"])
            else "N/A"
        ),
    )

    st.caption(
        f"{data['Stock']} · "
        f"{data['Sector']} · "
        f"{data['Ticker']}"
    )