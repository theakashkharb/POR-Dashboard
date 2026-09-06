from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.stocks import calculate_stock_period_returns


CORRELATION_STOCK_COUNT = 20


def build_stock_correlation(
    market_data,
    start_date,
    end_date,
):
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
        .sort_values("Returns", ascending=False)
        .head(CORRELATION_STOCK_COUNT)
    )

    selected_tickers = top_stocks["Ticker"].tolist()

    data = market_data[
        market_data["Ticker"].isin(selected_tickers)
        & (market_data["Date"] >= pd.Timestamp(start_date))
        & (market_data["Date"] <= pd.Timestamp(end_date))
    ].copy()

    if data.empty:
        raise ValueError(
            "No price data available for the selected stocks."
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

    stock_returns_daily = prices.pct_change(fill_method=None)

    correlation = stock_returns_daily.corr()

    return correlation


def find_correlation_pairs(correlation):
    upper_triangle = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1,
        ).astype(bool)
    )

    pairs = upper_triangle.stack().dropna()

    if pairs.empty:
        raise ValueError(
            "No valid stock correlation pairs available."
        )

    most_pair = pairs.idxmax()
    most_value = float(pairs.loc[most_pair])

    least_pair = pairs.abs().idxmin()
    least_value = float(pairs.loc[least_pair])

    return {
        "Most Pair": f"{most_pair[0]} / {most_pair[1]}",
        "Most Value": most_value,
        "Least Pair": f"{least_pair[0]} / {least_pair[1]}",
        "Least Value": least_value,
    }


def render_stock_correlation(
    market_data,
    start_date,
    end_date,
):
    st.subheader("Stock Correlation")

    correlation = build_stock_correlation(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )

    if correlation.empty:
        st.info(
            "No Stock Correlation data available "
            "for the selected period."
        )
        return

    pairs = find_correlation_pairs(correlation)

    col1, col2 = st.columns(2)

    col1.metric(
        "Most Correlated Pair",
        pairs["Most Pair"],
        f"{pairs['Most Value']:.2f}",
    )

    col2.metric(
        "Least Correlated Pair",
        pairs["Least Pair"],
        f"{pairs['Least Value']:.2f}",
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale=[
            [0.0, "#d73027"],
            [0.5, "#f7f7f7"],
            [1.0, "#1a9850"],
        ],
    )

    fig.update_layout(
        margin=dict(
            t=10,
            l=10,
            r=10,
            b=10,
        ),
        height=800,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )