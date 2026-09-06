from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics.stocks.market import (
    calculate_market_breadth,
    calculate_stock_period_returns,
    find_best_stock,
    find_worst_stock,
)


def calculate_stock_correlation_pairs(
    returns: pd.DataFrame,
    start_date,
    end_date,
):
    period_returns = returns.loc[
        (returns.index >= pd.Timestamp(start_date))
        & (returns.index <= pd.Timestamp(end_date))
    ].copy()

    if period_returns.empty:
        raise ValueError(
            "No return data available for correlation analysis."
        )

    correlation = period_returns.corr()

    if correlation.shape[0] < 2:
        raise ValueError(
            "At least two stocks are required for correlation analysis."
        )

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

    most_correlated_pair = pairs.idxmax()
    most_correlated_value = float(
        pairs.loc[most_correlated_pair]
    )

    least_correlated_pair = pairs.abs().idxmin()
    least_correlated_value = float(
        pairs.loc[least_correlated_pair]
    )

    return {
        "Most Correlated Pair": (
            f"{most_correlated_pair[0]} / "
            f"{most_correlated_pair[1]}"
        ),
        "Most Correlated Value": most_correlated_value,
        "Least Correlated Pair": (
            f"{least_correlated_pair[0]} / "
            f"{least_correlated_pair[1]}"
        ),
        "Least Correlated Value": least_correlated_value,
    }


def build_snapshot_data(
    snapshot: dict[str, float],
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    start_date,
    end_date,
) -> dict:
    stock_returns = calculate_stock_period_returns(
        market_data=market_data,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    breadth = calculate_market_breadth(stock_returns)

    best_stock = find_best_stock(stock_returns)
    worst_stock = find_worst_stock(stock_returns)

    period_returns = returns.loc[
        (returns.index >= pd.Timestamp(start_date))
        & (returns.index <= pd.Timestamp(end_date))
    ].copy()

    sector_returns = {}

    universe_mapping = universe[
        ["sector", "yf_ticker"]
    ].copy()

    universe_mapping["yf_ticker"] = (
        universe_mapping["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    for sector in universe_mapping["sector"].dropna().unique():
        tickers = universe_mapping.loc[
            universe_mapping["sector"] == sector,
            "yf_ticker",
        ].tolist()

        available_tickers = [
            ticker
            for ticker in tickers
            if ticker in period_returns.columns
        ]

        if not available_tickers:
            continue

        daily_sector_returns = (
            period_returns[available_tickers]
            .mean(axis=1, skipna=True)
            .dropna()
        )

        if daily_sector_returns.empty:
            continue

        sector_returns[sector] = (
            1.0 + daily_sector_returns
        ).prod() - 1.0

    if not sector_returns:
        raise ValueError(
            "No sector returns available for snapshot."
        )

    sector_return_series = pd.Series(sector_returns)

    best_sector = sector_return_series.idxmax()
    worst_sector = sector_return_series.idxmin()

    correlation_pairs = calculate_stock_correlation_pairs(
        returns=returns,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "Returns": snapshot["Returns"],
        "Volatility": snapshot["Volatility"],
        "Sharpe": snapshot["Sharpe"],
        "Max Drawdown": snapshot["Max Drawdown"],
        "Top Sector": best_sector,
        "Top Sector Return": float(
            sector_return_series.loc[best_sector]
        ),
        "Worst Sector": worst_sector,
        "Worst Sector Return": float(
            sector_return_series.loc[worst_sector]
        ),
        "Top Stock": best_stock["Ticker"],
        "Top Stock Return": float(
            best_stock["Returns"]
        ),
        "Worst Stock": worst_stock["Ticker"],
        "Worst Stock Return": float(
            worst_stock["Returns"]
        ),
        "Advancing Stocks": breadth["Advancing Stocks"],
        "Declining Stocks": breadth["Declining Stocks"],
        "Stocks Tracked": breadth["Stocks Tracked"],
        "Positive Stocks %": (
            breadth["Advancing Stocks"]
            / breadth["Stocks Tracked"]
            if breadth["Stocks Tracked"] > 0
            else float("nan")
        ),
        "Most Correlated Pair": correlation_pairs[
            "Most Correlated Pair"
        ],
        "Most Correlated Value": correlation_pairs[
            "Most Correlated Value"
        ],
        "Least Correlated Pair": correlation_pairs[
            "Least Correlated Pair"
        ],
        "Least Correlated Value": correlation_pairs[
            "Least Correlated Value"
        ],
    }


def render_snapshot(
    snapshot: dict[str, float],
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    start_date,
    end_date,
) -> None:
    st.subheader("Market Snapshot")

    data = build_snapshot_data(
        snapshot=snapshot,
        market_data=market_data,
        universe=universe,
        returns=returns,
        start_date=start_date,
        end_date=end_date,
    )

    # Core market metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Returns",
        f"{data['Returns']:.2%}",
    )

    col2.metric(
        "Volatility",
        f"{data['Volatility']:.2%}",
    )

    col3.metric(
        "Sharpe",
        f"{data['Sharpe']:.2f}",
    )

    col4.metric(
        "Max Drawdown",
        f"{data['Max Drawdown']:.2%}",
    )

    # Best and worst performers
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Top Sector",
        data["Top Sector"],
        f"{data['Top Sector Return']:.2%}",
    )

    col2.metric(
        "Worst Sector",
        data["Worst Sector"],
        f"{data['Worst Sector Return']:.2%}",
    )

    col3.metric(
        "Top Stock",
        data["Top Stock"],
        f"{data['Top Stock Return']:.2%}",
    )

    col4.metric(
        "Worst Stock",
        data["Worst Stock"],
        f"{data['Worst Stock Return']:.2%}",
    )

    # Market breadth
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Advancing Stocks",
        f"{data['Advancing Stocks']}",
    )

    col2.metric(
        "Declining Stocks",
        f"{data['Declining Stocks']}",
    )

    col3.metric(
        "Stocks Positive",
        f"{data['Positive Stocks %']:.1%}",
    )

    # Correlation summary
    col1, col2 = st.columns(2)

    col1.metric(
        "Most Correlated Stock Pair",
        data["Most Correlated Pair"],
        f"{data['Most Correlated Value']:.2f}",
    )

    col2.metric(
        "Least Correlated Stock Pair",
        data["Least Correlated Pair"],
        f"{data['Least Correlated Value']:.2f}",
    )