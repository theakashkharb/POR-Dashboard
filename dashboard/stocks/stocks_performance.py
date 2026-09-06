from __future__ import annotations

import streamlit as st

from src.analytics.stocks.performance import (
    calculate_rolling_1y_return,
    calculate_stock_performance,
    calculate_win_rate,
)


def render_stock_performance(
    market_data,
    selected_ticker: str,
    start_date,
    end_date,
) -> None:
    """
    Render the stock Performance section.

    Quantitative calculations are handled by:
        src.analytics.stocks.performance
    """

    st.subheader("Performance")

    performance = calculate_stock_performance(
        market_data=market_data,
        ticker=selected_ticker,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    win_rate = calculate_win_rate(
        market_data=market_data,
        ticker=selected_ticker,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    rolling_1y = calculate_rolling_1y_return(
        market_data=market_data,
        ticker=selected_ticker,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "1M Return",
        _format_percentage(performance["1M Return"]),
    )

    col2.metric(
        "3M Return",
        _format_percentage(performance["3M Return"]),
    )

    col3.metric(
        "6M Return",
        _format_percentage(performance["6M Return"]),
    )

    col4.metric(
        "1Y Return",
        _format_percentage(performance["1Y Return"]),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "3Y Return",
        _format_percentage(performance["3Y Return"]),
    )

    col2.metric(
        "CAGR",
        _format_percentage(performance["CAGR"]),
    )

    col3.metric(
        "Daily Win Rate",
        _format_percentage(win_rate["Daily Win Rate"]),
    )

    col4.metric(
        "Monthly Win Rate",
        _format_percentage(win_rate["Monthly Win Rate"]),
    )

    if not rolling_1y.empty:
        st.caption(
            "Rolling 1Y Return — calculated daily"
        )

        latest_rolling_1y = rolling_1y.iloc[-1]

        st.metric(
            "Latest Rolling 1Y Return",
            _format_percentage(latest_rolling_1y),
        )


def _format_percentage(value) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:
            return "N/A"
    except Exception:
        return "N/A"

    return f"{float(value):.2%}"