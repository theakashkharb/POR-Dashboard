from __future__ import annotations

import streamlit as st

from src.analytics.stocks.technical import (
    calculate_stock_technical_metrics,
)


def _format_value(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value:.{decimals}f}"


def _format_percent(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def render_stock_price_structure(
    market_data,
    ticker: str,
    start_date,
    end_date,
) -> None:
    """Render the Price Structure section for the selected stock."""

    st.subheader("Price Structure")

    metrics = calculate_stock_technical_metrics(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    row1 = st.columns(4)

    row1[0].metric(
        "Current Price",
        _format_value(metrics["Current Price"]),
    )

    row1[1].metric(
        "52W High",
        _format_value(metrics["52W High"]),
    )

    row1[2].metric(
        "52W Low",
        _format_value(metrics["52W Low"]),
    )

    row1[3].metric(
        "MA Trend",
        metrics["MA Trend"],
    )

    row2 = st.columns(4)

    row2[0].metric(
        "Distance from 52W High",
        _format_percent(metrics["Distance from 52W High"]),
    )

    row2[1].metric(
        "Distance from 52W Low",
        _format_percent(metrics["Distance from 52W Low"]),
    )

    row2[2].metric(
        "Price vs 50 DMA",
        _format_percent(metrics["Price vs 50 DMA"]),
    )

    row2[3].metric(
        "Price vs 200 DMA",
        _format_percent(metrics["Price vs 200 DMA"]),
    )

    row3 = st.columns(2)

    row3[0].metric(
        "Autocorrelation Lag 1",
        _format_value(metrics["Autocorrelation Lag 1"]),
    )

    row3[1].metric(
        "Autocorrelation Lag 5",
        _format_value(metrics["Autocorrelation Lag 5"]),
    )