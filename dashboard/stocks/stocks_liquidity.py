from __future__ import annotations

import streamlit as st

from src.analytics.stocks.liquidity import (
    calculate_stock_liquidity_metrics,
)


def _format_number(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value:,.{decimals}f}"


def _format_scientific(value) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value:.2e}"


def render_stock_liquidity(
    market_data,
    ticker: str,
    start_date,
    end_date,
) -> None:
    """Render the Liquidity section for the selected stock."""

    st.subheader("Liquidity")

    metrics = calculate_stock_liquidity_metrics(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    row1 = st.columns(2)

    row1[0].metric(
        "Average Volume",
        _format_number(metrics["Average Volume"], 0),
    )

    row1[1].metric(
        "Average Traded Value",
        _format_number(metrics["Average Traded Value"], 0),
    )

    row2 = st.columns(2)

    row2[0].metric(
        "Amihud Illiquidity",
        _format_scientific(metrics["Amihud Illiquidity"]),
    )

    row2[1].metric(
        "Volume Z-Score",
        _format_number(metrics["Volume Z-Score"]),
    )