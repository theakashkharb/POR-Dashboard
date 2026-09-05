from __future__ import annotations

from __future__ import annotations

import numpy as np

import pandas as pd

import streamlit as st

import plotly.graph_objects as go


from .core import *
from .core import (
    _safe_float,
    _format_percent,
    _format_sharpe,
    _normalize_dates,
    _analysis_window,
    _get_price_matrix,
    _get_returns,
    _annualized_return,
    _total_return,
    _annualized_volatility,
    _sharpe_ratio,
    _maximum_drawdown,
    _calculate_sector_returns,
    _market_metrics,
    _pastel_sector_style,
    _pastel_stock_style,
    _correlation_relationships,
    TRADING_DAYS,
    PERIOD_OPTIONS,
)

# ============================================================
# SECTION
# ============================================================

def _sector_correlation(
    sector_returns: pd.DataFrame,
) -> pd.DataFrame:

    if sector_returns.empty:
        return pd.DataFrame()

    return sector_returns.corr()


def _render_sector_correlation(
    sector_returns: pd.DataFrame,
) -> None:

    st.subheader("Sector Correlation")

    correlation = _sector_correlation(
        sector_returns
    )

    if correlation.empty:
        st.info("No correlation data available.")
        return

    display = correlation.copy()

    # Round only for presentation.
    display = display.round(2)

    # IMPORTANT:
    # No background colors here.
    # Correlation table intentionally stays clean.
    st.dataframe(
        display,
        use_container_width=True,
        height=650,
        hide_index=False,
    )

    st.caption(
        "Correlation ranges from -1 to +1. "
        "Positive values indicate sectors tend to move together; "
        "negative values indicate opposite movement; "
        "values near zero indicate little linear relationship."
    )

    relationships = _correlation_relationships(
        correlation
    )

    highest = relationships["highest_positive"]
    negative = relationships["strongest_negative"]
    weakest = relationships["weakest"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Highest Positive Correlation**")

        if highest:
            a, b, value = highest

            st.write(
                f"{a} ↔ {b}"
            )

            st.metric(
                "Correlation",
                f"{value:+.2f}",
            )

    with c2:
        st.markdown("**Strongest Negative Correlation**")

        if negative and negative[2] < 0:
            a, b, value = negative

            st.write(
                f"{a} ↔ {b}"
            )

            st.metric(
                "Correlation",
                f"{value:+.2f}",
            )
        else:
            st.write(
                "No negative correlation pair in this period."
            )

    with c3:
        st.markdown("**Weakest Correlation**")

        if weakest:
            a, b, value = weakest

            st.write(
                f"{a} ↔ {b}"
            )

            st.metric(
                "Correlation",
                f"{value:+.2f}",
            )


