from __future__ import annotations

import pandas as pd
import streamlit as st

from .core import (
    _analysis_window,
    _calculate_sector_returns,
    _market_metrics,
    _pastel_sector_style,
    PERIOD_OPTIONS,
)

from .snapshot import (
    _render_market_snapshot,
)

from .market_map import (
    _render_market_map,
)

from .sector_performance import (
    _build_sector_performance,
)

from .top_stocks import (
    _build_stock_performance,
    _render_top_performing_stocks,
)

from .correlation import (
    _sector_correlation,
    _render_sector_correlation,
)


# ============================================================
# SAFE SECTION RENDERER
# ============================================================

def _render_section_safely(
    section_name: str,
    render_function,
    *args,
    **kwargs,
) -> None:

    try:

        render_function(
            *args,
            **kwargs,
        )

    except Exception as exc:

        st.error(
            f"{section_name} failed to render."
        )

        st.exception(exc)


# ============================================================
# MAIN MARKET PAGE
# ============================================================

def render_market_section(
    data: pd.DataFrame,
    selected_universe: pd.DataFrame,
    selection_type: str,
    sector: str | None,
    index_name: str | None,
    start_date,
    end_date,
) -> None:

    st.header("Market")

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    if (
        selection_type == "Sector"
        and sector
    ):

        universe_label = sector

    elif (
        selection_type == "Index"
        and index_name
    ):

        universe_label = index_name

    elif selection_type == "Custom Stocks":

        universe_label = "Custom Stocks"

    else:

        universe_label = selection_type

    # --------------------------------------------------------
    # FULL LOADED DATA RANGE
    # --------------------------------------------------------

    try:

        full_data_start = (
            pd.to_datetime(
                data["Date"]
            )
            .min()
            .date()
        )

        full_data_end = (
            pd.to_datetime(
                data["Date"]
            )
            .max()
            .date()
        )

    except Exception:

        full_data_start = start_date
        full_data_end = end_date

    st.caption(
        f"{universe_label} • "
        f"{full_data_start} → {full_data_end}"
    )

    # --------------------------------------------------------
    # TIME PERIOD
    # --------------------------------------------------------

    period = st.selectbox(
        "Time Period",
        list(PERIOD_OPTIONS.keys()),
        index=0,
        key="market_analysis_period",
    )

    period_definition = (
        PERIOD_OPTIONS[period]
    )

    analysis_data = _analysis_window(
        data=data,
        start_date=start_date,
        end_date=end_date,
        period=period_definition,
    )

    if analysis_data.empty:

        st.warning(
            "No data is available for the selected time period."
        )

        return

    analysis_start = (
        analysis_data["Date"]
        .min()
        .date()
    )

    analysis_end = (
        analysis_data["Date"]
        .max()
        .date()
    )

    st.caption(
        f"Analysis window: "
        f"{analysis_start} → {analysis_end}"
    )

    # ========================================================
    # SHARED ANALYTICS
    # ========================================================

    try:

        sector_returns = (
            _calculate_sector_returns(
                data=analysis_data,
                universe=selected_universe,
            )
        )

    except Exception as exc:

        st.error(
            "Market analytics calculation failed."
        )

        st.exception(exc)

        return

    if sector_returns.empty:

        st.warning(
            "Unable to calculate sector analytics "
            "for the selected universe."
        )

        return

    # --------------------------------------------------------
    # SECTOR PERFORMANCE
    # --------------------------------------------------------

    try:

        performance = (
            _build_sector_performance(
                sector_returns
            )
        )

    except Exception as exc:

        performance = pd.DataFrame()

        st.warning(
            "Sector performance calculation failed."
        )

        st.exception(exc)

    # --------------------------------------------------------
    # STOCK PERFORMANCE
    # --------------------------------------------------------

    try:

        stock_performance = (
            _build_stock_performance(
                data=analysis_data,
                universe=selected_universe,
            )
        )

    except Exception as exc:

        stock_performance = pd.DataFrame()

        st.warning(
            "Stock performance calculation failed."
        )

        st.exception(exc)

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    try:

        correlation = (
            _sector_correlation(
                sector_returns
            )
        )

    except Exception as exc:

        correlation = pd.DataFrame()

        st.warning(
            "Sector correlation calculation failed."
        )

        st.exception(exc)

    # --------------------------------------------------------
    # MARKET METRICS
    # --------------------------------------------------------

    try:

        metrics = _market_metrics(
            sector_returns
        )

    except Exception as exc:

        metrics = {
            "market_returns": pd.Series(
                dtype=float
            )
        }

        st.warning(
            "Market metrics calculation failed."
        )

        st.exception(exc)

    # ========================================================
    # 1. MARKET SNAPSHOT
    # ========================================================

    _render_section_safely(
        "Market Snapshot",
        _render_market_snapshot,
        metrics=metrics,
        sector_returns=sector_returns,
        market_returns=metrics.get(
            "market_returns",
            pd.Series(dtype=float),
        ),
        correlation=correlation,
        stock_performance=stock_performance,
    )

    # ========================================================
    # 2. MARKET MAP
    # ========================================================

    _render_section_safely(
        "Market Map",
        _render_market_map,
        stock_performance,
    )

    # ========================================================
    # 3. SECTOR PERFORMANCE
    # ========================================================

    def render_sector_performance():

        st.subheader(
            "Sector Performance"
        )

        if performance.empty:

            st.info(
                "No sector performance data available."
            )

            return

        display = performance[
            [
                "Sector",
                "Total Return",
                "Annualized Return",
                "Volatility",
                "Sharpe",
                "Max Drawdown",
            ]
        ].copy()

        display = display.sort_values(
            "Annualized Return",
            ascending=False,
        )

        styled = _pastel_sector_style(
            display
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Ranked by annualized return. "
            "Pastel green indicates stronger performance; "
            "pastel red indicates higher risk or weaker outcomes."
        )

    _render_section_safely(
        "Sector Performance",
        render_sector_performance,
    )

    # ========================================================
    # 4. TOP PERFORMING STOCKS
    # ========================================================

    _render_section_safely(
        "Top Performing Stocks",
        _render_top_performing_stocks,
        stock_performance=stock_performance,
        selection_type=selection_type,
        sector=sector,
    )

    # ========================================================
    # 5. SECTOR CORRELATION
    # ========================================================

    _render_section_safely(
        "Sector Correlation",
        _render_sector_correlation,
        sector_returns,
    )
