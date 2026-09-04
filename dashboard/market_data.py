from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics.market import get_market_analysis


# ============================================================
# CACHED MARKET ANALYTICS
# ============================================================

@st.cache_data(show_spinner=False)
def _get_cached_market_analysis(
    data: pd.DataFrame,
    selected_universe: pd.DataFrame,
):
    return get_market_analysis(
        market_data=data,
        universe_data=selected_universe,
        risk_free_rate=0.0,
    )


# ============================================================
# HELPERS
# ============================================================

def _format_pct(value) -> str:
    if pd.isna(value):
        return "—"

    return f"{value:.2%}"


def _format_number(value) -> str:
    if pd.isna(value):
        return "—"

    return f"{value:.2f}"


# ============================================================
# MARKET OVERVIEW
# ============================================================

def render_market_overview(
    data: pd.DataFrame,
    selected_universe: pd.DataFrame,
    selection_type: str,
    sector: str | None,
    index_name: str | None,
    start_date,
    end_date,
) -> None:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if selection_type == "Sector":
        title = sector or "Sector"

    elif selection_type == "Index":
        title = index_name or "Index"

    else:
        title = "Custom Universe"

    st.header("Market Research")

    st.caption(
        f"{title} • {start_date} → {end_date}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if data.empty:
        st.warning("No market data available.")
        return

    if selected_universe.empty:
        st.warning("No universe information available.")
        return

    # --------------------------------------------------------
    # MARKET ANALYTICS
    # --------------------------------------------------------

    analysis = _get_cached_market_analysis(
        data,
        selected_universe,
    )

    sector_returns = analysis["sector_returns"]
    market_returns = analysis["market_returns"]
    performance = analysis["performance"]
    risk_return = analysis["risk_return"]
    correlation = analysis["correlation"]

    if sector_returns.empty:
        st.warning(
            "Insufficient data to calculate market analytics."
        )
        return

    # ========================================================
    # MARKET METRICS
    # ========================================================

    market_total_return = (
        (1 + market_returns).prod() - 1
        if not market_returns.empty
        else np.nan
    )

    market_annualized_return = (
        performance["annualized_return"].mean()
        if not performance.empty
        else np.nan
    )

    market_volatility = (
        market_returns.std() * np.sqrt(252)
        if not market_returns.empty
        else np.nan
    )

    if (
        not market_returns.empty
        and market_returns.std() > 0
    ):
        market_sharpe = (
            market_returns.mean()
            / market_returns.std()
            * np.sqrt(252)
        )
    else:
        market_sharpe = np.nan

    market_wealth = (
        1 + market_returns
    ).cumprod()

    market_drawdown = (
        market_wealth
        / market_wealth.cummax()
    ) - 1

    market_max_drawdown = (
        market_drawdown.min()
        if not market_drawdown.empty
        else np.nan
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    cols = st.columns(5)

    cols[0].metric(
        "Total Return",
        _format_pct(market_total_return),
    )

    cols[1].metric(
        "Avg Sector Return",
        _format_pct(market_annualized_return),
    )

    cols[2].metric(
        "Volatility",
        _format_pct(market_volatility),
    )

    cols[3].metric(
        "Sharpe Ratio",
        _format_number(market_sharpe),
    )

    cols[4].metric(
        "Max Drawdown",
        _format_pct(market_max_drawdown),
    )

    st.divider()

    # ========================================================
    # MARKET PERFORMANCE
    # ========================================================

    st.subheader("Market Performance")

    market_growth = (
        (1 + market_returns)
        .cumprod()
        .sub(1)
        .mul(100)
    )

    market_growth.name = "Market"

    st.line_chart(
        market_growth,
        use_container_width=True,
    )

    st.caption(
        "Equal-weighted aggregate of the available sectors."
    )

    st.divider()

    # ========================================================
    # SECTOR PERFORMANCE
    # ========================================================

    st.subheader("Sector Performance")

    if performance.empty:
        st.info("No sector performance data available.")

    else:

        display = performance.copy()

        display = display.rename(
            columns={
                "sector": "Sector",
                "total_return": "Total Return",
                "annualized_return": "Annualized Return",
                "annualized_volatility": "Volatility",
                "sharpe_ratio": "Sharpe",
                "maximum_drawdown": "Max Drawdown",
            }
        )

        st.dataframe(
            display.style.format(
                {
                    "Total Return": "{:.2%}",
                    "Annualized Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ========================================================
    # SECTOR CUMULATIVE PERFORMANCE
    # ========================================================

    st.subheader("Sector Cumulative Performance")

    cumulative = (
        (1 + sector_returns)
        .cumprod()
        .sub(1)
        .mul(100)
    )

    st.line_chart(
        cumulative,
        use_container_width=True,
    )

    st.caption(
        "Cumulative return of each sector using "
        "equal-weighted daily constituent returns."
    )

    st.divider()

    # ========================================================
    # RISK / RETURN
    # ========================================================

    st.subheader("Sector Risk vs Return")

    if not risk_return.empty:

        chart_data = risk_return.copy()

        chart_data = chart_data.rename(
            columns={
                "sector": "Sector",
                "annualized_volatility": "Volatility",
                "annualized_return": "Return",
                "sharpe_ratio": "Sharpe",
            }
        )

        st.scatter_chart(
            chart_data,
            x="Volatility",
            y="Return",
            size="Sharpe",
            color="Sector",
            use_container_width=True,
        )

    st.divider()

    # ========================================================
    # CORRELATION
    # ========================================================

    st.subheader("Sector Correlation")

    if correlation.empty:
        st.info("No correlation data available.")

    else:

        st.dataframe(
            correlation.style.format("{:.2f}"),
            use_container_width=True,
        )

    st.divider()

    # ========================================================
    # DRAWDOWN
    # ========================================================

    st.subheader("Sector Drawdown")

    drawdowns = {}

    for column in sector_returns.columns:

        returns = sector_returns[column].dropna()

        if returns.empty:
            continue

        wealth = (
            1 + returns
        ).cumprod()

        drawdown = (
            wealth
            / wealth.cummax()
        ) - 1

        drawdowns[column] = drawdown

    if drawdowns:

        drawdown_df = pd.DataFrame(drawdowns)

        st.line_chart(
            drawdown_df,
            use_container_width=True,
        )


# ============================================================
# MARKET DATA
# ============================================================

def render_market_data(
    data: pd.DataFrame,
    selected_universe: pd.DataFrame,
) -> None:

    st.subheader("Latest Market Data")

    if data.empty:
        st.info("No market data available.")
        return

    latest_date = data["Date"].max()

    latest = (
        data[data["Date"] == latest_date]
        .copy()
    )

    latest = latest.merge(
        selected_universe[
            [
                "symbol",
                "sector",
                "yf_ticker",
            ]
        ],
        left_on="Ticker",
        right_on="yf_ticker",
        how="left",
    )

    latest = latest[
        [
            "symbol",
            "sector",
            "Date",
            "Close",
            "Volume",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
        }
    )

    latest = latest.sort_values(
        ["Sector", "Symbol"]
    )

    st.dataframe(
        latest,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MARKET SECTION
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

    render_market_overview(
        data=data,
        selected_universe=selected_universe,
        selection_type=selection_type,
        sector=sector,
        index_name=index_name,
        start_date=start_date,
        end_date=end_date,
    )