import numpy as np
import pandas as pd
import streamlit as st

from src.features.volatility import (
    calculate_historical_volatility,
    calculate_rolling_historical_volatility,
    calculate_ewma_volatility,
    calculate_rolling_ewma_volatility,
)


# ============================================================
# HELPERS
# ============================================================

def _symbol_name(ticker: str) -> str:
    return ticker.replace(".NS", "")


# ============================================================
# CURRENT VOLATILITY
# ============================================================

def get_current_volatility(
    data,
    historical_window,
    ewma_span,
):

    historical = calculate_historical_volatility(
        data,
        window=historical_window,
    ).rename("Historical Volatility")

    ewma = calculate_ewma_volatility(
        data,
        span=ewma_span,
    ).rename("EWMA Volatility")

    result = pd.concat(
        [historical, ewma],
        axis=1,
    )

    result.index.name = "Ticker"

    return result.reset_index()


# ============================================================
# HISTORICAL VOLATILITY
# ============================================================

def render_historical_volatility(
    data,
    window,
):

    st.subheader("Historical Volatility")

    volatility = calculate_rolling_historical_volatility(
        data,
        window=window,
    )

    tickers = sorted(
        volatility["Ticker"].unique()
    )

    selected = st.selectbox(
        "Select Stock",
        tickers,
        format_func=_symbol_name,
        key="historical_vol_stock",
    )

    stock = volatility[
        volatility["Ticker"] == selected
    ].copy()

    stock["Date"] = pd.to_datetime(
        stock["Date"]
    )

    chart = (
        stock
        .set_index("Date")
        [["Historical_Volatility"]]
        * 100
    )

    st.line_chart(
        chart,
        use_container_width=True,
        y_label="Annualized Volatility (%)",
    )

    current = stock[
        "Historical_Volatility"
    ].iloc[-1]

    st.metric(
        "Current Historical Volatility",
        f"{current:.2%}",
    )


# ============================================================
# EWMA VOLATILITY
# ============================================================

def render_ewma_volatility(
    data,
    span,
):

    st.subheader("EWMA Volatility")

    volatility = calculate_rolling_ewma_volatility(
        data,
        span=span,
    )

    tickers = sorted(
        volatility["Ticker"].unique()
    )

    selected = st.selectbox(
        "Select Stock",
        tickers,
        format_func=_symbol_name,
        key="ewma_vol_stock",
    )

    stock = volatility[
        volatility["Ticker"] == selected
    ].copy()

    stock["Date"] = pd.to_datetime(
        stock["Date"]
    )

    chart = (
        stock
        .set_index("Date")
        [["EWMA_Volatility"]]
        * 100
    )

    st.line_chart(
        chart,
        use_container_width=True,
        y_label="Annualized Volatility (%)",
    )

    current = stock[
        "EWMA_Volatility"
    ].iloc[-1]

    st.metric(
        "Current EWMA Volatility",
        f"{current:.2%}",
    )


# ============================================================
# CURRENT VOLATILITY
# ============================================================

def render_current_volatility(
    data,
    historical_window,
    ewma_span,
):

    st.subheader("Current Volatility")

    result = get_current_volatility(
        data,
        historical_window,
        ewma_span,
    )

    result["Asset"] = result["Ticker"].map(
        _symbol_name
    )

    result["Difference"] = (
        result["EWMA Volatility"]
        - result["Historical Volatility"]
    )

    display = result[
        [
            "Asset",
            "Historical Volatility",
            "EWMA Volatility",
            "Difference",
        ]
    ]

    st.dataframe(
        display.style.format(
            {
                "Historical Volatility": "{:.2%}",
                "EWMA Volatility": "{:.2%}",
                "Difference": "{:+.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VOLATILITY STATISTICS
# ============================================================

def render_volatility_statistics(data):

    st.subheader("Volatility Statistics")

    historical = calculate_rolling_historical_volatility(
        data,
        window=90,
    )

    stats = (
        historical
        .groupby("Ticker")["Historical_Volatility"]
        .agg(
            Average="mean",
            Minimum="min",
            Maximum="max",
            Current="last",
        )
        .reset_index()
    )

    stats["Asset"] = stats["Ticker"].map(
        _symbol_name
    )

    stats = stats[
        [
            "Asset",
            "Average",
            "Minimum",
            "Maximum",
            "Current",
        ]
    ]

    st.dataframe(
        stats.style.format(
            {
                "Average": "{:.2%}",
                "Minimum": "{:.2%}",
                "Maximum": "{:.2%}",
                "Current": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VOLATILITY RANKING
# ============================================================

def render_volatility_ranking(data):

    st.subheader("Volatility Ranking")

    current = calculate_historical_volatility(
        data,
        window=90,
    )

    ranking = (
        current
        .rename("Annualized Volatility")
        .reset_index()
        .sort_values(
            "Annualized Volatility",
            ascending=True,
        )
    )

    ranking["Asset"] = ranking["Ticker"].map(
        _symbol_name
    )

    ranking["Rank"] = np.arange(
        1,
        len(ranking) + 1,
    )

    ranking = ranking[
        [
            "Rank",
            "Asset",
            "Annualized Volatility",
        ]
    ]

    st.dataframe(
        ranking.style.format(
            {
                "Annualized Volatility":
                    "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# HISTORICAL VS EWMA
# ============================================================

def render_volatility_comparison(data):

    st.subheader("Historical vs EWMA")

    result = get_current_volatility(
        data,
        historical_window=90,
        ewma_span=90,
    )

    result["Asset"] = result["Ticker"].map(
        _symbol_name
    )

    comparison = result[
        [
            "Asset",
            "Historical Volatility",
            "EWMA Volatility",
        ]
    ].set_index("Asset")

    st.bar_chart(
        comparison * 100,
        use_container_width=True,
        y_label="Annualized Volatility (%)",
    )


# ============================================================
# VOLATILITY SECTION
# ============================================================

def render_volatility_section(data):

    st.header("Volatility Analysis")

    analysis = st.selectbox(
        "Analysis",
        [
            "Historical Volatility",
            "EWMA Volatility",
            "Current Volatility",
            "Volatility Statistics",
            "Volatility Ranking",
            "Historical vs EWMA",
        ],
        key="volatility_analysis_view",
    )

    st.divider()

    if analysis == "Historical Volatility":

        window = st.select_slider(
            "Historical Volatility Window",
            options=[20, 60, 90, 180, 252],
            value=90,
            key="historical_vol_window",
        )

        render_historical_volatility(
            data,
            window,
        )

    elif analysis == "EWMA Volatility":

        span = st.select_slider(
            "EWMA Span",
            options=[20, 60, 90, 180, 252],
            value=90,
            key="ewma_vol_span",
        )

        render_ewma_volatility(
            data,
            span,
        )

    elif analysis == "Current Volatility":

        col1, col2 = st.columns(2)

        with col1:

            historical_window = st.select_slider(
                "Historical Window",
                options=[20, 60, 90, 180, 252],
                value=90,
                key="current_historical_window",
            )

        with col2:

            ewma_span = st.select_slider(
                "EWMA Span",
                options=[20, 60, 90, 180, 252],
                value=90,
                key="current_ewma_span",
            )

        render_current_volatility(
            data,
            historical_window,
            ewma_span,
        )

    elif analysis == "Volatility Statistics":

        render_volatility_statistics(data)

    elif analysis == "Volatility Ranking":

        render_volatility_ranking(data)

    elif analysis == "Historical vs EWMA":

        render_volatility_comparison(data)