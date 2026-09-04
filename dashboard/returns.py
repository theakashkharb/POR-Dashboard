import numpy as np
import pandas as pd
import streamlit as st

from src.features.returns import (
    calculate_cumulative_returns,
    calculate_geometric_expected_return,
    calculate_historical_expected_return,
    calculate_rolling_returns,
    calculate_simple_returns,
    calculate_log_returns,
)


TRADING_DAYS = 252


# ============================================================
# HELPERS
# ============================================================

def _symbol_name(ticker: str) -> str:
    return ticker.replace(".NS", "")


def prepare_returns(
    data: pd.DataFrame,
    return_type: str,
):
    if return_type == "Simple":
        return calculate_simple_returns(data), "Return"

    return calculate_log_returns(data), "Log_Return"


# ============================================================
# DAILY RETURNS
# ============================================================

def render_daily_returns(data, return_type):

    st.subheader("Daily Returns")

    returns_data, return_column = prepare_returns(
        data,
        return_type,
    )

    tickers = sorted(
        returns_data["Ticker"].unique()
    )

    selected = st.selectbox(
        "Select Stock",
        tickers,
        format_func=_symbol_name,
        key="daily_return_stock",
    )

    stock = returns_data[
        returns_data["Ticker"] == selected
    ].copy()

    stock["Date"] = pd.to_datetime(stock["Date"])

    chart = (
        stock
        .set_index("Date")[[return_column]]
        * 100
    )

    st.line_chart(
        chart,
        use_container_width=True,
        y_label="Daily Return (%)",
    )

    series = stock[return_column].dropna()

    cols = st.columns(4)

    cols[0].metric(
        "Mean",
        f"{series.mean():.2%}",
    )

    cols[1].metric(
        "Volatility",
        f"{series.std():.2%}",
    )

    cols[2].metric(
        "Best Day",
        f"{series.max():.2%}",
    )

    cols[3].metric(
        "Worst Day",
        f"{series.min():.2%}",
    )


# ============================================================
# CUMULATIVE RETURNS
# ============================================================

def render_cumulative_returns(data, return_type):

    st.subheader("Cumulative Returns")

    returns_data, return_column = prepare_returns(
        data,
        return_type,
    )

    if return_column == "Return":

        cumulative = calculate_cumulative_returns(
            returns_data
        )

    else:

        returns_data = returns_data.sort_values(
            ["Ticker", "Date"]
        )

        returns_data["Cumulative_Return"] = (
            returns_data
            .groupby("Ticker")[return_column]
            .cumsum()
            .pipe(np.exp)
            - 1
        )

        cumulative = returns_data

    chart = cumulative.pivot(
        index="Date",
        columns="Ticker",
        values="Cumulative_Return",
    )

    chart.columns = [
        _symbol_name(x)
        for x in chart.columns
    ]

    st.line_chart(
        chart * 100,
        use_container_width=True,
        y_label="Cumulative Return (%)",
    )


# ============================================================
# ROLLING RETURNS
# ============================================================

def render_rolling_returns(data):

    st.subheader("Rolling Returns")

    col1, col2 = st.columns(2)

    with col1:

        window = st.select_slider(
            "Rolling Window",
            options=[20, 60, 90, 180, 252],
            value=90,
            key="rolling_return_window",
        )

    tickers = sorted(
        data["Ticker"].unique()
    )

    with col2:

        selected = st.selectbox(
            "Select Stock",
            tickers,
            format_func=_symbol_name,
            key="rolling_return_stock",
        )

    rolling = calculate_rolling_returns(
        data,
        window=window,
    )

    stock = rolling[
        rolling["Ticker"] == selected
    ].copy()

    stock["Date"] = pd.to_datetime(
        stock["Date"]
    )

    chart = (
        stock
        .set_index("Date")[["Rolling_Return"]]
        * 100
    )

    st.line_chart(
        chart,
        use_container_width=True,
        y_label="Rolling Return (%)",
    )

    series = stock["Rolling_Return"].dropna()

    cols = st.columns(3)

    cols[0].metric(
        "Current",
        f"{series.iloc[-1]:.2%}",
    )

    cols[1].metric(
        "Maximum",
        f"{series.max():.2%}",
    )

    cols[2].metric(
        "Minimum",
        f"{series.min():.2%}",
    )


# ============================================================
# RETURN DISTRIBUTION
# ============================================================

def render_return_distribution(
    data,
    return_type,
):

    st.subheader("Return Distribution")

    returns_data, return_column = prepare_returns(
        data,
        return_type,
    )

    tickers = sorted(
        returns_data["Ticker"].unique()
    )

    selected = st.selectbox(
        "Select Stock",
        tickers,
        format_func=_symbol_name,
        key="distribution_stock",
    )

    series = (
        returns_data[
            returns_data["Ticker"] == selected
        ][return_column]
        .dropna()
    )

    if series.empty:

        st.warning(
            "No return observations available."
        )

        return

    histogram, bins = np.histogram(
        series,
        bins=40,
    )

    histogram_data = pd.DataFrame(
        {
            "Return": bins[:-1] * 100,
            "Frequency": histogram,
        }
    )

    st.bar_chart(
        histogram_data.set_index("Return"),
        use_container_width=True,
        y_label="Frequency",
    )

    cols = st.columns(4)

    cols[0].metric(
        "Mean",
        f"{series.mean():.2%}",
    )

    cols[1].metric(
        "Median",
        f"{series.median():.2%}",
    )

    cols[2].metric(
        "Skewness",
        f"{series.skew():.2f}",
    )

    cols[3].metric(
        "Kurtosis",
        f"{series.kurt():.2f}",
    )


# ============================================================
# RETURN STATISTICS
# ============================================================

def render_return_statistics(
    data,
    return_type,
):

    st.subheader("Return Statistics")

    returns_data, return_column = prepare_returns(
        data,
        return_type,
    )

    rows = []

    for ticker in sorted(
        returns_data["Ticker"].unique()
    ):

        series = (
            returns_data[
                returns_data["Ticker"] == ticker
            ][return_column]
            .dropna()
        )

        if series.empty:
            continue

        rows.append(
            {
                "Asset": _symbol_name(ticker),
                "Mean": series.mean(),
                "Std Dev": series.std(),
                "Minimum": series.min(),
                "Maximum": series.max(),
                "Median": series.median(),
                "Win Rate": (series > 0).mean(),
                "Skewness": series.skew(),
                "Kurtosis": series.kurt(),
            }
        )

    statistics = pd.DataFrame(rows)

    if statistics.empty:

        st.warning(
            "No return statistics available."
        )

        return

    st.dataframe(
        statistics.style.format(
            {
                "Mean": "{:.2%}",
                "Std Dev": "{:.2%}",
                "Minimum": "{:.2%}",
                "Maximum": "{:.2%}",
                "Median": "{:.2%}",
                "Win Rate": "{:.2%}",
                "Skewness": "{:.2f}",
                "Kurtosis": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EXPECTED RETURNS
# ============================================================

def render_expected_returns(
    data,
    return_type,
):

    st.subheader(
        "Annualized Expected Returns"
    )

    returns_data, return_column = prepare_returns(
        data,
        return_type,
    )

    matrix = returns_data.pivot(
        index="Date",
        columns="Ticker",
        values=return_column,
    )

    matrix = matrix.dropna()

    if matrix.empty:

        st.warning(
            "Not enough overlapping observations "
            "to calculate expected returns."
        )

        return

    arithmetic = (
        calculate_historical_expected_return(
            matrix,
            annualization=TRADING_DAYS,
        )
    )

    geometric = (
        calculate_geometric_expected_return(
            matrix,
            annualization=TRADING_DAYS,
        )
    )

    result = pd.DataFrame(
        {
            "Asset": [
                _symbol_name(x)
                for x in arithmetic.index
            ],
            "Arithmetic Return":
                arithmetic.values,
            "Geometric Return":
                geometric.values,
        }
    )

    st.dataframe(
        result.style.format(
            {
                "Arithmetic Return":
                    "{:.2%}",
                "Geometric Return":
                    "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RETURNS SECTION
# ============================================================

def render_returns_section(data):

    st.header("Returns Analysis")

    return_type = st.radio(
        "Return Type",
        ["Simple", "Log"],
        horizontal=True,
        key="returns_analysis_type",
    )

    analysis = st.selectbox(
        "Analysis",
        [
            "Daily Returns",
            "Cumulative Returns",
            "Rolling Returns",
            "Return Distribution",
            "Return Statistics",
            "Expected Returns",
        ],
        key="returns_analysis_view",
    )

    st.divider()

    if analysis == "Daily Returns":

        render_daily_returns(
            data,
            return_type,
        )

    elif analysis == "Cumulative Returns":

        render_cumulative_returns(
            data,
            return_type,
        )

    elif analysis == "Rolling Returns":

        render_rolling_returns(data)

    elif analysis == "Return Distribution":

        render_return_distribution(
            data,
            return_type,
        )

    elif analysis == "Return Statistics":

        render_return_statistics(
            data,
            return_type,
        )

    elif analysis == "Expected Returns":

        render_expected_returns(
            data,
            return_type,
        )