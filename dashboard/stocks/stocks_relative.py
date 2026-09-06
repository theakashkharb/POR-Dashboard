from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.stocks.relative import (
    calculate_benchmark_correlation,
    calculate_rolling_beta,
    calculate_up_down_capture,
)
from src.analytics.stocks.data import _stock_return_series
from src.data.market_data import load_nifty50_data


ROLLING_BETA_DAYS = 126


@st.cache_data
def get_nifty50_data() -> pd.DataFrame:
    """Load the saved NIFTY 50 historical market data."""
    return load_nifty50_data()


def _build_market_sensitivity(
    market_data,
    ticker: str,
    start_date,
    end_date,
) -> tuple[pd.Series, pd.Series]:
    """Build aligned stock and NIFTY 50 daily return series."""

    stock_returns = _stock_return_series(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    benchmark_data = get_nifty50_data()

    if benchmark_data.empty:
        return stock_returns, pd.Series(dtype=float)

    benchmark_data = benchmark_data.copy()
    benchmark_data["Date"] = pd.to_datetime(
        benchmark_data["Date"]
    )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    benchmark_data = benchmark_data[
        (benchmark_data["Date"] >= start)
        & (benchmark_data["Date"] <= end)
    ]

    if benchmark_data.empty:
        return stock_returns, pd.Series(dtype=float)

    benchmark_prices = (
        benchmark_data
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")["Close"]
    )

    benchmark_returns = benchmark_prices.pct_change().dropna()

    stock_returns.index = pd.to_datetime(stock_returns.index)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

    stock_returns, benchmark_returns = stock_returns.align(
        benchmark_returns,
        join="inner",
    )

    valid = (
        stock_returns.notna()
        & benchmark_returns.notna()
    )

    return (
        stock_returns.loc[valid],
        benchmark_returns.loc[valid],
    )


def _format_value(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value:.{decimals}f}"


def _format_percent(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def render_stock_relative(
    market_data,
    ticker: str,
    start_date,
    end_date,
) -> None:
    """Render Market Sensitivity for the selected stock."""

    st.subheader("Market Sensitivity")

    stock_returns, benchmark_returns = _build_market_sensitivity(
        market_data=market_data,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    if stock_returns.empty or benchmark_returns.empty:
        st.warning(
            "NIFTY 50 benchmark data is unavailable for the selected period."
        )
        return

    if len(stock_returns) < 2:
        st.warning(
            "Not enough overlapping stock and NIFTY 50 data "
            "to calculate market sensitivity."
        )
        return

    rolling_beta = calculate_rolling_beta(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        window=ROLLING_BETA_DAYS,
    )

    correlation = calculate_benchmark_correlation(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
    )

    capture = calculate_up_down_capture(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
    )

    rolling_beta_valid = rolling_beta.dropna()

    latest_beta = (
        rolling_beta_valid.iloc[-1]
        if not rolling_beta_valid.empty
        else float("nan")
    )

    row = st.columns(4)

    row[0].metric(
        "Rolling 6M Beta",
        _format_value(latest_beta),
    )

    row[1].metric(
        "NIFTY 50 Correlation",
        _format_value(correlation),
    )

    row[2].metric(
        "Up Capture",
        _format_percent(capture.get("Up Capture")),
    )

    row[3].metric(
        "Down Capture",
        _format_percent(capture.get("Down Capture")),
    )

    st.caption(
        "Benchmark: NIFTY 50 (^NSEI) | "
        "Rolling beta window: 126 trading days"
    )