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

def _build_stock_performance(
    data: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:

    if data.empty or universe.empty:
        return pd.DataFrame()

    prices = _get_price_matrix(data)

    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change(fill_method=None)

    mapping = (
        universe[
            ["yf_ticker", "symbol", "sector"]
        ]
        .dropna(subset=["yf_ticker"])
        .drop_duplicates("yf_ticker")
        .set_index("yf_ticker")
    )

    rows = []

    for ticker in returns.columns:

        stock_returns = returns[ticker].dropna()

        if stock_returns.empty:
            continue

        if ticker not in mapping.index:
            continue

        meta = mapping.loc[ticker]

        symbol = str(meta["symbol"])
        sector = str(meta["sector"])

        rows.append(
            {
                "Symbol": symbol,
                "Ticker": ticker,
                "Sector": sector,
                "Total Return": _total_return(stock_returns),
                "Annualized Return": _annualized_return(stock_returns),
                "Volatility": _annualized_volatility(stock_returns),
                "Sharpe": _sharpe_ratio(stock_returns),
                "Max Drawdown": _maximum_drawdown(stock_returns),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    return result.sort_values(
        "Annualized Return",
        ascending=False,
    ).reset_index(drop=True)


def _get_top_stocks(
    stock_performance: pd.DataFrame,
    selection_type: str,
    sector: str | None,
) -> pd.DataFrame:

    if stock_performance.empty:
        return pd.DataFrame()

    data = stock_performance.copy()

    # Sector selection:
    # show top 3 positive stocks in selected sector.
    if selection_type == "Sector" and sector:
        data = data[data["Sector"] == sector].copy()

        data = data[
            data["Total Return"] > 0
        ]

        return (
            data
            .sort_values(
                "Annualized Return",
                ascending=False,
            )
            .head(3)
        )

    # Index/custom:
    # top 3 positive-return stocks from each sector.
    data = data[
        data["Total Return"] > 0
    ].copy()

    if data.empty:
        return pd.DataFrame()

    result = (
        data
        .sort_values(
            ["Sector", "Annualized Return"],
            ascending=[True, False],
        )
        .groupby("Sector", group_keys=False)
        .head(3)
    )

    return result.reset_index(drop=True)


def _render_top_performing_stocks(
    stock_performance: pd.DataFrame,
    selection_type: str,
    sector: str | None,
) -> None:

    st.subheader("Top Performing Stocks")

    top_stocks = _get_top_stocks(
        stock_performance=stock_performance,
        selection_type=selection_type,
        sector=sector,
    )

    if top_stocks.empty:
        st.info(
            "No positive-return stocks are available "
            "for the selected period."
        )
        return

    display = top_stocks[
        [
            "Symbol",
            "Sector",
            "Total Return",
            "Annualized Return",
            "Volatility",
            "Sharpe",
        ]
    ].copy()

    if selection_type == "Sector" and sector:
        caption = (
            f"Top 3 positive-return stocks in {sector}, "
            "ranked by annualized return."
        )
    else:
        caption = (
            "Top 3 positive-return stocks from each sector, "
            "ranked by annualized return."
        )

    st.caption(caption)

    styled = _pastel_stock_style(display)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )


