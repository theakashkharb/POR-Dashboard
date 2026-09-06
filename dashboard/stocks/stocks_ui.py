from __future__ import annotations

from datetime import timedelta

import streamlit as st

from src.data.market_data import load_market_data, load_universe
from src.data.prepare import prepare_market_data, prepare_universe

from dashboard.stocks.stocks_snapshot import render_stock_snapshot
from dashboard.stocks.stocks_performance import render_stock_performance
from dashboard.stocks.stocks_risk import render_stock_risk
from dashboard.stocks.stocks_relative import render_stock_relative
from dashboard.stocks.stocks_trading import render_stock_price_structure
from dashboard.stocks.stocks_liquidity import render_stock_liquidity
from dashboard.stocks.stocks_outperformers import render_stock_outperformers


PERIOD_OPTIONS = [
    "Last Week",
    "Last 2 Weeks",
    "Last 3 Weeks",
    "Last 1 Month",
    "Last 3 Months",
    "Last 6 Months",
    "Last 1 Year",
    "Last 2 Years",
    "Last 3 Years",
    "Last 5 Years",
    "Last 10 Years",
    "All",
]


def get_period_start(period, data_start, data_end):
    if period == "All":
        return data_start

    days = {
        "Last Week": 7,
        "Last 2 Weeks": 14,
        "Last 3 Weeks": 21,
        "Last 1 Month": 30,
        "Last 3 Months": 90,
        "Last 6 Months": 180,
        "Last 1 Year": 365,
        "Last 2 Years": 730,
        "Last 3 Years": 1095,
        "Last 5 Years": 1825,
        "Last 10 Years": 3650,
    }

    return max(
        data_end - timedelta(days=days[period]),
        data_start,
    )


def apply_stocks_style():
    st.markdown(
        """
        <style>

        /* Main section spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #F8F6F2;
            border: 1px solid #E8E3DB;
            border-radius: 14px;
            padding: 14px 16px;
            min-height: 88px;
        }

        div[data-testid="stMetricLabel"] {
            color: #77736C;
            font-size: 0.82rem;
            font-weight: 500;
        }

        div[data-testid="stMetricValue"] {
            color: #292824;
            font-size: 1.55rem;
            font-weight: 600;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            border: 1px solid #E5E1DA;
            border-radius: 14px;
            margin-bottom: 12px;
            overflow: hidden;
        }

        div[data-testid="stExpander"] details summary {
            background: #FAF9F7;
            padding: 14px 18px;
        }

        div[data-testid="stExpander"] details summary:hover {
            background: #F4F1EC;
        }

        /* Expander text */
        div[data-testid="stExpander"] details summary p {
            font-weight: 600;
            color: #35332F;
        }

        /* Captions */
        .stCaption {
            color: #8A867F;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stocks_page():
    apply_stocks_style()

    st.title("Stocks")

    universe = prepare_universe(load_universe())
    market_data = prepare_market_data(load_market_data())

    st.subheader("Stock Selection")

    stock_options = (
        universe[
            [
                "symbol",
                "yf_ticker",
                "sector",
            ]
        ]
        .drop_duplicates(subset=["yf_ticker"])
        .sort_values("symbol")
    )

    selected_symbol = st.selectbox(
        "Select Stock",
        stock_options["symbol"].tolist(),
    )

    selected_stock = stock_options[
        stock_options["symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_stock["yf_ticker"]

    st.caption(
        f"Sector: {selected_stock['sector']} | "
        f"Ticker: {selected_ticker}"
    )

    selected_data = market_data[
        market_data["Ticker"] == selected_ticker
    ].copy()

    if selected_data.empty:
        st.warning(
            "No market data available for the selected stock."
        )
        return

    data_start = selected_data["Date"].min().date()
    data_end = selected_data["Date"].max().date()

    selected_period = st.selectbox(
        "Analysis Period",
        PERIOD_OPTIONS,
        index=6,
    )

    start_date = get_period_start(
        selected_period,
        data_start,
        data_end,
    )

    end_date = data_end

    st.caption(
        f"Analysis period: {start_date} → {end_date}"
    )

    # ---------------------------------------------------------
    # NIFTY 50 Outperformers
    # ---------------------------------------------------------

    with st.expander(
        "NIFTY 50 Outperformers",
        expanded=True,
    ):
        render_stock_outperformers(
            market_data=market_data,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Stock Overview
    # ---------------------------------------------------------

    with st.expander(
        "Stock Overview",
        expanded=True,
    ):
        render_stock_snapshot(
            market_data=market_data,
            universe=universe,
            selected_ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Performance
    # ---------------------------------------------------------

    with st.expander(
        "Performance",
        expanded=True,
    ):
        render_stock_performance(
            market_data=market_data,
            selected_ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------

    with st.expander(
        "Risk",
        expanded=True,
    ):
        render_stock_risk(
            market_data=market_data,
            selected_ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Market Sensitivity
    # ---------------------------------------------------------

    with st.expander(
        "Market Sensitivity",
        expanded=False,
    ):
        render_stock_relative(
            market_data=market_data,
            ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Price Structure
    # ---------------------------------------------------------

    with st.expander(
        "Price Structure",
        expanded=False,
    ):
        render_stock_price_structure(
            market_data=market_data,
            ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )

    # ---------------------------------------------------------
    # Liquidity
    # ---------------------------------------------------------

    with st.expander(
        "Liquidity",
        expanded=False,
    ):
        render_stock_liquidity(
            market_data=market_data,
            ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )