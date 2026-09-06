from __future__ import annotations

from datetime import timedelta

import streamlit as st

from src.analytics.market import calculate_market_snapshot
from src.data.market_data import load_market_data, load_universe
from src.data.prepare import prepare_market_data, prepare_universe
from src.features.returns import calculate_simple_returns, create_price_matrix

from dashboard.market.market_map import render_market_map
from dashboard.market.sector_correlation import render_sector_correlation
from dashboard.market.sector_performance import render_sector_performance
from dashboard.market.snapshot import render_snapshot
from dashboard.market.stock_correlation import render_stock_correlation
from dashboard.market.top_stocks import render_top_stocks


PERIOD_OPTIONS = [
    "Last Week",
    "Last 2 Weeks",
    "Last 3 Weeks",
    "Last 1 Month",
    "Last 2 Months",
    "Last 3 Months",
    "Last 6 Months",
    "Last 1 Year",
    "Last 2 Years",
    "Last 3 Years",
    "Last 5 Years",
    "Last 10 Years",
    "All",
]


def get_period_start(period: str, data_start, data_end):
    if period == "All":
        return data_start

    if period == "Last Week":
        start = data_end - timedelta(days=7)
    elif period == "Last 2 Weeks":
        start = data_end - timedelta(days=14)
    elif period == "Last 3 Weeks":
        start = data_end - timedelta(days=21)
    elif period == "Last 1 Month":
        start = data_end - timedelta(days=30)
    elif period == "Last 2 Months":
        start = data_end - timedelta(days=60)
    elif period == "Last 3 Months":
        start = data_end - timedelta(days=90)
    elif period == "Last 6 Months":
        start = data_end - timedelta(days=180)
    elif period == "Last 1 Year":
        start = data_end - timedelta(days=365)
    elif period == "Last 2 Years":
        start = data_end - timedelta(days=730)
    elif period == "Last 3 Years":
        start = data_end - timedelta(days=1095)
    elif period == "Last 5 Years":
        start = data_end - timedelta(days=1825)
    elif period == "Last 10 Years":
        start = data_end - timedelta(days=3650)
    else:
        raise ValueError(f"Unsupported period: {period}")

    return max(start, data_start)


def render_market_page():
    st.title("Market")

    universe = prepare_universe(load_universe())
    market_data = prepare_market_data(load_market_data())

    prices = create_price_matrix(market_data)
    returns = calculate_simple_returns(prices)

    data_start = prices.index.min().date()
    data_end = prices.index.max().date()

    selected_period = st.selectbox(
        "Analysis Period",
        PERIOD_OPTIONS,
        index=7,
    )

    start_date = get_period_start(
        selected_period,
        data_start,
        data_end,
    )

    end_date = data_end

    snapshot = calculate_market_snapshot(
        returns=returns,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    # 1. MARKET SNAPSHOT
    render_snapshot(
        snapshot=snapshot,
        market_data=market_data,
        universe=universe,
        returns=returns,
        start_date=start_date,
        end_date=end_date,
    )

    # 2. MARKET MAP
    render_market_map(
        market_data=market_data,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
    )

    # 3. SECTOR PERFORMANCE
    render_sector_performance(
        returns=returns,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
    )

    # 4. SECTOR CORRELATION
    render_sector_correlation(
        returns=returns,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
    )

    # 5. TOP STOCK PERFORMANCE
    render_top_stocks(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )

    # 6. STOCK CORRELATION
    render_stock_correlation(
        market_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )