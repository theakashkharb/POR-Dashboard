from __future__ import annotations

import streamlit as st

from src.data.market_data import (
    load_market_data,
    load_universe,
)

from src.data.prepare import (
    prepare_market_data,
    prepare_universe,
)


def render_screening_page() -> None:
    st.title("Screening")

    universe = prepare_universe(
        load_universe()
    )

    market_data = prepare_market_data(
        load_market_data()
    )

    # =========================================================
    # UNIVERSE
    # =========================================================

    st.subheader("NIFTY 500 Universe")

    total_stocks = len(
        universe
    )

    market_tickers = set(
        market_data["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    data_available = int(
        universe["yf_ticker"]
        .isin(market_tickers)
        .sum()
    )

    data_missing = (
        total_stocks
        - data_available
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Universe",
        f"{total_stocks:,}",
    )

    col2.metric(
        "Data Available",
        f"{data_available:,}",
    )

    col3.metric(
        "Data Missing",
        f"{data_missing:,}",
    )

    st.caption(
        "Screening will apply data-quality, "
        "liquidity and investability rules."
    )

    # =========================================================
    # CURRENT ELIGIBLE UNIVERSE
    # =========================================================

    eligible_universe = universe[
        universe["yf_ticker"]
        .isin(market_tickers)
    ].copy()

    st.subheader(
        "Eligible Universe"
    )

    st.metric(
        "Stocks Eligible for Next Stage",
        f"{len(eligible_universe):,}",
    )

    display_data = (
        eligible_universe[
            [
                "symbol",
                "yf_ticker",
                "sector",
            ]
        ]
        .rename(
            columns={
                "symbol": "Stock",
                "yf_ticker": "Ticker",
                "sector": "Sector",
            }
        )
        .sort_values("Stock")
        .reset_index(drop=True)
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )