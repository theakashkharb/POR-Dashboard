import pandas as pd
import streamlit as st


# ============================================================
# SELECTED UNIVERSE
# ============================================================

def render_selected_universe(
    data,
    selected_universe,
    sector,
    start_date,
    end_date,
):

    st.subheader(
        "Selected Universe"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Sector",
            sector,
        )

    with col2:

        st.metric(
            "Stocks",
            len(selected_universe),
        )

    with col3:

        st.metric(
            "Start Date",
            str(start_date),
        )

    with col4:

        st.metric(
            "End Date",
            str(end_date),
        )


# ============================================================
# MARKET DATA
# ============================================================

def render_market_data(
    data,
    selected_universe,
):

    st.subheader(
        "Market Data"
    )

    market_data = data.merge(
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

    market_data["Date"] = pd.to_datetime(
        market_data["Date"]
    )

    market_data = market_data[
        [
            "symbol",
            "sector",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
        }
    )

    market_data = market_data.sort_values(
        ["Date", "Symbol"],
        ascending=[False, True],
    )

    st.dataframe(
        market_data,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STOCK DETAIL
# ============================================================

def render_stock_detail(
    data,
    selected_universe,
):

    st.subheader(
        "Stock Detail"
    )

    tickers = sorted(
        data["Ticker"].unique()
    )

    detail_ticker = st.selectbox(
        "Select Stock",
        tickers,
        key="detail_stock",
    )

    stock_data = (
        data[
            data["Ticker"]
            == detail_ticker
        ]
        .sort_values("Date")
        .copy()
    )

    stock_info = selected_universe[
        selected_universe["yf_ticker"]
        == detail_ticker
    ]

    if not stock_info.empty:

        stock_symbol = (
            stock_info.iloc[0]["symbol"]
        )

        stock_sector = (
            stock_info.iloc[0]["sector"]
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Symbol",
                stock_symbol,
            )

        with col2:

            st.metric(
                "Sector",
                stock_sector,
            )

    st.line_chart(
        stock_data.set_index("Date")[
            "Close"
        ],
        use_container_width=True,
    )


# ============================================================
# NORMALIZED PRICE
# ============================================================

def render_normalized_price(
    data,
):

    st.subheader(
        "Normalized Price"
    )

    st.caption(
        "All selected stocks are rebased to 100 "
        "at the beginning of the selected period."
    )

    prices = (
        data
        .pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )
        .sort_index()
    )

    prices = prices.dropna(
        how="all"
    )

    normalized_prices = (
        prices
        .apply(
            lambda x: (
                x / x.dropna().iloc[0]
            ) * 100
        )
    )

    st.line_chart(
        normalized_prices,
        use_container_width=True,
    )


# ============================================================
# COMPLETE MARKET DATA SECTION
# ============================================================

def render_market_section(
    data,
    selected_universe,
    sector,
    start_date,
    end_date,
):

    render_selected_universe(
        data,
        selected_universe,
        sector,
        start_date,
        end_date,
    )

    render_market_data(
        data,
        selected_universe,
    )

    render_stock_detail(
        data,
        selected_universe,
    )

    render_normalized_price(
        data,
    )