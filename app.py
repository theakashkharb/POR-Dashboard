from __future__ import annotations

import streamlit as st

from dashboard.sidebar import render_sidebar
from dashboard.market_data import render_market_section


st.set_page_config(
    page_title="POR Dashboard",
    page_icon="📊",
    layout="wide",
)


st.title(
    "Portfolio Optimization & Risk Analytics"
)

st.caption(
    "Market Research → Stock Selection → "
    "Portfolio Construction → Risk → Backtesting"
)


render_sidebar()


# ============================================================
# WAIT FOR DATA
# ============================================================

if "market_data" not in st.session_state:

    st.info(
        "Select a Sector, Index, or Custom Stocks "
        "from the sidebar, choose a date range, "
        "then click **Load Data**."
    )

    st.stop()


# ============================================================
# LOADED STATE
# ============================================================

data = st.session_state["market_data"]

selected_universe = st.session_state[
    "selected_universe"
]

selection_type = st.session_state[
    "loaded_selection_type"
]

sector = st.session_state.get(
    "loaded_sector"
)

index_name = st.session_state.get(
    "loaded_index"
)

start_date = st.session_state[
    "loaded_start"
]

end_date = st.session_state[
    "loaded_end"
]


# ============================================================
# TOP NAVIGATION
# ============================================================

st.divider()

page = st.radio(
    "Navigation",
    [
        "Market",
        "Stocks",
        "Portfolio",
        "Risk",
        "Backtest",
        "Performance",
        "Monitoring",
        "Reporting",
    ],
    horizontal=True,
    key="main_navigation",
)


# ============================================================
# MARKET
# ============================================================

if page == "Market":

    render_market_section(
        data=data,
        selected_universe=selected_universe,
        selection_type=selection_type,
        sector=sector,
        index_name=index_name,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================
# FUTURE STAGES
# ============================================================

elif page == "Stocks":

    st.header("Stock Analysis")

    st.info(
        "Stock analysis will be built after "
        "the Market stage is finalized."
    )


elif page == "Portfolio":

    st.header("Portfolio Construction")

    st.info(
        "Portfolio construction will be built "
        "after stock selection."
    )


elif page == "Risk":

    st.header("Risk Management")

    st.info(
        "Portfolio risk management will be built "
        "after portfolio construction."
    )


elif page == "Backtest":

    st.header("Backtesting")

    st.info(
        "Backtesting will be built after "
        "portfolio and risk stages."
    )


elif page == "Performance":

    st.header("Performance & Attribution")

    st.info(
        "Performance analysis will be built "
        "after backtesting."
    )


elif page == "Monitoring":

    st.header("Monitoring")

    st.info(
        "Portfolio monitoring will be built "
        "after performance analysis."
    )


elif page == "Reporting":

    st.header("Reporting & Final Decision")

    st.info(
        "Final reporting will be built "
        "at the end of the workflow."
    )