from __future__ import annotations

import streamlit as st

from dashboard.market.market_ui import render_market_page
from dashboard.stocks.stocks_ui import render_stocks_page


st.set_page_config(
    page_title="POR",
    layout="wide",
)

page = st.radio(
    "Navigate",
    [
        "Market",
        "Stocks",
    ],
    horizontal=True,
)

if page == "Market":
    render_market_page()

elif page == "Stocks":
    render_stocks_page()