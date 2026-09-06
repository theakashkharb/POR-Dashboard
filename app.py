import streamlit as st

from dashboard.market.market_ui import render_market_page


st.set_page_config(
    page_title="POR",
    page_icon="📊",
    layout="wide",
)

render_market_page()