import streamlit as st

from dashboard.sidebar import render_sidebar
from dashboard.market_data import render_market_section
from dashboard.returns import render_returns_section
from dashboard.volatility import render_volatility_section
from dashboard.correlation import render_correlation_section
from dashboard.risk_return import render_risk_return_section


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="POR Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("Portfolio Optimization & Risk Analytics")

st.caption(
    "Market Data • Asset Analytics • Portfolio Construction "
    "• Risk • Stress Testing • Backtesting"
)


# ============================================================
# SIDEBAR
# ============================================================

render_sidebar()


# ============================================================
# DATA CHECK
# ============================================================

if "market_data" not in st.session_state:

    st.info(
        "Select a sector, stocks, and date range "
        "then click **Load Data**."
    )

    st.stop()


data = st.session_state["market_data"]

selected_universe = st.session_state["selected_universe"]
sector = st.session_state["loaded_sector"]
start_date = st.session_state["loaded_start"]
end_date = st.session_state["loaded_end"]


# ============================================================
# TOP NAVIGATION
# ============================================================

section = st.radio(
    "Dashboard Section",
    [
        "Market Data",
        "Returns",
        "Volatility",
        "Relationship",
        "Risk–Return",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_section",
)

st.divider()


# ============================================================
# SECTION ROUTER
# ============================================================

if section == "Market Data":

    render_market_section(
        data=data,
        selected_universe=selected_universe,
        sector=sector,
        start_date=start_date,
        end_date=end_date,
    )


elif section == "Returns":

    render_returns_section(data)


elif section == "Volatility":

    render_volatility_section(data)


elif section == "Relationship":

    render_correlation_section(data)


elif section == "Risk–Return":

    render_risk_return_section(data)