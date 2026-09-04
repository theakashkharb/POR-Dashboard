from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.repository import load_market_data, load_universe


UNIVERSE_PATH = "data/raw/nifty500_universe.csv"

DATA_MIN_DATE = pd.Timestamp("2000-01-01").date()
DATA_MAX_DATE = pd.Timestamp.today().date()


# ============================================================
# CACHED UNIVERSE
# ============================================================

@st.cache_data(show_spinner=False)
def _load_universe() -> pd.DataFrame:

    universe = pd.read_csv(UNIVERSE_PATH)

    required_columns = {
        "sector",
        "symbol",
        "yf_ticker",
    }

    missing = required_columns - set(universe.columns)

    if missing:
        raise ValueError(
            f"Universe file is missing columns: {sorted(missing)}"
        )

    universe["sector"] = universe["sector"].astype(str)
    universe["symbol"] = universe["symbol"].astype(str)
    universe["yf_ticker"] = universe["yf_ticker"].astype(str)

    return universe


# ============================================================
# LOCAL MARKET DATA
# ============================================================

@st.cache_data(show_spinner=False)
def _load_local_market_data() -> pd.DataFrame:
    return load_market_data()


# ============================================================
# CLEAR LOADED DATA
# ============================================================

def _clear_loaded_data() -> None:

    for key in [
        "market_data",
        "selected_universe",
        "loaded_selection_type",
        "loaded_sector",
        "loaded_index",
        "loaded_start",
        "loaded_end",
    ]:
        st.session_state.pop(key, None)


# ============================================================
# RENDER SIDEBAR
# ============================================================

def render_sidebar() -> None:

    universe = _load_universe()

    st.sidebar.header("Data Selection")

    # ========================================================
    # UNIVERSE TYPE
    # ========================================================

    selection_type = st.sidebar.radio(
        "Universe",
        [
            "Sector",
            "Index",
            "Custom Stocks",
        ],
        key="universe_selection_type",
    )

    selected_sector = None
    selected_index = None

    selected_universe = pd.DataFrame()

    # ========================================================
    # SECTOR
    # ========================================================

    if selection_type == "Sector":

        sectors = sorted(
            universe["sector"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_sector = st.sidebar.selectbox(
            "Sector",
            sectors,
            key="selected_sector",
        )

        selected_universe = universe[
            universe["sector"] == selected_sector
        ].copy()

    # ========================================================
    # INDEX
    # ========================================================

    elif selection_type == "Index":

        selected_index = st.sidebar.selectbox(
            "Index",
            ["NIFTY 500"],
            key="selected_index",
        )

        selected_universe = universe.copy()

        st.sidebar.caption(
            "Current index universe is the NIFTY 500 dataset."
        )

    # ========================================================
    # CUSTOM STOCKS
    # ========================================================

    else:

        stock_options = (
            universe["symbol"]
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

        selected_symbols = st.sidebar.multiselect(
            "Stocks",
            options=stock_options,
            default=[],
            key="selected_custom_stocks",
            help="Select individual stocks.",
        )

        selected_universe = universe[
            universe["symbol"].isin(selected_symbols)
        ].copy()

    # ========================================================
    # DATE RANGE
    # ========================================================

    st.sidebar.subheader("Date Range")

    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.Timestamp("2020-01-01").date(),
        min_value=DATA_MIN_DATE,
        max_value=DATA_MAX_DATE,
        key="data_start_date",
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=DATA_MAX_DATE,
        min_value=DATA_MIN_DATE,
        max_value=DATA_MAX_DATE,
        key="data_end_date",
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if start_date > end_date:

        st.sidebar.error(
            "Start Date must be before End Date."
        )

    if selected_universe.empty:

        if selection_type == "Custom Stocks":

            st.sidebar.caption(
                "Select stocks to continue."
            )

    # ========================================================
    # LOAD BUTTON
    # ========================================================

    load_data = st.sidebar.button(
        "Load Data",
        type="primary",
        use_container_width=True,
        key="load_market_data",
    )

    if not load_data:
        return

    # ========================================================
    # VALIDATION BEFORE LOADING
    # ========================================================

    if start_date > end_date:

        st.sidebar.error(
            "Invalid date range."
        )

        return

    if selected_universe.empty:

        st.sidebar.error(
            "No securities selected."
        )

        return

    selected_tickers = (
        selected_universe["yf_ticker"]
        .dropna()
        .unique()
        .tolist()
    )

    # ========================================================
    # LOAD FROM LOCAL PARQUET
    # ========================================================

    with st.spinner(
        f"Loading {len(selected_tickers)} securities..."
    ):

        try:

            market_data = _load_local_market_data()

            if market_data.empty:

                st.sidebar.error(
                    "Local market dataset is empty."
                )

                return

            # ------------------------------------------------
            # FILTER DATE
            # ------------------------------------------------

            market_data = market_data[
                (market_data["Date"] >= pd.Timestamp(start_date))
                & (market_data["Date"] <= pd.Timestamp(end_date))
            ]

            # ------------------------------------------------
            # FILTER SELECTED TICKERS
            # ------------------------------------------------

            market_data = market_data[
                market_data["Ticker"].isin(selected_tickers)
            ]

            if market_data.empty:

                st.sidebar.error(
                    "No market data available for the selected "
                    "securities and date range."
                )

                return

            # ------------------------------------------------
            # SORT
            # ------------------------------------------------

            market_data = (
                market_data
                .sort_values(
                    ["Ticker", "Date"]
                )
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # STORE SESSION STATE
            # ------------------------------------------------

            st.session_state["market_data"] = market_data

            st.session_state[
                "selected_universe"
            ] = selected_universe.copy()

            st.session_state[
                "loaded_selection_type"
            ] = selection_type

            st.session_state[
                "loaded_sector"
            ] = selected_sector

            st.session_state[
                "loaded_index"
            ] = selected_index

            st.session_state[
                "loaded_start"
            ] = start_date

            st.session_state[
                "loaded_end"
            ] = end_date

            st.sidebar.success(
                f"Loaded {len(selected_tickers)} securities."
            )

        except Exception as exc:

            st.sidebar.error(
                f"Unable to load local market data: {exc}"
            )