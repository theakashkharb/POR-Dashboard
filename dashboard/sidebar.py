import pandas as pd
import streamlit as st

from src.data.manager import get_data


UNIVERSE_PATH = "data/raw/nifty500_universe.csv"

DATA_MIN_DATE = pd.Timestamp(
    "1996-01-01"
).date()

DATA_MAX_DATE = pd.Timestamp.today().date()


# ============================================================
# LOAD UNIVERSE
# ============================================================

@st.cache_data
def load_universe():

    universe = pd.read_csv(
        UNIVERSE_PATH
    )

    universe["sector"] = (
        universe["sector"].astype(str)
    )

    universe["symbol"] = (
        universe["symbol"].astype(str)
    )

    universe["yf_ticker"] = (
        universe["yf_ticker"].astype(str)
    )

    return universe


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    universe = load_universe()

    sectors = sorted(
        universe["sector"]
        .dropna()
        .unique()
        .tolist()
    )

    st.sidebar.header(
        "Data Selection"
    )

    selected_sector = st.sidebar.selectbox(
        "Sector",
        ["All Sectors"] + sectors,
    )

    # --------------------------------------------------------
    # Filter universe by sector
    # --------------------------------------------------------

    if selected_sector == "All Sectors":

        filtered_universe = (
            universe.copy()
        )

    else:

        filtered_universe = universe[
            universe["sector"]
            == selected_sector
        ].copy()

    # --------------------------------------------------------
    # Stock selection
    # --------------------------------------------------------

    stock_options = (
        filtered_universe["symbol"]
        .sort_values()
        .tolist()
    )

    selected_symbols = st.sidebar.multiselect(
        "Stocks",
        options=stock_options,
        default=stock_options[:5],
    )

    selected_universe = (
        filtered_universe[
            filtered_universe["symbol"]
            .isin(selected_symbols)
        ]
        .copy()
    )

    selected_tickers = (
        selected_universe[
            "yf_ticker"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Date Range"
    )

    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.Timestamp(
            "2020-01-01"
        ).date(),
        min_value=DATA_MIN_DATE,
        max_value=DATA_MAX_DATE,
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=DATA_MAX_DATE,
        min_value=DATA_MIN_DATE,
        max_value=DATA_MAX_DATE,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if start_date > end_date:

        st.sidebar.error(
            "Start Date must be before End Date."
        )

    if not selected_tickers:

        st.sidebar.warning(
            "Select at least one stock."
        )

    # --------------------------------------------------------
    # Load button
    # --------------------------------------------------------

    load_data = st.sidebar.button(
        "Load Data",
        type="primary",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------

    if load_data:

        if start_date > end_date:

            st.error(
                "Start Date must be before End Date."
            )

            st.stop()

        if not selected_tickers:

            st.error(
                "Please select at least one stock."
            )

            st.stop()

        with st.spinner(
            "Downloading and processing market data..."
        ):

            try:

                data = get_data(
                    tickers=selected_tickers,
                    start=start_date,
                    end=end_date,
                )

                if data.empty:

                    st.error(
                        "No market data available."
                    )

                    st.stop()

                data["Date"] = pd.to_datetime(
                    data["Date"]
                )

                data = data.sort_values(
                    ["Ticker", "Date"]
                )

                st.session_state[
                    "market_data"
                ] = data

                st.session_state[
                    "selected_universe"
                ] = selected_universe.copy()

                st.session_state[
                    "loaded_sector"
                ] = selected_sector

                st.session_state[
                    "loaded_start"
                ] = start_date

                st.session_state[
                    "loaded_end"
                ] = end_date

            except Exception as e:

                st.error(
                    f"Unable to load market data: {e}"
                )

                st.stop()

    return universe