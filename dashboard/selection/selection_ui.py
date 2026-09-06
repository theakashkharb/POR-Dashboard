from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.selection import (
    build_factor_dataset,
    calculate_multifactor_score,
    select_top_stocks,
)


SELECTION_METHODS = [
    "High Momentum",
    "Low Volatility",
    "Low Beta",
    "Risk-Adjusted",
    "Trend / Technical",
    "Volatility Regime",
    "Multi-Factor",
]


def render_stock_selection(
    market_data: pd.DataFrame,
    universe: pd.DataFrame,
    end_date,
) -> None:
    """
    Render the POR stock-selection interface.

    The screen ranks the available universe according
    to the selected quantitative selection method and
    returns a maximum of 25 stocks.
    """

    st.subheader("Stock Selection")

    # ---------------------------------------------------------
    # Selection controls
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Selection Method",
            SELECTION_METHODS,
        )

    with col2:
        number_of_stocks = st.number_input(
            "Number of Stocks",
            min_value=1,
            max_value=25,
            value=25,
            step=1,
        )

    # ---------------------------------------------------------
    # Build factor dataset
    # ---------------------------------------------------------

    with st.spinner(
        "Calculating selection factors..."
    ):
        factor_data = build_factor_dataset(
            market_data=market_data,
            universe=universe,
            end_date=end_date,
        )

    if factor_data.empty:
        st.warning(
            "No stocks are available for selection."
        )
        return

    # ---------------------------------------------------------
    # Selection logic
    # ---------------------------------------------------------

    if method == "High Momentum":

        score_column = (
            "12_1M_Momentum"
        )

        selected = select_top_stocks(
            factor_data=factor_data,
            score_column=score_column,
            number_of_stocks=number_of_stocks,
            ascending=False,
        )

        selected["Score"] = (
            selected[score_column]
        )

        score_label = "12–1M Momentum"

    elif method == "Low Volatility":

        score_column = "Volatility"

        selected = select_top_stocks(
            factor_data=factor_data,
            score_column=score_column,
            number_of_stocks=number_of_stocks,
            ascending=True,
        )

        selected["Score"] = (
            selected[score_column]
        )

        score_label = "Annualized Volatility"

    elif method == "Low Beta":

        score_column = "Beta"

        selected = select_top_stocks(
            factor_data=factor_data,
            score_column=score_column,
            number_of_stocks=number_of_stocks,
            ascending=True,
        )

        selected["Score"] = (
            selected[score_column]
        )

        score_label = "Beta"

    elif method == "Risk-Adjusted":

        score_column = (
            "Return_Volatility"
        )

        selected = select_top_stocks(
            factor_data=factor_data,
            score_column=score_column,
            number_of_stocks=number_of_stocks,
            ascending=False,
        )

        selected["Score"] = (
            selected[score_column]
        )

        score_label = "Return / Volatility"

    elif method == "Trend / Technical":

        technical_columns = [
            "Distance_52W_High",
            "MA_Trend_Consistency",
            "ADX",
            "OBV_Trend",
        ]

        technical_data = (
            factor_data[
                [
                    "symbol",
                    "yf_ticker",
                    "sector",
                ]
                + technical_columns
            ]
            .copy()
        )

        for column in technical_columns:
            technical_data[
                f"{column}_Z"
            ] = (
                technical_data[column]
                .rank(
                    pct=True
                )
            )

        technical_data[
            "Technical_Score"
        ] = (
            technical_data[
                "Distance_52W_High_Z"
            ]
            + technical_data[
                "MA_Trend_Consistency_Z"
            ]
            + technical_data[
                "ADX_Z"
            ]
            + technical_data[
                "OBV_Trend_Z"
            ]
        ) / 4.0

        selected = select_top_stocks(
            factor_data=technical_data,
            score_column="Technical_Score",
            number_of_stocks=number_of_stocks,
            ascending=False,
        )

        selected["Score"] = (
            selected["Technical_Score"]
        )

        score_label = "Technical Score"

    elif method == "Volatility Regime":

        score_column = (
            "Volatility_Regime"
        )

        selected = select_top_stocks(
            factor_data=factor_data,
            score_column=score_column,
            number_of_stocks=number_of_stocks,
            ascending=True,
        )

        selected["Score"] = (
            selected[score_column]
        )

        score_label = "Volatility Regime"

    elif method == "Multi-Factor":

        selected_data = (
            calculate_multifactor_score(
                factor_data=factor_data,
            )
        )

        selected = select_top_stocks(
            factor_data=selected_data,
            score_column="Multi_Factor_Score",
            number_of_stocks=number_of_stocks,
            ascending=False,
        )

        selected["Score"] = (
            selected[
                "Multi_Factor_Score"
            ]
        )

        score_label = "Multi-Factor Score"

    else:
        raise ValueError(
            f"Unsupported selection method: "
            f"{method}"
        )

    if selected.empty:
        st.warning(
            "No stocks could be selected "
            "using this method."
        )
        return

    # ---------------------------------------------------------
    # Selection summary
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Stocks Ranked",
        f"{len(factor_data):,}",
    )

    col2.metric(
        "Stocks Selected",
        f"{len(selected):,}",
    )

    col3.metric(
        "Selection Method",
        method,
    )

    st.caption(
        f"Selection date: "
        f"{pd.Timestamp(end_date).date()}"
    )

    # ---------------------------------------------------------
    # Selected stocks
    # ---------------------------------------------------------

    st.markdown(
        "### Selected Stocks"
    )

    display_data = selected[
        [
            "Selection Rank",
            "symbol",
            "yf_ticker",
            "sector",
            "Score",
        ]
    ].copy()

    display_data = display_data.rename(
        columns={
            "Selection Rank": "Rank",
            "symbol": "Stock",
            "yf_ticker": "Ticker",
            "sector": "Sector",
            "Score": score_label,
        }
    )

    # Format score according to factor type.

    if method in {
        "High Momentum",
    }:
        display_data[
            score_label
        ] = display_data[
            score_label
        ].map(
            lambda value: (
                f"{value:.2%}"
                if pd.notna(value)
                else "N/A"
            )
        )

    elif method == "Low Volatility":
        display_data[
            score_label
        ] = display_data[
            score_label
        ].map(
            lambda value: (
                f"{value:.2%}"
                if pd.notna(value)
                else "N/A"
            )
        )

    elif method == "Risk-Adjusted":
        display_data[
            score_label
        ] = display_data[
            score_label
        ].map(
            lambda value: (
                f"{value:.2f}"
                if pd.notna(value)
                else "N/A"
            )
        )

    elif method == "Volatility Regime":
        display_data[
            score_label
        ] = display_data[
            score_label
        ].map(
            lambda value: (
                f"{value:.2f}x"
                if pd.notna(value)
                else "N/A"
            )
        )

    else:
        display_data[
            score_label
        ] = display_data[
            score_label
        ].map(
            lambda value: (
                f"{value:.2f}"
                if pd.notna(value)
                else "N/A"
            )
        )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )