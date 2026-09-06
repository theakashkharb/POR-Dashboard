from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def build_sector_correlation(
    returns,
    universe,
    start_date,
    end_date,
):
    required_universe_columns = {
        "sector",
        "yf_ticker",
    }

    missing_columns = (
        required_universe_columns
        - set(universe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Universe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    period_returns = returns.loc[
        (returns.index >= pd.Timestamp(start_date))
        & (returns.index <= pd.Timestamp(end_date))
    ].copy()

    if period_returns.empty:
        raise ValueError(
            "No return data available for selected period."
        )

    mapping = universe[
        ["sector", "yf_ticker"]
    ].copy()

    mapping["yf_ticker"] = (
        mapping["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    sector_returns = {}

    for sector in mapping["sector"].dropna().unique():
        sector_tickers = mapping.loc[
            mapping["sector"] == sector,
            "yf_ticker",
        ].tolist()

        available_tickers = [
            ticker
            for ticker in sector_tickers
            if ticker in period_returns.columns
        ]

        if not available_tickers:
            continue

        sector_returns[sector] = (
            period_returns[available_tickers]
            .mean(axis=1, skipna=True)
        )

    if not sector_returns:
        raise ValueError(
            "No sector return data could be created."
        )

    sector_returns = pd.DataFrame(
        sector_returns
    )

    return sector_returns.corr()


def render_sector_correlation(
    returns,
    universe,
    start_date,
    end_date,
):
    st.subheader("Sector Correlation")

    correlation = build_sector_correlation(
        returns=returns,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
    )

    if correlation.empty:
        st.info(
            "No Sector Correlation data available "
            "for the selected period."
        )
        return

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale=[
            [0.0, "#d73027"],
            [0.5, "#f7f7f7"],
            [1.0, "#1a9850"],
        ],
    )

    fig.update_layout(
        margin=dict(
            t=10,
            l=10,
            r=10,
            b=10,
        ),
        height=700,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )