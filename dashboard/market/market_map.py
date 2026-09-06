from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics.stocks import build_market_map_data


def render_market_map(
    market_data,
    universe,
    start_date,
    end_date,
) -> None:
    """
    Render the Market Map.

    All quantitative calculations are handled
    by src.analytics.stocks.

    This file is responsible only for presentation.
    """

    st.subheader("Market Map")

    market_map_data = build_market_map_data(
        market_data=market_data,
        universe=universe,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    if market_map_data.empty:
        st.info(
            "No Market Map data available "
            "for the selected period."
        )
        return

    display_data = market_map_data.copy()

    display_data["Size"] = 1

    fig = px.treemap(
        display_data,
        path=[
            "sector",
            "symbol",
        ],
        values="Size",
        color="Returns",
        color_continuous_scale=[
            [0.0, "#d73027"],
            [0.5, "#f7f7f7"],
            [1.0, "#1a9850"],
        ],
        color_continuous_midpoint=0,
        hover_data={
            "Ticker": True,
            "Returns": ":.2%",
            "sector": False,
            "symbol": False,
            "Size": False,
        },
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

    fig.update_traces(
        textinfo="label",
        hovertemplate=(
            "<b>%{label}</b>"
            "<br>Return: %{customdata[1]:.2%}"
            "<extra></extra>"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )