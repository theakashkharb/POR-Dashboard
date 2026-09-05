from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PREPARE MARKET MAP DATA
# ============================================================

def _prepare_market_map_data(
    stock_performance: pd.DataFrame,
) -> pd.DataFrame:

    if stock_performance.empty:
        return pd.DataFrame()

    required_columns = [
        "Symbol",
        "Sector",
        "Total Return",
        "Annualized Return",
        "Volatility",
        "Sharpe",
    ]

    missing = [
        column
        for column in required_columns
        if column not in stock_performance.columns
    ]

    if missing:
        raise ValueError(
            f"Market Map is missing required columns: {missing}"
        )

    data = stock_performance[
        required_columns
    ].copy()

    data = data.dropna(
        subset=[
            "Symbol",
            "Sector",
            "Total Return",
        ]
    )

    if data.empty:
        return pd.DataFrame()

    selected_parts = []

    # --------------------------------------------------------
    # TOP 4 + BOTTOM 4 FROM EACH SECTOR
    # --------------------------------------------------------

    for sector_name, group in data.groupby(
        "Sector",
        sort=True,
    ):

        group = group.copy()

        top = (
            group
            .sort_values(
                "Total Return",
                ascending=False,
            )
            .head(4)
        )

        bottom = (
            group
            .sort_values(
                "Total Return",
                ascending=True,
            )
            .head(4)
        )

        combined = pd.concat(
            [
                top,
                bottom,
            ],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=["Symbol"],
            keep="first",
        )

        combined["Sector"] = sector_name

        selected_parts.append(
            combined
        )

    if not selected_parts:
        return pd.DataFrame()

    stocks = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # BLOCK SIZE = ABSOLUTE TOTAL RETURN
    # --------------------------------------------------------

    stocks["Size"] = (
        stocks["Total Return"]
        .abs()
        .fillna(0)
        .clip(lower=0.0001)
    )

    # --------------------------------------------------------
    # SECTOR NODES
    # --------------------------------------------------------

    sector_rows = []

    for sector_name, group in stocks.groupby(
        "Sector",
        sort=True,
    ):

        sector_rows.append(
            {
                "NodeType": "Sector",
                "ID": f"Sector::{sector_name}",
                "Label": sector_name,
                "Parent": "Market",
                "Sector": sector_name,
                "Symbol": "",
                "Size": group["Size"].sum(),
                "Total Return": 0.0,
                "Annualized Return": 0.0,
                "Volatility": 0.0,
                "Sharpe": 0.0,
            }
        )

    sectors = pd.DataFrame(
        sector_rows
    )

    # --------------------------------------------------------
    # STOCK NODES
    # --------------------------------------------------------

    stocks["NodeType"] = "Stock"

    stocks["ID"] = (
        "Stock::"
        + stocks["Sector"].astype(str)
        + "::"
        + stocks["Symbol"].astype(str)
    )

    stocks["Label"] = (
        stocks["Symbol"].astype(str)
    )

    stocks["Parent"] = (
        "Sector::"
        + stocks["Sector"].astype(str)
    )

    # --------------------------------------------------------
    # MARKET ROOT
    # --------------------------------------------------------

    market_row = pd.DataFrame(
        [
            {
                "NodeType": "Market",
                "ID": "Market",
                "Label": "Market",
                "Parent": "",
                "Sector": "",
                "Symbol": "",
                "Size": sectors["Size"].sum(),
                "Total Return": 0.0,
                "Annualized Return": 0.0,
                "Volatility": 0.0,
                "Sharpe": 0.0,
            }
        ]
    )

    # --------------------------------------------------------
    # ALIGN COLUMNS
    # --------------------------------------------------------

    columns = [
        "NodeType",
        "ID",
        "Label",
        "Parent",
        "Sector",
        "Symbol",
        "Size",
        "Total Return",
        "Annualized Return",
        "Volatility",
        "Sharpe",
    ]

    stock_nodes = stocks[
        columns
    ]

    sector_nodes = sectors[
        columns
    ]

    result = pd.concat(
        [
            market_row,
            sector_nodes,
            stock_nodes,
        ],
        ignore_index=True,
    )

    return result


# ============================================================
# MARKET MAP FIGURE
# ============================================================

def _market_map_figure(
    map_data: pd.DataFrame,
) -> go.Figure | None:

    if map_data.empty:
        return None

    # --------------------------------------------------------
    # STOCKS USE REAL RETURNS
    # MARKET + SECTOR NODES STAY NEUTRAL
    # --------------------------------------------------------

    color_values = np.where(
        map_data["NodeType"].eq("Stock"),
        map_data["Total Return"],
        0.0,
    )

    customdata = np.column_stack(
        [
            map_data["Sector"].astype(str),
            map_data["Total Return"].astype(float),
            map_data["Annualized Return"].astype(float),
            map_data["Volatility"].astype(float),
            map_data["Sharpe"].astype(float),
        ]
    )

    # --------------------------------------------------------
    # PASTEL RETURN COLOR PROFILE
    #
    # Negative → pastel red
    # Zero     → soft neutral
    # Positive → pastel green
    # --------------------------------------------------------

    colorscale = [
        [0.00, "#e8a3a3"],
        [0.20, "#efb7b7"],
        [0.40, "#f5d0d0"],
        [0.50, "#f4f4f2"],
        [0.60, "#d9edd9"],
        [0.80, "#add6ae"],
        [1.00, "#78bd7d"],
    ]

    fig = go.Figure(
        go.Treemap(
            ids=map_data["ID"],
            labels=map_data["Label"],
            parents=map_data["Parent"],
            values=map_data["Size"],

            customdata=customdata,

            marker=dict(
                colors=color_values,

                colorscale=colorscale,

                cmid=0,

                line=dict(
                    color="white",
                    width=2,
                ),

                colorbar=dict(
                    title="Return",
                    tickformat=".0%",
                ),
            ),

            texttemplate="%{label}",

            hovertemplate=(
                "<b>%{label}</b><br>"
                "Sector: %{customdata[0]}<br>"
                "Total Return: %{customdata[1]:.2%}<br>"
                "Annualized Return: %{customdata[2]:.2%}<br>"
                "Volatility: %{customdata[3]:.2%}<br>"
                "Sharpe: %{customdata[4]:.2f}"
                "<extra></extra>"
            ),

            branchvalues="total",

            tiling=dict(
                pad=3,
            ),
        )
    )

    fig.update_layout(
        height=720,

        margin=dict(
            l=5,
            r=5,
            t=5,
            b=5,
        ),

        font=dict(
            size=12,
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# ============================================================
# RENDER MARKET MAP
# ============================================================

def _render_market_map(
    stock_performance: pd.DataFrame,
) -> None:

    st.subheader(
        "Market Map"
    )

    st.caption(
        "Market → Sector → Stock. "
        "Each sector shows its 4 best and 4 weakest performers. "
        "Larger blocks represent larger absolute returns. "
        "Soft green = positive return • "
        "Soft red = negative return."
    )

    map_data = _prepare_market_map_data(
        stock_performance
    )

    if map_data.empty:

        st.info(
            "No stock performance data available "
            "for the Market Map."
        )

        return

    fig = _market_map_figure(
        map_data
    )

    if fig is not None:

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )


# ============================================================
# PUBLIC COMPATIBILITY
# ============================================================

render_market_map = _render_market_map