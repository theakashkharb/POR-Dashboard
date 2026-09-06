from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.market import (
    calculate_annualized_volatility,
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
)


def build_sector_performance(
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    required_universe_columns = {"sector", "yf_ticker"}
    missing_columns = required_universe_columns - set(universe.columns)

    if missing_columns:
        raise ValueError(
            f"Universe is missing columns: {sorted(missing_columns)}"
        )

    period_returns = returns.loc[
        (returns.index >= pd.Timestamp(start_date))
        & (returns.index <= pd.Timestamp(end_date))
    ].copy()

    if period_returns.empty:
        raise ValueError(
            "No return data available for selected period."
        )

    mapping = universe[["sector", "yf_ticker"]].copy()

    mapping["yf_ticker"] = (
        mapping["yf_ticker"]
        .astype(str)
        .str.strip()
    )

    sector_data = {}

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

        daily_sector_returns = period_returns[
            available_tickers
        ].mean(
            axis=1,
            skipna=True,
        )

        daily_sector_returns = daily_sector_returns.dropna()

        if daily_sector_returns.empty:
            continue

        sector_data[sector] = {
            "Returns": (1.0 + daily_sector_returns).prod() - 1.0,
            "Volatility": calculate_annualized_volatility(
                daily_sector_returns
            ),
            "Sharpe": calculate_sharpe_ratio(
                daily_sector_returns
            ),
            "Max Drawdown": calculate_maximum_drawdown(
                daily_sector_returns
            ),
            "Stocks": len(available_tickers),
        }

    if not sector_data:
        raise ValueError(
            "No sector performance data could be created."
        )

    result = (
        pd.DataFrame.from_dict(
            sector_data,
            orient="index",
        )
        .reset_index()
        .rename(columns={"index": "Sector"})
    )

    result = (
        result
        .sort_values(
            "Returns",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def render_sector_performance(
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    start_date,
    end_date,
) -> None:
    st.subheader("Sector Performance")

    sector_data = build_sector_performance(
        returns=returns,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
    )

    display_data = sector_data.copy()

    display_data["Returns"] = display_data["Returns"].map(
        lambda value: f"{value:.2%}"
    )

    display_data["Volatility"] = display_data["Volatility"].map(
        lambda value: f"{value:.2%}"
    )

    display_data["Sharpe"] = display_data["Sharpe"].map(
        lambda value: f"{value:.2f}"
    )

    display_data["Max Drawdown"] = display_data[
        "Max Drawdown"
    ].map(
        lambda value: f"{value:.2%}"
    )

    st.dataframe(
        display_data[
            [
                "Sector",
                "Returns",
                "Volatility",
                "Sharpe",
                "Max Drawdown",
                "Stocks",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )