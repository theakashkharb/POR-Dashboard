"""
POR-Dashboard
Dashboard Layout Components
===========================

Reusable Streamlit UI components used across the dashboard.

This module contains presentation/layout logic only.
Calculations remain inside the project's analytical engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def configure_page(
    title: str = "POR-Dashboard",
    page_icon: str = "📊",
) -> None:
    """
    Configure the Streamlit application page.
    """

    st.set_page_config(
        page_title=title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

def render_header(
    title: str = "POR-Dashboard",
    subtitle: str = (
        "Systematic Portfolio Optimization & Risk Analytics Platform"
    ),
) -> None:
    """
    Render the main application header.
    """

    st.title(title)

    st.caption(subtitle)

    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar_header() -> None:
    """Render sidebar branding."""

    st.sidebar.title("POR-Dashboard")

    st.sidebar.caption(
        "Portfolio Management & Risk Laboratory"
    )

    st.sidebar.divider()


def render_navigation(
    pages: list[str],
) -> str:
    """
    Render page navigation.

    Returns
    -------
    str
        Selected page.
    """

    return st.sidebar.radio(
        "Navigation",
        pages,
    )


# ============================================================
# PORTFOLIO LAB CONTROLS
# ============================================================

def render_asset_selector(
    available_assets: list[str],
    default_assets: list[str] | None = None,
) -> list[str]:
    """
    Render stock-selection control.
    """

    if not available_assets:
        st.warning("No assets available.")

        return []

    if default_assets is None:
        default_assets = available_assets[
            : min(5, len(available_assets))
        ]

    default_assets = [
        asset
        for asset in default_assets
        if asset in available_assets
    ]

    return st.multiselect(
        "Select Stocks",
        options=available_assets,
        default=default_assets,
        help=(
            "Choose the stocks that will form the portfolio universe."
        ),
    )


def render_portfolio_method_selector(
    methods: list[str],
) -> str:
    """
    Render portfolio-construction selector.
    """

    if not methods:
        raise ValueError(
            "No portfolio construction methods available."
        )

    return st.selectbox(
        "Portfolio Construction",
        options=methods,
        help=(
            "Select how portfolio weights should be constructed."
        ),
    )


def render_risk_method_selector(
    methods: list[str],
) -> str:
    """
    Render risk-management selector.
    """

    if not methods:
        raise ValueError(
            "No risk-management methods available."
        )

    return st.selectbox(
        "Risk Management",
        options=methods,
        help=(
            "Select how portfolio risk should be controlled."
        ),
    )


def render_stress_scenario_selector(
    scenarios: list[str],
) -> str:
    """
    Render stress-test scenario selector.
    """

    if not scenarios:
        raise ValueError(
            "No stress-test scenarios available."
        )

    return st.selectbox(
        "Stress Test",
        options=scenarios,
        help=(
            "Select the scenario used to stress the portfolio."
        ),
    )


# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

def render_date_controls(
    prices: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Render date-range controls based on available price data.
    """

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            "prices index must be a DatetimeIndex"
        )

    minimum_date = prices.index.min().date()
    maximum_date = prices.index.max().date()

    start_date, end_date = st.date_input(
        "Analysis Period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date > end_date:
        raise ValueError(
            "Start date must be before end date."
        )

    return start_date, end_date


def render_backtest_controls() -> dict:
    """
    Render backtesting parameters.
    """

    st.subheader("Backtest Settings")

    col1, col2, col3 = st.columns(3)

    with col1:

        train_window = st.number_input(
            "Training Window",
            min_value=20,
            max_value=2000,
            value=252,
            step=1,
        )

    with col2:

        rebalance_frequency = st.selectbox(
            "Rebalance Frequency",
            options=[
                "D",
                "W",
                "M",
                "Q",
                "Y",
            ],
            index=2,
        )

    with col3:

        max_turnover = st.number_input(
            "Maximum Turnover",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

    return {
        "train_window": int(train_window),
        "rebalance_frequency": rebalance_frequency,
        "max_turnover": float(max_turnover),
    }


def render_capital_control() -> float:
    """
    Render initial capital input.
    """

    return float(
        st.number_input(
            "Initial Capital",
            min_value=1.0,
            value=1_000_000.0,
            step=100_000.0,
        )
    )


# ============================================================
# STRESS PARAMETERS
# ============================================================

def render_stress_parameters(
    scenario: str,
    available_assets: list[str],
    sectors: list[str] | None = None,
) -> dict:
    """
    Render scenario-specific stress parameters.
    """

    parameters: dict = {}

    # --------------------------------------------------------
    # Market Crash
    # --------------------------------------------------------

    if scenario == "Market Crash":

        crash = st.slider(
            "Market Shock",
            min_value=-1.0,
            max_value=0.0,
            value=-0.20,
            step=0.01,
            format="%.0f%%",
        )

        parameters["crash_return"] = float(crash)

    # --------------------------------------------------------
    # Severe Crash
    # --------------------------------------------------------

    elif scenario == "Severe Crash":

        crash = st.slider(
            "Severe Market Shock",
            min_value=-1.0,
            max_value=0.0,
            value=-0.40,
            step=0.01,
            format="%.0f%%",
        )

        parameters["crash_return"] = float(crash)

    # --------------------------------------------------------
    # Volatility Spike
    # --------------------------------------------------------

    elif scenario == "Volatility Spike":

        multiplier = st.slider(
            "Volatility Multiplier",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
        )

        parameters["multiplier"] = float(
            multiplier
        )

    # --------------------------------------------------------
    # Interest Rate Shock
    # --------------------------------------------------------

    elif scenario == "Interest Rate Shock":

        shock = st.slider(
            "Asset Return Shock",
            min_value=-1.0,
            max_value=0.0,
            value=-0.05,
            step=0.01,
            format="%.0f%%",
        )

        assets = st.multiselect(
            "Rate-Sensitive Assets",
            options=available_assets,
            default=available_assets,
        )

        parameters["shock"] = float(shock)

        parameters[
            "rate_sensitive_assets"
        ] = assets

    # --------------------------------------------------------
    # Sector Shock
    # --------------------------------------------------------

    elif scenario == "Sector Shock":

        if not sectors:
            st.warning(
                "Sector data is required for sector stress testing."
            )

            return parameters

        sector = st.selectbox(
            "Sector",
            options=sectors,
        )

        shock = st.slider(
            "Sector Shock",
            min_value=-1.0,
            max_value=0.0,
            value=-0.25,
            step=0.01,
            format="%.0f%%",
        )

        parameters["sector"] = sector
        parameters["shock"] = float(shock)

    # --------------------------------------------------------
    # Single Stock Crash
    # --------------------------------------------------------

    elif scenario == "Single Stock Crash":

        if not available_assets:
            return parameters

        ticker = st.selectbox(
            "Stock",
            options=available_assets,
        )

        crash = st.slider(
            "Stock Crash",
            min_value=-1.0,
            max_value=0.0,
            value=-0.50,
            step=0.01,
            format="%.0f%%",
        )

        parameters["ticker"] = ticker

        parameters[
            "crash_return"
        ] = float(crash)

    # --------------------------------------------------------
    # Correlation Spike
    # --------------------------------------------------------

    elif scenario == "Correlation Spike":

        correlation = st.slider(
            "Stressed Correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
        )

        parameters[
            "correlation_target"
        ] = float(correlation)

    # --------------------------------------------------------
    # Liquidity / Exposure Shock
    # --------------------------------------------------------

    elif scenario == "Liquidity / Exposure Shock":

        scaling = st.slider(
            "Effective Exposure",
            min_value=0.01,
            max_value=1.0,
            value=0.50,
            step=0.01,
            format="%.0f%%",
        )

        parameters[
            "liquidity_scaling"
        ] = float(scaling)

    # --------------------------------------------------------
    # Historical Crisis
    # --------------------------------------------------------

    elif scenario == "Historical Crisis":

        col1, col2 = st.columns(2)

        with col1:

            start = st.date_input(
                "Crisis Start",
            )

        with col2:

            end = st.date_input(
                "Crisis End",
            )

        parameters[
            "crisis_start"
        ] = str(start)

        parameters[
            "crisis_end"
        ] = str(end)

    # --------------------------------------------------------
    # Custom Shock
    # --------------------------------------------------------

    elif scenario == "Custom Shock":

        st.caption(
            "Enter an individual return shock for each selected asset."
        )

        shocks = {}

        for asset in available_assets:

            shocks[asset] = st.number_input(
                f"{asset} Shock",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
            )

        parameters["shocks"] = shocks

    return parameters


# ============================================================
# RUN BUTTON
# ============================================================

def render_run_button(
    label: str = "🚀 Run Analysis",
) -> bool:
    """
    Render the primary analysis button.
    """

    return st.button(
        label,
        type="primary",
        use_container_width=True,
    )


# ============================================================
# KPI CARDS
# ============================================================

def render_metric_cards(
    metrics: dict,
    columns: list[str],
    formats: dict[str, str] | None = None,
) -> None:
    """
    Render metrics as Streamlit KPI cards.

    Parameters
    ----------
    metrics : dict
        Metric name -> value.

    columns : list[str]
        Metrics to display.

    formats : dict, optional
        Metric name -> Python format string.
    """

    if formats is None:
        formats = {}

    metric_columns = st.columns(
        len(columns)
    )

    for column, metric_name in zip(
        metric_columns,
        columns,
    ):

        value = metrics.get(
            metric_name
        )

        if value is None:
            display_value = "—"

        elif isinstance(value, (float, int)):

            fmt = formats.get(
                metric_name,
                "{:.2f}",
            )

            try:
                display_value = fmt.format(
                    value
                )

            except (ValueError, IndexError):
                display_value = str(value)

        else:

            display_value = str(value)

        column.metric(
            label=metric_name,
            value=display_value,
        )


# ============================================================
# WEIGHTS TABLE
# ============================================================

def render_weights_table(
    weights: pd.Series | dict,
    title: str = "Portfolio Weights",
) -> None:
    """
    Render portfolio weights table.
    """

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    weights = pd.Series(
        weights,
        dtype=float,
    )

    table = pd.DataFrame(
        {
            "Asset": weights.index,
            "Weight": weights.values,
        }
    )

    table["Weight"] = (
        table["Weight"]
        .astype(float)
    )

    table["Weight %"] = (
        table["Weight"] * 100
    ).round(2)

    table = table.sort_values(
        "Weight",
        ascending=False,
    )

    st.subheader(title)

    st.dataframe(
        table[
            [
                "Asset",
                "Weight %",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GENERIC DATAFRAME
# ============================================================

def render_dataframe(
    dataframe: pd.DataFrame,
    title: str | None = None,
    height: int | None = None,
) -> None:
    """
    Render a DataFrame safely.
    """

    if title:
        st.subheader(title)

    if dataframe is None:
        st.info("No data available.")

        return

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        dataframe = pd.DataFrame(
            dataframe
        )

    kwargs = {
        "use_container_width": True,
        "hide_index": True,
    }

    if height is not None:
        kwargs["height"] = height

    st.dataframe(
        dataframe,
        **kwargs,
    )


# ============================================================
# EXPOSURE TABLE
# ============================================================

def render_exposure_table(
    exposure: pd.Series | dict,
    title: str = "Exposure",
) -> None:
    """
    Render industry/sector exposure.
    """

    if isinstance(exposure, dict):
        exposure = pd.Series(exposure)

    exposure = pd.Series(
        exposure,
        dtype=float,
    )

    table = (
        exposure
        .rename("Exposure")
        .to_frame()
    )

    table["Exposure %"] = (
        table["Exposure"] * 100
    ).round(2)

    table = table.sort_values(
        "Exposure",
        ascending=False,
    )

    st.subheader(title)

    st.dataframe(
        table[
            ["Exposure %"]
        ],
        use_container_width=True,
    )


# ============================================================
# RISK CONTRIBUTION TABLE
# ============================================================

def render_risk_contribution_table(
    risk_table: pd.DataFrame,
) -> None:
    """
    Render portfolio risk contribution table.
    """

    st.subheader(
        "Risk Contribution"
    )

    if risk_table is None:
        st.info(
            "Risk contribution data unavailable."
        )

        return

    table = risk_table.copy()

    if "risk_contribution" in table.columns:

        table[
            "risk_contribution"
        ] = (
            table[
                "risk_contribution"
            ] * 100
        )

    st.dataframe(
        table,
        use_container_width=True,
    )


# ============================================================
# STRESS SUMMARY
# ============================================================

def render_stress_summary(
    stress_result: dict,
) -> None:
    """
    Render headline stress-test metrics.
    """

    if not stress_result:
        st.info(
            "No stress-test results available."
        )

        return

    st.subheader(
        stress_result.get(
            "scenario",
            "Stress Test",
        )
    )

    portfolio_return = stress_result.get(
        "portfolio_return"
    )

    portfolio_loss = stress_result.get(
        "portfolio_loss"
    )

    stressed_nav = stress_result.get(
        "stressed_nav"
    )

    metrics = {}

    if portfolio_return is not None:
        metrics["Portfolio Return"] = (
            float(portfolio_return) * 100
        )

    if portfolio_loss is not None:
        metrics["Portfolio Loss"] = (
            float(portfolio_loss) * 100
        )

    if stressed_nav is not None:
        metrics["Stressed NAV"] = (
            float(stressed_nav)
        )

    if metrics:

        cols = st.columns(
            len(metrics)
        )

        for col, (name, value) in zip(
            cols,
            metrics.items(),
        ):

            if name != "Stressed NAV":

                col.metric(
                    name,
                    f"{value:.2f}%",
                )

            else:

                col.metric(
                    name,
                    f"{value:,.2f}",
                )


# ============================================================
# SECTION HEADER
# ============================================================

def render_section_header(
    title: str,
    description: str | None = None,
) -> None:
    """
    Render a consistent section heading.
    """

    st.header(title)

    if description:

        st.caption(
            description
        )


# ============================================================
# EMPTY STATE
# ============================================================

def render_empty_state(
    title: str,
    message: str,
) -> None:
    """
    Render an empty-state message.
    """

    st.info(
        f"**{title}**\n\n{message}"
    )


# ============================================================
# ERROR DISPLAY
# ============================================================

def render_analysis_error(
    error: Exception,
) -> None:
    """
    Display an analysis error in a user-friendly way.
    """

    st.error(
        "Analysis could not be completed."
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            str(error)
        )


# ============================================================
# SUCCESS MESSAGE
# ============================================================

def render_analysis_success() -> None:
    """Display successful analysis message."""

    st.success(
        "Analysis completed successfully."
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer() -> None:
    """
    Render application footer.
    """

    st.divider()

    st.caption(
        "POR-Dashboard • Systematic Portfolio Optimization "
        "& Risk Analytics Platform"
    )