"""
POR-Dashboard
Dashboard Charts
================

Reusable Plotly visualizations for the portfolio dashboard.

This module contains visualization logic only.
Portfolio calculations remain inside the analytical engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _validate_series(
    series: pd.Series,
    name: str = "series",
) -> pd.Series:
    """Validate and clean a pandas Series."""

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if series.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    if result.isna().any():
        raise ValueError(
            f"{name} contains non-numeric or missing values."
        )

    if not np.isfinite(
        result.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return result


def _validate_dataframe(
    dataframe: pd.DataFrame,
    name: str = "dataframe",
) -> pd.DataFrame:
    """Validate a DataFrame."""

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return dataframe.copy()


def _base_layout(
    title: str,
    height: int = 450,
) -> dict:
    """Return common Plotly layout settings."""

    return {
        "title": {
            "text": title,
            "x": 0.01,
        },
        "height": height,
        "margin": {
            "l": 50,
            "r": 30,
            "t": 60,
            "b": 50,
        },
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    }


# ============================================================
# NAV CHART
# ============================================================

def plot_nav(
    nav: pd.Series | pd.DataFrame,
    benchmark_nav: pd.Series | None = None,
    title: str = "Portfolio NAV",
) -> go.Figure:
    """
    Plot portfolio NAV and optional benchmark NAV.
    """

    if isinstance(nav, pd.DataFrame):

        if nav.shape[1] != 1:
            raise ValueError(
                "nav DataFrame must contain exactly one column."
            )

        nav = nav.iloc[:, 0]

    nav = _validate_series(
        nav,
        "nav",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav.values,
            mode="lines",
            name="Portfolio",
        )
    )

    if benchmark_nav is not None:

        benchmark_nav = _validate_series(
            benchmark_nav,
            "benchmark_nav",
        )

        benchmark_nav = benchmark_nav.reindex(
            nav.index
        ).dropna()

        if not benchmark_nav.empty:

            fig.add_trace(
                go.Scatter(
                    x=benchmark_nav.index,
                    y=benchmark_nav.values,
                    mode="lines",
                    name="Benchmark",
                )
            )

    fig.update_layout(
        **_base_layout(
            title,
            height=500,
        ),
        yaxis_title="NAV",
        xaxis_title="Date",
    )

    return fig


# ============================================================
# CUMULATIVE RETURN CHART
# ============================================================

def plot_cumulative_returns(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    title: str = "Cumulative Returns",
) -> go.Figure:
    """
    Plot cumulative returns for one or more portfolios.

    Parameters
    ----------
    returns : DataFrame
        Portfolio return series.
        Each column represents a portfolio.

    benchmark_returns : Series, optional
        Benchmark returns.
    """

    returns = _validate_dataframe(
        returns,
        "returns",
    )

    returns = returns.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if returns.isna().any().any():
        raise ValueError(
            "returns contains missing or non-numeric values."
        )

    cumulative = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    fig = go.Figure()

    for column in cumulative.columns:

        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[column] * 100,
                mode="lines",
                name=str(column),
            )
        )

    if benchmark_returns is not None:

        benchmark_returns = _validate_series(
            benchmark_returns,
            "benchmark_returns",
        )

        benchmark_cumulative = (
            (1.0 + benchmark_returns)
            .cumprod()
            - 1.0
        )

        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values * 100,
                mode="lines",
                name="Benchmark",
                line=dict(
                    dash="dash"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(
            title,
            height=500,
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
    )

    return fig


# ============================================================
# DRAWDOWN CHART
# ============================================================

def plot_drawdown(
    nav: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """
    Plot portfolio drawdown.
    """

    nav = _validate_series(
        nav,
        "nav",
    )

    running_max = nav.cummax()

    drawdown = (
        nav / running_max
    ) - 1.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=400,
        ),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )

    return fig


# ============================================================
# WEIGHT CHART
# ============================================================

def plot_weights(
    weights: pd.Series | dict,
    title: str = "Portfolio Allocation",
) -> go.Figure:
    """
    Plot portfolio weights.
    """

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    weights = _validate_series(
        pd.Series(weights),
        "weights",
    )

    weights = weights.sort_values(
        ascending=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=weights.values * 100,
            y=weights.index.astype(str),
            orientation="h",
            name="Weight",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                350,
                35 * len(weights),
            ),
        ),
        xaxis_title="Weight (%)",
        yaxis_title="Asset",
        showlegend=False,
    )

    return fig


# ============================================================
# RISK CONTRIBUTION CHART
# ============================================================

def plot_risk_contributions(
    risk_contributions: pd.Series | dict,
    title: str = "Risk Contribution",
) -> go.Figure:
    """
    Plot percentage risk contribution by asset.
    """

    if isinstance(
        risk_contributions,
        dict,
    ):
        risk_contributions = pd.Series(
            risk_contributions
        )

    risk_contributions = _validate_series(
        pd.Series(risk_contributions),
        "risk_contributions",
    )

    risk_contributions = (
        risk_contributions
        .sort_values(ascending=True)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=risk_contributions.values * 100,
            y=risk_contributions.index.astype(str),
            orientation="h",
            name="Risk Contribution",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                350,
                35 * len(risk_contributions),
            ),
        ),
        xaxis_title="Risk Contribution (%)",
        yaxis_title="Asset",
        showlegend=False,
    )

    return fig


# ============================================================
# RISK / RETURN SCATTER
# ============================================================

def plot_risk_return(
    performance: pd.DataFrame,
    return_column: str = "CAGR",
    risk_column: str = "Annualized Volatility",
    label_column: str | None = None,
    title: str = "Risk vs Return",
) -> go.Figure:
    """
    Plot portfolio risk versus return.

    Expected columns
    ----------------
    return_column
        Portfolio return metric.

    risk_column
        Portfolio risk metric.

    label_column
        Optional portfolio-name column.
    """

    performance = _validate_dataframe(
        performance,
        "performance",
    )

    if return_column not in performance.columns:
        raise ValueError(
            f"Missing column: {return_column}"
        )

    if risk_column not in performance.columns:
        raise ValueError(
            f"Missing column: {risk_column}"
        )

    if label_column is not None:

        if label_column not in performance.columns:
            raise ValueError(
                f"Missing column: {label_column}"
            )

        labels = performance[
            label_column
        ].astype(str)

    else:

        labels = performance.index.astype(str)

    x = pd.to_numeric(
        performance[risk_column],
        errors="coerce",
    )

    y = pd.to_numeric(
        performance[return_column],
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x[valid] * 100,
            y=y[valid] * 100,
            mode="markers+text",
            text=labels[valid],
            textposition="top center",
            name="Portfolio",
            marker={
                "size": 12,
            },
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=500,
        ),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Return (%)",
        showlegend=False,
    )

    return fig


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def plot_performance_comparison(
    performance: pd.DataFrame,
    metric: str = "CAGR",
    title: str | None = None,
) -> go.Figure:
    """
    Compare portfolio performance across methods.
    """

    performance = _validate_dataframe(
        performance,
        "performance",
    )

    if metric not in performance.columns:
        raise ValueError(
            f"Missing metric column: {metric}"
        )

    values = pd.to_numeric(
        performance[metric],
        errors="coerce",
    )

    valid = values.notna()

    labels = (
        performance.index.astype(str)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels[valid],
            y=values[valid] * 100,
            name=metric,
        )
    )

    if title is None:
        title = f"Portfolio {metric} Comparison"

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Portfolio",
        yaxis_title=f"{metric} (%)",
        showlegend=False,
    )

    return fig


# ============================================================
# TURNOVER CHART
# ============================================================

def plot_turnover(
    turnover: pd.Series | dict,
    title: str = "Average Turnover",
) -> go.Figure:
    """
    Plot average portfolio turnover.
    """

    if isinstance(
        turnover,
        dict,
    ):
        turnover = pd.Series(turnover)

    turnover = _validate_series(
        pd.Series(turnover),
        "turnover",
    )

    turnover = turnover.sort_values(
        ascending=False
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=turnover.index.astype(str),
            y=turnover.values * 100,
            name="Turnover",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Portfolio",
        yaxis_title="Average Turnover (%)",
        showlegend=False,
    )

    return fig


# ============================================================
# VOLATILITY CHART
# ============================================================

def plot_volatility(
    volatility: pd.Series | pd.DataFrame,
    title: str = "Portfolio Volatility",
) -> go.Figure:
    """
    Plot volatility over time.
    """

    if isinstance(
        volatility,
        pd.Series,
    ):
        volatility = volatility.to_frame(
            "Volatility"
        )

    volatility = _validate_dataframe(
        volatility,
        "volatility",
    )

    fig = go.Figure()

    for column in volatility.columns:

        series = _validate_series(
            volatility[column],
            str(column),
        )

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values * 100,
                mode="lines",
                name=str(column),
            )
        )

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
    )

    return fig


# ============================================================
# EWMA VOLATILITY CHART
# ============================================================

def plot_ewma_volatility(
    ewma_volatility: pd.Series,
    historical_volatility: float | None = None,
    title: str = "EWMA Volatility",
) -> go.Figure:
    """
    Plot EWMA volatility and optional historical volatility level.
    """

    ewma_volatility = _validate_series(
        ewma_volatility,
        "ewma_volatility",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ewma_volatility.index,
            y=ewma_volatility.values * 100,
            mode="lines",
            name="EWMA Volatility",
        )
    )

    if historical_volatility is not None:

        historical_volatility = float(
            historical_volatility
        )

        if not np.isfinite(
            historical_volatility
        ):
            raise ValueError(
                "historical_volatility must be finite."
            )

        fig.add_hline(
            y=historical_volatility * 100,
            line_dash="dash",
            annotation_text="Historical Volatility",
        )

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
    )

    return fig


# ============================================================
# BETA CHART
# ============================================================

def plot_asset_beta(
    betas: pd.Series | dict,
    title: str = "Asset Beta",
) -> go.Figure:
    """
    Plot beta for each asset.
    """

    if isinstance(
        betas,
        dict,
    ):
        betas = pd.Series(betas)

    betas = _validate_series(
        pd.Series(betas),
        "betas",
    )

    betas = betas.sort_values(
        ascending=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=betas.values,
            y=betas.index.astype(str),
            orientation="h",
            name="Beta",
        )
    )

    fig.add_vline(
        x=1.0,
        line_dash="dash",
        annotation_text="Beta = 1",
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                350,
                35 * len(betas),
            ),
        ),
        xaxis_title="Beta",
        yaxis_title="Asset",
        showlegend=False,
    )

    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """
    Plot asset correlation matrix.
    """

    correlation = _validate_dataframe(
        correlation,
        "correlation",
    )

    correlation = correlation.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if correlation.isna().any().any():
        raise ValueError(
            "correlation contains invalid values."
        )

    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(
            "correlation must be square."
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns.astype(str),
            y=correlation.index.astype(str),
            zmin=-1,
            zmax=1,
            text=np.round(
                correlation.values,
                2,
            ),
            texttemplate="%{text}",
            colorbar_title="Correlation",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                450,
                40 * len(correlation),
            ),
        ),
        xaxis_title="Asset",
        yaxis_title="Asset",
    )

    return fig


# ============================================================
# PORTFOLIO RISK CONTRIBUTION OVER TIME
# ============================================================

def plot_risk_contribution_history(
    risk_history: pd.DataFrame,
    title: str = "Risk Contribution Over Time",
) -> go.Figure:
    """
    Plot historical risk contribution by asset.
    """

    risk_history = _validate_dataframe(
        risk_history,
        "risk_history",
    )

    risk_history = risk_history.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if risk_history.isna().any().any():
        raise ValueError(
            "risk_history contains invalid values."
        )

    fig = go.Figure()

    for column in risk_history.columns:

        fig.add_trace(
            go.Scatter(
                x=risk_history.index,
                y=risk_history[column] * 100,
                mode="lines",
                stackgroup="one",
                name=str(column),
            )
        )

    fig.update_layout(
        **_base_layout(
            title,
            height=500,
        ),
        xaxis_title="Date",
        yaxis_title="Risk Contribution (%)",
    )

    return fig


# ============================================================
# STRESS TEST BAR CHART
# ============================================================

def plot_stress_losses(
    stress_losses: pd.Series | dict,
    title: str = "Stress-Test Loss by Asset",
) -> go.Figure:
    """
    Plot stress-test loss contribution by asset.
    """

    if isinstance(
        stress_losses,
        dict,
    ):
        stress_losses = pd.Series(
            stress_losses
        )

    stress_losses = _validate_series(
        pd.Series(stress_losses),
        "stress_losses",
    )

    stress_losses = stress_losses.sort_values(
        ascending=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=stress_losses.values * 100,
            y=stress_losses.index.astype(str),
            orientation="h",
            name="Stress Loss",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                350,
                35 * len(stress_losses),
            ),
        ),
        xaxis_title="Loss (%)",
        yaxis_title="Asset",
        showlegend=False,
    )

    return fig


# ============================================================
# STRESS SCENARIO COMPARISON
# ============================================================

def plot_stress_scenario_comparison(
    results: pd.DataFrame,
    metric: str = "Portfolio Loss",
    title: str = "Stress Scenario Comparison",
) -> go.Figure:
    """
    Compare stress-test results across scenarios.

    Expected format
    ---------------
    Index:
        Scenario names.

    Columns:
        Stress metrics.
    """

    results = _validate_dataframe(
        results,
        "results",
    )

    if metric not in results.columns:
        raise ValueError(
            f"Missing metric column: {metric}"
        )

    values = pd.to_numeric(
        results[metric],
        errors="coerce",
    )

    valid = values.notna()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=results.index.astype(str)[valid],
            y=values[valid] * 100,
            name=metric,
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Scenario",
        yaxis_title=f"{metric} (%)",
        showlegend=False,
    )

    return fig


# ============================================================
# NAV + DRAWDOWN COMBINED CHART
# ============================================================

def plot_nav_and_drawdown(
    nav: pd.Series,
    title: str = "NAV & Drawdown",
) -> go.Figure:
    """
    Plot NAV and drawdown using two y-axes.
    """

    nav = _validate_series(
        nav,
        "nav",
    )

    running_max = nav.cummax()

    drawdown = (
        nav / running_max
    ) - 1.0

    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav.values,
            mode="lines",
            name="NAV",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=500,
        )
    )

    fig.update_xaxes(
        title_text="Date"
    )

    fig.update_yaxes(
        title_text="NAV",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Drawdown (%)",
        secondary_y=True,
    )

    return fig


# ============================================================
# MONTHLY RETURNS HEATMAP
# ============================================================

def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns",
) -> go.Figure:
    """
    Create a calendar-style monthly return heatmap.

    Rows:
        Year

    Columns:
        Month
    """

    returns = _validate_series(
        returns,
        "returns",
    )

    if not isinstance(
        returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "returns index must be a DatetimeIndex."
        )

    monthly = (
        (1.0 + returns)
        .resample("ME")
        .prod()
        - 1.0
    )

    table = pd.DataFrame(
        {
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        }
    )

    pivot = table.pivot(
        index="Year",
        columns="Month",
        values="Return",
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    pivot = pivot.reindex(
        columns=range(1, 13)
    )

    pivot.columns = month_names

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns,
            y=pivot.index,
            text=np.round(
                pivot.values * 100,
                2,
            ),
            texttemplate="%{text}%",
            colorbar_title="Return (%)",
        )
    )

    fig.update_layout(
        **_base_layout(
            title,
            height=max(
                400,
                35 * len(pivot),
            ),
        ),
        xaxis_title="Month",
        yaxis_title="Year",
    )

    return fig


# ============================================================
# GENERIC TIME SERIES
# ============================================================

def plot_time_series(
    data: pd.Series | pd.DataFrame,
    title: str,
    yaxis_title: str = "Value",
) -> go.Figure:
    """
    Generic time-series chart.

    Useful when a dashboard metric does not need
    a specialized chart function.
    """

    if isinstance(
        data,
        pd.Series,
    ):

        data = data.to_frame(
            data.name or "Value"
        )

    data = _validate_dataframe(
        data,
        "data",
    )

    fig = go.Figure()

    for column in data.columns:

        series = _validate_series(
            data[column],
            str(column),
        )

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=str(column),
            )
        )

    fig.update_layout(
        **_base_layout(
            title,
            height=450,
        ),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )

    return fig