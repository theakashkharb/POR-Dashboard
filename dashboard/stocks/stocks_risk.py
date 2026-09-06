from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.stocks.data import _stock_return_series
from src.analytics.stocks.risk import (
    calculate_stock_drawdown,
    calculate_stock_risk_metrics,
)


def build_risk_data(
    market_data: pd.DataFrame,
    selected_ticker: str,
    start_date: str,
    end_date: str,
):
    returns = _stock_return_series(
        market_data=market_data,
        ticker=selected_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    if returns.empty:
        raise ValueError(
            "Not enough data to calculate stock risk."
        )

    drawdown = calculate_stock_drawdown(returns)

    risk_summary = calculate_stock_risk_metrics(
        market_data=market_data,
        ticker=selected_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    drawdown_data = pd.DataFrame(
        {
            "Date": returns.index,
            "Drawdown": drawdown.values,
        }
    )

    return drawdown_data, risk_summary


def _format_metric(value, decimals=2):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        return "N/A"

    return f"{value:.{decimals}f}"


def render_stock_risk(
    market_data: pd.DataFrame,
    selected_ticker: str,
    start_date: str,
    end_date: str,
) -> None:

    st.subheader("Risk")

    data, risk = build_risk_data(
        market_data=market_data,
        selected_ticker=selected_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Annualized Volatility",
        f"{risk['Annualized Volatility']:.2%}",
    )

    col2.metric(
        "Sharpe Ratio",
        _format_metric(risk["Sharpe"]),
    )

    col3.metric(
        "Sortino Ratio",
        _format_metric(risk["Sortino"]),
    )

    col4.metric(
        "CVaR 95% (Expected Shortfall)",
        f"{risk['CVaR 95%']:.2%}",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Maximum Drawdown",
        f"{risk['Max Drawdown']:.2%}",
    )

    recovery_duration = risk["Recovery Duration"]

    if recovery_duration is None:
        recovery_text = "Not Recovered"
    else:
        recovery_text = (
            f"{recovery_duration} trading days"
        )

    col2.metric(
        "Recovery Duration",
        recovery_text,
    )

    col3.metric(
        "Calmar Ratio",
        _format_metric(risk["Calmar"]),
    )

    st.markdown("**Drawdown History**")

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            line=dict(
                color="#C97B84",
                width=2.5,
            ),
            fill="tozeroy",
            fillcolor="rgba(201, 123, 132, 0.16)",
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b>"
                "<br>Drawdown: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_color="#B9B4AC",
    )

    figure.update_layout(
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
        plot_bgcolor="#FCFBF9",
        paper_bgcolor="#FCFBF9",
        hovermode="x",
        showlegend=False,
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Drawdown (%)",
            showgrid=True,
            gridcolor="#ECE8E1",
            zeroline=False,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )