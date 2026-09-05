from __future__ import annotations

from __future__ import annotations

import numpy as np

import pandas as pd

import streamlit as st

import plotly.graph_objects as go


from .core import *
from .core import (
    _safe_float,
    _format_percent,
    _format_sharpe,
    _normalize_dates,
    _analysis_window,
    _get_price_matrix,
    _get_returns,
    _annualized_return,
    _total_return,
    _annualized_volatility,
    _sharpe_ratio,
    _maximum_drawdown,
    _calculate_sector_returns,
    _market_metrics,
    _pastel_sector_style,
    _pastel_stock_style,
    _correlation_relationships,
    TRADING_DAYS,
    PERIOD_OPTIONS,
)

# ============================================================
# SECTION
# ============================================================

def _build_sector_performance(
    sector_returns: pd.DataFrame,
) -> pd.DataFrame:

    if sector_returns.empty:
        return pd.DataFrame()

    rows = []

    for sector in sector_returns.columns:

        returns = sector_returns[sector].dropna()

        if returns.empty:
            continue

        rows.append(
            {
                "Sector": sector,
                "Total Return": _total_return(returns),
                "Annualized Return": _annualized_return(returns),
                "Volatility": _annualized_volatility(returns),
                "Sharpe": _sharpe_ratio(returns),
                "Max Drawdown": _maximum_drawdown(returns),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Annualized Return",
            ascending=False,
        )
        .reset_index(drop=True)
    )


