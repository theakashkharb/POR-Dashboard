from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.stocks.data import _stock_return_series
from src.analytics.stocks.performance import calculate_stock_performance

TRADING_DAYS = 252


def calculate_stock_volatility(
    returns: pd.Series,
) -> float:
    returns = returns.dropna()

    if returns.empty:
        return float("nan")

    return float(
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )


def calculate_stock_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    returns = returns.dropna()

    if returns.empty:
        return float("nan")

    daily_rf = (
        (1.0 + risk_free_rate)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    excess_returns = returns - daily_rf
    volatility = returns.std(ddof=1)

    if volatility == 0 or pd.isna(volatility):
        return float("nan")

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(TRADING_DAYS)
    )


def calculate_stock_sortino(
    returns: pd.Series,
    target_return: float = 0.0,
) -> float:
    returns = returns.dropna()

    if returns.empty:
        return float("nan")

    daily_target = (
        (1.0 + target_return)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    excess_returns = returns - daily_target
    downside = np.minimum(
        excess_returns,
        0.0,
    )

    downside_deviation = np.sqrt(
        np.mean(downside ** 2)
    ) * np.sqrt(TRADING_DAYS)

    annualized_excess_return = (
        excess_returns.mean()
        * TRADING_DAYS
    )

    if downside_deviation == 0:
        return float("nan")

    return float(
        annualized_excess_return
        / downside_deviation
    )


def calculate_stock_drawdown(
    returns: pd.Series,
) -> pd.Series:
    returns = returns.dropna()

    if returns.empty:
        return pd.Series(
            dtype=float,
            name="Drawdown",
        )

    wealth = (
        1.0 + returns
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth / running_peak - 1.0
    )

    drawdown.name = "Drawdown"

    return drawdown


def calculate_stock_max_drawdown_recovery(
    returns: pd.Series,
) -> dict[str, float | int | None]:
    returns = returns.dropna()

    if returns.empty:
        return {
            "Max Drawdown": float("nan"),
            "Recovery Duration": None,
        }

    wealth = (
        1.0 + returns
    ).cumprod()

    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    trough_position = int(
        drawdown.to_numpy().argmin()
    )

    max_drawdown = float(
        drawdown.iloc[trough_position]
    )

    if max_drawdown >= 0:
        return {
            "Max Drawdown": 0.0,
            "Recovery Duration": 0,
        }

    peak_value = float(
        running_peak.iloc[trough_position]
    )

    recovery_duration = None

    for position in range(
        trough_position + 1,
        len(wealth),
    ):
        if wealth.iloc[position] >= peak_value:
            recovery_duration = (
                position - trough_position
            )
            break

    return {
        "Max Drawdown": max_drawdown,
        "Recovery Duration": recovery_duration,
    }


def calculate_stock_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    returns = returns.dropna()

    if returns.empty:
        return float("nan")

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    threshold = returns.quantile(
        1.0 - confidence
    )

    tail_returns = returns[
        returns <= threshold
    ]

    if tail_returns.empty:
        return float("nan")

    return float(
        -tail_returns.mean()
    )


def calculate_stock_calmar(
    cagr: float,
    max_drawdown: float,
) -> float:
    if pd.isna(cagr) or pd.isna(max_drawdown):
        return float("nan")

    if max_drawdown == 0:
        return float("nan")

    return float(
        cagr / abs(max_drawdown)
    )

def calculate_stock_risk_metrics(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | None]:
    returns = _stock_return_series(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    performance = calculate_stock_performance(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    drawdown = calculate_stock_max_drawdown_recovery(
        returns
    )

    cvar = calculate_stock_cvar(
        returns,
        confidence=0.95,
    )

    return {
        "Annualized Volatility":
            calculate_stock_volatility(returns),
        "Sharpe":
            calculate_stock_sharpe(
                returns,
                risk_free_rate,
            ),
        "Sortino":
            calculate_stock_sortino(returns),
        "Max Drawdown":
            drawdown["Max Drawdown"],
        "Recovery Duration":
            drawdown["Recovery Duration"],
        "CVaR 95%":
            cvar,
        "Calmar":
            calculate_stock_calmar(
                performance["CAGR"],
                drawdown["Max Drawdown"],
            ),
    }