from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.repository import get_sector_data


def calculate_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert stock prices into daily returns.
    """

    if data.empty:
        return pd.DataFrame()

    prices = (
        data
        .pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )
        .sort_index()
    )

    returns = prices.pct_change()

    return returns


def equal_weight_sector_returns(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate equal-weighted daily sector returns.

    Each stock receives the same weight on each day.
    """

    returns = calculate_returns(data)

    if returns.empty:
        return pd.Series(dtype=float)

    sector_returns = returns.mean(
        axis=1,
        skipna=True,
    )

    return sector_returns.dropna()


def cumulative_returns(
    returns: pd.Series,
) -> pd.Series:
    """
    Calculate cumulative growth of 1 unit.
    """

    if returns.empty:
        return pd.Series(dtype=float)

    return (1 + returns).cumprod()


def annualized_return(
    returns: pd.Series,
) -> float:
    """
    Calculate annualized return using daily returns.
    """

    if returns.empty:
        return np.nan

    periods = len(returns)

    if periods < 2:
        return np.nan

    total_return = (
        (1 + returns).prod()
    )

    years = periods / 252

    if years <= 0:
        return np.nan

    return total_return ** (1 / years) - 1


def annualized_volatility(
    returns: pd.Series,
) -> float:
    """
    Calculate annualized volatility.
    """

    if returns.empty:
        return np.nan

    return returns.std() * np.sqrt(252)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    """

    if returns.empty:
        return np.nan

    daily_rf = (
        (1 + risk_free_rate) ** (1 / 252)
    ) - 1

    excess_returns = returns - daily_rf

    volatility = excess_returns.std()

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(252)
    )


def maximum_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum drawdown.
    """

    if returns.empty:
        return np.nan

    wealth = (1 + returns).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth / running_peak
    ) - 1

    return drawdown.min()


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Calculate rolling annualized volatility.
    """

    if returns.empty:
        return pd.Series(dtype=float)

    return (
        returns
        .rolling(window)
        .std()
        * np.sqrt(252)
    )


def sector_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Calculate the main sector-level risk/return metrics.
    """

    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "maximum_drawdown": np.nan,
        }

    cumulative = cumulative_returns(
        returns
    )

    total_return = (
        cumulative.iloc[-1] - 1
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return(
            returns
        ),
        "annualized_volatility": annualized_volatility(
            returns
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate,
        ),
        "maximum_drawdown": maximum_drawdown(
            returns
        ),
    }


def get_sector_analysis(
    sector: str,
    start: str | None = None,
    end: str | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Complete equal-weighted sector analysis.
    """

    data = get_sector_data(
        sector,
        start=start,
        end=end,
    )

    if data.empty:
        return {
            "sector": sector,
            "data": pd.DataFrame(),
            "returns": pd.Series(dtype=float),
            "cumulative_returns": pd.Series(dtype=float),
            "rolling_volatility": pd.Series(dtype=float),
            "metrics": {},
        }

    returns = equal_weight_sector_returns(
        data
    )

    cumulative = cumulative_returns(
        returns
    )

    rolling_vol = rolling_volatility(
        returns
    )

    metrics = sector_metrics(
        returns,
        risk_free_rate,
    )

    return {
        "sector": sector,
        "data": data,
        "returns": returns,
        "cumulative_returns": cumulative,
        "rolling_volatility": rolling_vol,
        "metrics": metrics,
    }