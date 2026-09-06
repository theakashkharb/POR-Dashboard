from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_market_returns(
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the equal-weighted market return
    across all available stocks for each date.
    """

    if returns.empty:
        raise ValueError(
            "Return data is empty."
        )

    return returns.mean(
        axis=1,
        skipna=True,
    )


def calculate_total_return(
    returns: pd.Series,
) -> float:
    """
    Calculate total compounded return
    over the supplied period.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Return series is empty."
        )

    return float(
        (1.0 + returns).prod() - 1.0
    )


def calculate_annualized_return(
    returns: pd.Series,
) -> float:
    """
    Calculate annualized compounded return.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Return series is empty."
        )

    total_return = calculate_total_return(
        returns
    )

    years = len(returns) / TRADING_DAYS

    if years <= 0:
        return float("nan")

    return float(
        (1.0 + total_return) ** (1.0 / years) - 1.0
    )


def calculate_annualized_volatility(
    returns: pd.Series,
) -> float:
    """
    Calculate annualized volatility.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Return series is empty."
        )

    return float(
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Return series is empty."
        )

    daily_risk_free_rate = (
        (1.0 + risk_free_rate)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    excess_returns = (
        returns - daily_risk_free_rate
    )

    volatility = returns.std(ddof=1)

    if volatility == 0:
        return float("nan")

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(TRADING_DAYS)
    )


def calculate_maximum_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum drawdown.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Return series is empty."
        )

    wealth = (
        1.0 + returns
    ).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max - 1.0
    )

    return float(drawdown.min())


def calculate_market_snapshot(
    returns: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Calculate the complete market snapshot
    for the selected period.
    """

    if returns.empty:
        raise ValueError(
            "Return data is empty."
        )

    market_returns = calculate_market_returns(
        returns
    )

    if start_date is not None:
        market_returns = market_returns[
            market_returns.index
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        market_returns = market_returns[
            market_returns.index
            <= pd.Timestamp(end_date)
        ]

    market_returns = market_returns.dropna()

    if market_returns.empty:
        raise ValueError(
            "No market returns available "
            "for this period."
        )

    return {
        "Returns": calculate_total_return(
            market_returns
        ),
        "Annualized Return": calculate_annualized_return(
            market_returns
        ),
        "Volatility": calculate_annualized_volatility(
            market_returns
        ),
        "Sharpe": calculate_sharpe_ratio(
            market_returns,
            risk_free_rate=risk_free_rate,
        ),
        "Max Drawdown": calculate_maximum_drawdown(
            market_returns
        ),
    }