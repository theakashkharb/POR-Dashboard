"""
POR-Dashboard
Stock Selection
==============

Selects stocks from the screened investment universe.

Available methods:
    - baseline
    - momentum
    - low_volatility
    - multi_factor
    - custom_quant

This module is responsible only for stock selection.
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# FACTOR CALCULATIONS
# ============================================================

def calculate_momentum(
    data: pd.DataFrame,
    lookback: int = 90,
) -> pd.Series:
    """Calculate price momentum for each ticker."""

    data = (
        data.sort_values(["Ticker", "Date"])
        .copy()
    )

    momentum = (
        data.groupby("Ticker")["Close"]
        .transform(
            lambda x: x / x.shift(lookback) - 1
        )
    )

    data["Momentum"] = momentum

    return (
        data.groupby("Ticker")["Momentum"]
        .last()
        .dropna()
    )


def calculate_volatility(
    data: pd.DataFrame,
    window: int = 90,
) -> pd.Series:
    """Calculate rolling historical volatility for each ticker."""

    data = (
        data.sort_values(["Ticker", "Date"])
        .copy()
    )

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    volatility = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .groupby(level=0)
        .last()
        .dropna()
    )

    return volatility


def calculate_mean_return(
    data: pd.DataFrame,
    window: int = 90,
) -> pd.Series:
    """Calculate rolling mean daily return for each ticker."""

    data = (
        data.sort_values(["Ticker", "Date"])
        .copy()
    )

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    mean_return = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .mean()
        .groupby(level=0)
        .last()
        .dropna()
    )

    return mean_return


# ============================================================
# INDIVIDUAL SELECTION METHODS
# ============================================================

def momentum_selection(
    data: pd.DataFrame,
    n_stocks: int = 25,
    lookback: int = 90,
) -> list[str]:
    """Select stocks with the highest momentum."""

    momentum = calculate_momentum(
        data,
        lookback=lookback,
    )

    return (
        momentum
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def low_volatility_selection(
    data: pd.DataFrame,
    n_stocks: int = 25,
    window: int = 90,
) -> list[str]:
    """Select stocks with the lowest volatility."""

    volatility = calculate_volatility(
        data,
        window=window,
    )

    return (
        volatility
        .sort_values(ascending=True)
        .head(n_stocks)
        .index
        .tolist()
    )


def multi_factor_selection(
    data: pd.DataFrame,
    n_stocks: int = 25,
    momentum_weight: float = 0.5,
    low_vol_weight: float = 0.5,
    lookback: int = 90,
) -> list[str]:
    """
    Select stocks using momentum and low-volatility factors.
    """

    momentum = calculate_momentum(
        data,
        lookback=lookback,
    )

    volatility = calculate_volatility(
        data,
        window=lookback,
    )

    metrics = pd.concat(
        [
            momentum.rename("momentum"),
            volatility.rename("volatility"),
        ],
        axis=1,
    ).dropna()

    metrics["momentum_score"] = (
        metrics["momentum"]
        .rank(pct=True)
    )

    metrics["low_vol_score"] = (
        1
        - metrics["volatility"].rank(pct=True)
    )

    metrics["score"] = (
        momentum_weight * metrics["momentum_score"]
        + low_vol_weight * metrics["low_vol_score"]
    )

    return (
        metrics["score"]
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def custom_quant_selection(
    data: pd.DataFrame,
    n_stocks: int = 25,
    momentum_weight: float = 0.4,
    low_vol_weight: float = 0.3,
    return_weight: float = 0.3,
    lookback: int = 90,
) -> list[str]:
    """
    Select stocks using momentum, low volatility,
    and mean return.
    """

    momentum = calculate_momentum(
        data,
        lookback=lookback,
    )

    volatility = calculate_volatility(
        data,
        window=lookback,
    )

    mean_return = calculate_mean_return(
        data,
        window=lookback,
    )

    metrics = pd.concat(
        [
            momentum.rename("momentum"),
            volatility.rename("volatility"),
            mean_return.rename("mean_return"),
        ],
        axis=1,
    ).dropna()

    metrics["momentum_score"] = (
        metrics["momentum"]
        .rank(pct=True)
    )

    metrics["low_vol_score"] = (
        1
        - metrics["volatility"].rank(pct=True)
    )

    metrics["return_score"] = (
        metrics["mean_return"]
        .rank(pct=True)
    )

    metrics["score"] = (
        momentum_weight * metrics["momentum_score"]
        + low_vol_weight * metrics["low_vol_score"]
        + return_weight * metrics["return_score"]
    )

    return (
        metrics["score"]
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )


def baseline_selection(
    data: pd.DataFrame,
    n_stocks: int = 25,
) -> list[str]:
    """Select the first n stocks alphabetically."""

    return (data["Ticker"].drop_duplicates().sort_values().head(n_stocks).tolist())


# ============================================================
# SELECTION DISPATCHER
# ============================================================

SELECTION_METHODS = {
    "baseline": baseline_selection,
    "momentum": momentum_selection,
    "low_volatility": low_volatility_selection,
    "multi_factor": multi_factor_selection,
    "custom_quant": custom_quant_selection,
}


def select_stocks(
    data: pd.DataFrame,
    method: str = "baseline",
    n_stocks: int = 25,
    lookback: int = 90,
) -> list[str]:
    """
    Select stocks using the requested selection method.
    """

    if method not in SELECTION_METHODS:
        available = ", ".join(
            SELECTION_METHODS.keys()
        )

        raise ValueError(
            f"Unknown selection method: {method}. "
            f"Available methods: {available}"
        )

    if n_stocks <= 0:
        raise ValueError(
            "n_stocks must be greater than zero."
        )

    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than zero."
        )

    if method == "baseline":
        return baseline_selection(
            data,
            n_stocks=n_stocks,
        )

    if method == "low_volatility":
        return low_volatility_selection(
            data,
            n_stocks=n_stocks,
            window=lookback,
        )

    selector = SELECTION_METHODS[method]

    return selector(
        data,
        n_stocks=n_stocks,
        lookback=lookback,
    )   