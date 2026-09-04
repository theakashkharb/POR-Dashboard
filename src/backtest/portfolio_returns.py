from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.validation import (
    validate_prices,
    validate_returns,
)


# ============================================================
# ASSET RETURNS
# ============================================================

def calculate_returns(prices):
    """
    Calculate simple asset returns.
    """
    prices = validate_prices(prices)

    returns = prices.pct_change()

    return returns.dropna(how="all")


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def calculate_portfolio_returns(
    asset_returns,
    weights,
):
    """
    Calculate portfolio returns using fixed weights.

    Portfolio return:
        Rp = sum(weight_i * return_i)
    """
    asset_returns = validate_returns(
        asset_returns
    )

    weights = pd.Series(
        weights,
        dtype=float,
    )

    common_assets = (
        asset_returns.columns
        .intersection(weights.index)
    )

    if len(common_assets) == 0:
        raise ValueError(
            "No common assets between "
            "returns and weights"
        )

    aligned_returns = (
        asset_returns[common_assets]
    )

    aligned_weights = (
        weights[common_assets]
    )

    if not np.isfinite(
        aligned_weights.values
    ).all():
        raise ValueError(
            "weights contain non-finite values"
        )

    if (
        aligned_weights.sum() == 0
    ):
        raise ValueError(
            "Portfolio weights sum to zero"
        )

    aligned_weights = (
        aligned_weights
        / aligned_weights.sum()
    )

    portfolio_returns = (
        aligned_returns
        .mul(
            aligned_weights,
            axis=1,
        )
        .sum(axis=1)
    )

    portfolio_returns.name = (
        "portfolio_return"
    )

    return portfolio_returns


# ============================================================
# CUMULATIVE PORTFOLIO RETURN
# ============================================================

def calculate_portfolio_return(
    returns,
):
    """
    Calculate cumulative portfolio return.
    """
    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    if len(returns) == 0:
        return np.nan

    return float(
        (1 + returns).prod() - 1
    )


# ============================================================
# PORTFOLIO NAV
# ============================================================

def calculate_nav(
    returns,
    initial_value=1.0,
):
    """
    Construct portfolio NAV from returns.
    """
    if not np.isfinite(
        initial_value
    ):
        raise ValueError(
            "initial_value must be finite"
        )

    if initial_value <= 0:
        raise ValueError(
            "initial_value must be greater than zero"
        )

    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    return (
        initial_value
        * (1 + returns).cumprod()
    )