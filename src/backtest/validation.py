from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# PRICE VALIDATION
# ============================================================

def validate_prices(prices):
    """
    Validate a price DataFrame.

    The input must be a non-empty pandas DataFrame.
    Duplicate columns are removed and the index is sorted.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "prices must be a pandas DataFrame"
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty"
        )

    prices = prices.sort_index()

    prices = prices.loc[
        :,
        ~prices.columns.duplicated()
    ]

    return prices


# ============================================================
# RETURN VALIDATION
# ============================================================

def validate_returns(returns):
    """
    Validate a return DataFrame.

    Duplicate columns are removed and the index is sorted.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            "returns must be a pandas DataFrame"
        )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    returns = returns.sort_index()

    returns = returns.loc[
        :,
        ~returns.columns.duplicated()
    ]

    return returns


# ============================================================
# ASSET VALIDATION
# ============================================================

def validate_selected_assets(
    prices,
    selected_assets,
):
    """
    Validate that the requested assets exist
    in the price DataFrame.
    """
    if not selected_assets:
        raise ValueError(
            "selected_assets cannot be empty"
        )

    missing_assets = [
        asset
        for asset in selected_assets
        if asset not in prices.columns
    ]

    if missing_assets:
        raise ValueError(
            "Selected assets not found in prices: "
            f"{missing_assets}"
        )

    return list(selected_assets)


# ============================================================
# WEIGHT VALIDATION
# ============================================================

def validate_weights(
    weights,
    assets=None,
    require_long_only=True,
    require_fully_invested=True,
):
    """
    Validate portfolio weights.

    Parameters
    ----------
    weights : array-like or Series
        Portfolio weights.

    assets : optional
        Asset names used to align unnamed weights.

    require_long_only : bool
        Reject negative weights when True.

    require_fully_invested : bool
        Require weights to sum approximately to 1.
    """
    weights = pd.Series(
        weights,
        dtype=float,
    )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty"
        )

    if not np.isfinite(
        weights.values
    ).all():
        raise ValueError(
            "weights contain non-finite values"
        )

    if require_long_only and (
        weights.values < 0
    ).any():
        raise ValueError(
            "negative weights are not supported"
        )

    if assets is not None:

        assets = list(assets)

        if isinstance(
            weights.index,
            pd.RangeIndex,
        ):
            if len(weights) != len(assets):
                raise ValueError(
                    "weights length must match "
                    "number of assets"
                )

            weights.index = assets

        else:
            missing = [
                asset
                for asset in assets
                if asset not in weights.index
            ]

            if missing:
                raise ValueError(
                    f"weights missing assets: {missing}"
                )

            weights = weights.loc[assets]

    if require_fully_invested:

        if not np.isclose(
            weights.sum(),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "weights must sum to 1"
            )

    return weights


# ============================================================
# BACKTEST PARAMETER VALIDATION
# ============================================================

def validate_train_window(
    train_window,
    minimum_observations=None,
):
    """
    Validate the walk-forward training window.
    """
    if not isinstance(
        train_window,
        (int, np.integer),
    ):
        raise TypeError(
            "train_window must be an integer"
        )

    if train_window <= 0:
        raise ValueError(
            "train_window must be greater than zero"
        )

    if (
        minimum_observations is not None
        and minimum_observations <= train_window
    ):
        raise ValueError(
            "Not enough data for train_window"
        )

    return int(train_window)


def validate_rebalance_frequency(
    rebalance_frequency,
):
    """
    Validate pandas-compatible rebalance frequency.
    """
    if not isinstance(
        rebalance_frequency,
        str,
    ):
        raise TypeError(
            "rebalance_frequency must be a string"
        )

    if not rebalance_frequency.strip():
        raise ValueError(
            "rebalance_frequency cannot be empty"
        )

    return rebalance_frequency


def validate_max_turnover(
    max_turnover,
):
    """
    Validate maximum turnover constraint.
    """
    if max_turnover is None:
        return None

    if not np.isfinite(
        max_turnover
    ):
        raise ValueError(
            "max_turnover must be finite"
        )

    if max_turnover < 0:
        raise ValueError(
            "max_turnover cannot be negative"
        )

    return float(max_turnover)


def validate_initial_capital(
    initial_capital,
):
    """
    Validate starting portfolio capital.
    """
    if not np.isfinite(
        initial_capital
    ):
        raise ValueError(
            "initial_capital must be finite"
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero"
        )

    return float(initial_capital)


# ============================================================
# BENCHMARK VALIDATION
# ============================================================

def validate_benchmark_prices(
    benchmark_prices,
):
    """
    Validate benchmark price data.

    A benchmark DataFrame must contain exactly
    one column.
    """
    if benchmark_prices is None:
        return None

    if isinstance(
        benchmark_prices,
        pd.DataFrame,
    ):
        if benchmark_prices.empty:
            raise ValueError(
                "benchmark_prices cannot be empty"
            )

        if benchmark_prices.shape[1] != 1:
            raise ValueError(
                "benchmark_prices must contain "
                "exactly one column"
            )

        benchmark_prices = (
            benchmark_prices.iloc[:, 0]
        )

    benchmark_prices = pd.Series(
        benchmark_prices,
        dtype=float,
    )

    if benchmark_prices.empty:
        raise ValueError(
            "benchmark_prices cannot be empty"
        )

    if not np.isfinite(
        benchmark_prices.values
    ).all():
        raise ValueError(
            "benchmark_prices contain "
            "non-finite values"
        )

    return benchmark_prices.sort_index()