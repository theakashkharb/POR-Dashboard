"""
POR-Dashboard
Risk Volatility
==============

Risk-specific volatility calculations and volatility-based scaling.

Historical and EWMA volatility estimation belongs to:
    src.features.volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_volatility(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """
    Calculate annualized portfolio volatility.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.
    covariance : array-like
        Annualized covariance matrix.

    Returns
    -------
    float
        Portfolio volatility.
    """
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    if covariance.ndim != 2:
        raise ValueError("covariance must be two-dimensional")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square")

    if len(weights) != covariance.shape[0]:
        raise ValueError(
            "weights and covariance dimensions must match"
        )

    variance = weights @ covariance @ weights

    if variance < 0 and not np.isclose(variance, 0):
        raise ValueError("portfolio variance cannot be negative")

    return float(np.sqrt(max(variance, 0.0)))


def volatility_targeting(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    target_volatility: float,
    max_leverage: float | None = None,
) -> np.ndarray:
    """
    Scale portfolio exposure to target a specified volatility.

    Parameters
    ----------
    weights : array-like
        Original portfolio weights.
    covariance : array-like
        Annualized covariance matrix.
    target_volatility : float
        Desired annualized portfolio volatility.
    max_leverage : float, optional
        Maximum allowed scaling factor.

    Returns
    -------
    np.ndarray
        Volatility-targeted weights.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive")

    weights_array = np.asarray(weights, dtype=float)

    current_volatility = portfolio_volatility(
        weights_array,
        covariance,
    )

    if current_volatility == 0:
        return weights_array.copy()

    scale = target_volatility / current_volatility

    if max_leverage is not None:
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

        scale = min(scale, max_leverage)

    return weights_array * scale


def volatility_limit(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    maximum_volatility: float,
) -> np.ndarray:
    """
    Reduce portfolio exposure when volatility exceeds a limit.

    If current volatility is below the limit, weights are unchanged.
    """
    if maximum_volatility <= 0:
        raise ValueError("maximum_volatility must be positive")

    weights_array = np.asarray(weights, dtype=float)

    current_volatility = portfolio_volatility(
        weights_array,
        covariance,
    )

    if current_volatility <= maximum_volatility:
        return weights_array.copy()

    scale = maximum_volatility / current_volatility

    return weights_array * scale


def dynamic_volatility_scaling(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    target_volatility: float,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> np.ndarray:
    """
    Dynamically scale portfolio weights based on current volatility.

    The scaling factor is:

        target_volatility / current_volatility

    constrained between min_scale and max_scale.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility must be positive")

    if min_scale < 0:
        raise ValueError("min_scale cannot be negative")

    if max_scale < min_scale:
        raise ValueError(
            "max_scale must be greater than or equal to min_scale"
        )

    weights_array = np.asarray(weights, dtype=float)

    current_volatility = portfolio_volatility(
        weights_array,
        covariance,
    )

    if current_volatility == 0:
        scale = max_scale
    else:
        scale = target_volatility / current_volatility
        scale = np.clip(scale, min_scale, max_scale)

    return weights_array * scale