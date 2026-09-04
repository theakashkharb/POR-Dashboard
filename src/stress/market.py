from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    """
    Validate and convert asset returns to a DataFrame.
    """
    returns = pd.DataFrame(returns, dtype=float)

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if not np.isfinite(returns.values).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    return returns


def _validate_weights(weights, assets):
    """
    Validate portfolio weights and align them to assets.
    """
    weights = pd.Series(weights, dtype=float)

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    if not np.isfinite(weights.values).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    if isinstance(weights.index, pd.RangeIndex):
        if len(weights) != len(assets):
            raise ValueError(
                "weights length must match number of assets."
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

    if np.any(weights.values < 0):
        raise ValueError(
            "negative weights are not supported."
        )

    total = float(weights.sum())

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    return weights / total


# ============================================================
# MARKET SHOCKS
# ============================================================

def apply_market_shock(
    returns,
    shock,
):
    """
    Apply the same return shock to every asset.

    Parameters
    ----------
    returns : DataFrame
        Asset return matrix.

    shock : float
        Market shock expressed as a decimal.

        Example:
            -0.10 = -10% market shock
             0.05 = +5% market shock

    Returns
    -------
    DataFrame
        Stressed return matrix.
    """
    returns = _validate_returns(
        returns
    )

    if not np.isfinite(shock):
        raise ValueError(
            "shock must be finite."
        )

    stressed = returns + float(shock)

    return stressed


def apply_asset_shocks(
    returns,
    shocks,
):
    """
    Apply individual shocks to each asset.

    Parameters
    ----------
    returns : DataFrame
        Asset return matrix.

    shocks : Series, dict, or array-like
        Shock for each asset.

    Returns
    -------
    DataFrame
        Stressed return matrix.
    """
    returns = _validate_returns(
        returns
    )

    shocks = pd.Series(
        shocks,
        dtype=float,
    )

    if len(shocks) != len(
        returns.columns
    ):
        raise ValueError(
            "shocks length must match number of assets."
        )

    if isinstance(
        shocks.index,
        pd.RangeIndex,
    ):
        shocks.index = returns.columns

    else:
        shocks = shocks.reindex(
            returns.columns
        )

    if shocks.isna().any():
        raise ValueError(
            "shocks must contain every asset."
        )

    if not np.isfinite(
        shocks.values
    ).all():
        raise ValueError(
            "shocks contain non-finite values."
        )

    stressed = (
        returns
        + shocks
    )

    return stressed


# ============================================================
# STANDARD MARKET SCENARIOS
# ============================================================

def market_crash(
    returns,
    shock=-0.20,
):
    """
    Simulate a broad market crash.

    Default:
        -20% shock to every asset.
    """
    return apply_market_shock(
        returns,
        shock,
    )


def severe_market_crash(
    returns,
    shock=-0.30,
):
    """
    Simulate a severe market crash.

    Default:
        -30% shock.
    """
    return apply_market_shock(
        returns,
        shock,
    )


def extreme_market_crash(
    returns,
    shock=-0.50,
):
    """
    Simulate an extreme market crash.

    Default:
        -50% shock.
    """
    return apply_market_shock(
        returns,
        shock,
    )


def market_rally(
    returns,
    shock=0.10,
):
    """
    Simulate a broad market rally.

    Default:
        +10% shock.
    """
    return apply_market_shock(
        returns,
        shock,
    )


# ============================================================
# PORTFOLIO IMPACT
# ============================================================

def portfolio_stressed_return(
    weights,
    stressed_returns,
):
    """
    Calculate portfolio return under a stress scenario.

    Returns
    -------
    float
        Portfolio stressed return.
    """
    stressed_returns = _validate_returns(
        stressed_returns
    )

    weights = _validate_weights(
        weights,
        stressed_returns.columns,
    )

    portfolio_returns = (
        stressed_returns
        @ weights
    )

    return float(
        portfolio_returns.mean()
    )


def portfolio_stress_loss(
    weights,
    stressed_returns,
):
    """
    Calculate portfolio loss from stressed returns.

    Positive value represents a loss.
    """
    stressed_return = (
        portfolio_stressed_return(
            weights,
            stressed_returns,
        )
    )

    return float(
        -stressed_return
    )


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

def run_market_scenario(
    returns,
    weights,
    shock,
):
    """
    Run one market stress scenario.

    Returns
    -------
    dict
        Scenario name, shock and portfolio impact.
    """
    returns = _validate_returns(
        returns
    )

    stressed_returns = (
        apply_market_shock(
            returns,
            shock,
        )
    )

    stressed_return = (
        portfolio_stressed_return(
            weights,
            stressed_returns,
        )
    )

    return {
        "shock": float(shock),
        "portfolio_return": stressed_return,
        "portfolio_loss": -stressed_return,
    }


def run_standard_market_scenarios(
    returns,
    weights,
):
    """
    Run standard market stress scenarios.

    Scenarios
    ---------
    Base
    Market Crash
    Severe Crash
    Extreme Crash
    Market Rally
    """
    scenarios = {
        "Base": 0.0,
        "Market Crash": -0.20,
        "Severe Crash": -0.30,
        "Extreme Crash": -0.50,
        "Market Rally": 0.10,
    }

    results = []

    for name, shock in scenarios.items():

        result = run_market_scenario(
            returns,
            weights,
            shock,
        )

        result["scenario"] = name

        results.append(result)

    return (
        pd.DataFrame(results)
        [
            [
                "scenario",
                "shock",
                "portfolio_return",
                "portfolio_loss",
            ]
        ]
    )


# ============================================================
# WORST CASE ANALYSIS
# ============================================================

def worst_market_scenario(
    returns,
    weights,
    shocks=None,
):
    """
    Find the worst portfolio outcome across
    supplied market shocks.

    Default shocks:
        -10%, -20%, -30%, -40%, -50%
    """
    returns = _validate_returns(
        returns
    )

    if shocks is None:
        shocks = [
            -0.10,
            -0.20,
            -0.30,
            -0.40,
            -0.50,
        ]

    results = []

    for shock in shocks:

        result = run_market_scenario(
            returns,
            weights,
            shock,
        )

        results.append(result)

    results = pd.DataFrame(
        results
    )

    worst_index = (
        results[
            "portfolio_return"
        ].idxmin()
    )

    return results.loc[
        worst_index
    ].to_dict()


# ============================================================
# STRESS SUMMARY
# ============================================================

def market_stress_summary(
    returns,
    weights,
):
    """
    Generate a complete market stress summary.
    """
    returns = _validate_returns(
        returns
    )

    weights = _validate_weights(
        weights,
        returns.columns,
    )

    scenarios = (
        run_standard_market_scenarios(
            returns,
            weights,
        )
    )

    worst = worst_market_scenario(
        returns,
        weights,
    )

    return {
        "scenarios": scenarios,
        "worst_case": worst,
        "number_of_assets": len(
            returns.columns
        ),
    }