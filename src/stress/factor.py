from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    """
    Validate asset return data.
    """
    returns = pd.DataFrame(
        returns,
        dtype=float,
    )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    if not np.isfinite(
        returns.values
    ).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    return returns


def _validate_factor_returns(
    factor_returns,
):
    """
    Validate factor return data.
    """
    factor_returns = pd.DataFrame(
        factor_returns,
        dtype=float,
    )

    if factor_returns.empty:
        raise ValueError(
            "factor_returns cannot be empty."
        )

    if not np.isfinite(
        factor_returns.values
    ).all():
        raise ValueError(
            "factor_returns contain "
            "non-finite values."
        )

    return factor_returns


def _validate_weights(
    weights,
    assets,
):
    """
    Validate and align portfolio weights.
    """
    weights = pd.Series(
        weights,
        dtype=float,
    )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    if not np.isfinite(
        weights.values
    ).all():
        raise ValueError(
            "weights contain "
            "non-finite values."
        )

    if isinstance(
        weights.index,
        pd.RangeIndex,
    ):
        if len(weights) != len(assets):
            raise ValueError(
                "weights length must match "
                "number of assets."
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

    if (
        weights.values < 0
    ).any():
        raise ValueError(
            "negative weights are not supported."
        )

    total = float(
        weights.sum()
    )

    if total <= 0:
        raise ValueError(
            "weights must have a "
            "positive sum."
        )

    return weights / total


# ============================================================
# FACTOR EXPOSURE
# ============================================================

def calculate_factor_exposure(
    asset_returns,
    factor_returns,
):
    """
    Estimate asset exposure to factors
    using ordinary least squares.

    Model:

        Asset Return = alpha
                     + beta * Factor Return
                     + residual

    For multiple factors:

        R = alpha + F beta + epsilon

    Returns
    -------
    DataFrame
        Factor beta exposure for each asset.
    """
    asset_returns = _validate_returns(
        asset_returns
    )

    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    aligned = pd.concat(
        [
            asset_returns,
            factor_returns,
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping observations "
            "between assets and factors."
        )

    asset_data = aligned[
        asset_returns.columns
    ]

    factor_data = aligned[
        factor_returns.columns
    ]

    X = factor_data.values

    # Add intercept.
    X = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    exposures = []

    for asset in asset_data.columns:

        y = asset_data[
            asset
        ].values

        coefficients = (
            np.linalg.lstsq(
                X,
                y,
                rcond=None,
            )[0]
        )

        exposures.append(
            coefficients[1:]
        )

    return pd.DataFrame(
        np.asarray(exposures),
        index=asset_data.columns,
        columns=factor_data.columns,
    )


# ============================================================
# FACTOR SHOCKS
# ============================================================

def apply_factor_shock(
    factor_returns,
    shocks,
):
    """
    Apply shocks to factor returns.

    Parameters
    ----------
    factor_returns : DataFrame
        Historical factor returns.

    shocks : Series, dict, or array-like
        Shock applied to each factor.

    Returns
    -------
    DataFrame
        Stressed factor returns.
    """
    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    shocks = pd.Series(
        shocks,
        dtype=float,
    )

    if len(shocks) != len(
        factor_returns.columns
    ):
        raise ValueError(
            "shocks length must match "
            "number of factors."
        )

    if isinstance(
        shocks.index,
        pd.RangeIndex,
    ):
        shocks.index = (
            factor_returns.columns
        )
    else:
        shocks = shocks.reindex(
            factor_returns.columns
        )

    if shocks.isna().any():
        raise ValueError(
            "shocks must contain "
            "every factor."
        )

    if not np.isfinite(
        shocks.values
    ).all():
        raise ValueError(
            "shocks contain "
            "non-finite values."
        )

    stressed = (
        factor_returns
        + shocks
    )

    return stressed


# ============================================================
# ASSET RETURNS FROM FACTOR SHOCK
# ============================================================

def factor_stressed_returns(
    asset_returns,
    factor_returns,
    factor_shocks,
):
    """
    Estimate asset returns under a factor shock.

    Uses estimated factor exposures:

        stressed asset return
        =
        beta @ stressed factors

    The historical asset alpha is also retained.
    """
    asset_returns = _validate_returns(
        asset_returns
    )

    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    stressed_factors = apply_factor_shock(
        factor_returns,
        factor_shocks,
    )

    # Calculate historical factor-driven
    # component and alpha.
    aligned = pd.concat(
        [
            asset_returns,
            factor_returns,
        ],
        axis=1,
        join="inner",
    ).dropna()

    factor_data = aligned[
        factor_returns.columns
    ]

    asset_data = aligned[
        asset_returns.columns
    ]

    factor_mean = (
        factor_data.mean()
    )

    predicted_factor_component = (
        factor_mean
        @ exposures.T
    )

    alpha = (
        asset_data.mean()
        - predicted_factor_component
    )

    stressed_factor_component = (
        stressed_factors
        @ exposures.T
    )

    stressed = (
        stressed_factor_component
        + alpha
    )

    stressed.columns = (
        asset_returns.columns
    )

    return stressed


# ============================================================
# PORTFOLIO FACTOR EXPOSURE
# ============================================================

def portfolio_factor_exposure(
    weights,
    asset_returns,
    factor_returns,
):
    """
    Calculate portfolio exposure to
    each risk factor.

    Portfolio factor exposure:

        beta_p = w' beta
    """
    asset_returns = _validate_returns(
        asset_returns
    )

    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    weights = _validate_weights(
        weights,
        asset_returns.columns,
    )

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    exposures = exposures.loc[
        weights.index
    ]

    portfolio_exposure = (
        weights.values
        @ exposures.values
    )

    return pd.Series(
        portfolio_exposure,
        index=exposures.columns,
        name="portfolio_exposure",
    )


# ============================================================
# PORTFOLIO FACTOR IMPACT
# ============================================================

def portfolio_factor_impact(
    weights,
    asset_returns,
    factor_returns,
    factor_shocks,
):
    """
    Calculate portfolio impact from
    factor shocks.

    Returns
    -------
    Series
        Portfolio return contribution
        from each factor.
    """
    asset_returns = _validate_returns(
        asset_returns
    )

    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    weights = _validate_weights(
        weights,
        asset_returns.columns,
    )

    exposure = portfolio_factor_exposure(
        weights,
        asset_returns,
        factor_returns,
    )

    shocks = pd.Series(
        factor_shocks,
        dtype=float,
    )

    if isinstance(
        shocks.index,
        pd.RangeIndex,
    ):
        if len(shocks) != len(
            exposure
        ):
            raise ValueError(
                "factor_shocks length "
                "must match factors."
            )

        shocks.index = exposure.index

    else:
        shocks = shocks.reindex(
            exposure.index
        )

    if shocks.isna().any():
        raise ValueError(
            "factor_shocks must contain "
            "every factor."
        )

    return (
        exposure * shocks
    )


# ============================================================
# FACTOR SCENARIO
# ============================================================

def run_factor_scenario(
    weights,
    asset_returns,
    factor_returns,
    factor_shocks,
):
    """
    Run one complete factor stress scenario.

    Returns
    -------
    dict
        Exposure, factor impact and
        total portfolio impact.
    """
    exposure = portfolio_factor_exposure(
        weights,
        asset_returns,
        factor_returns,
    )

    impact = portfolio_factor_impact(
        weights,
        asset_returns,
        factor_returns,
        factor_shocks,
    )

    return {
        "factor_exposure": exposure,
        "factor_impact": impact,
        "portfolio_return": float(
            impact.sum()
        ),
        "portfolio_loss": float(
            -impact.sum()
        ),
    }


# ============================================================
# STANDARD FACTOR SCENARIOS
# ============================================================

def run_standard_factor_scenarios(
    weights,
    asset_returns,
    factor_returns,
):
    """
    Run standard factor stress scenarios.

    Scenarios
    ---------
    Base
    Equity Crash
    Inflation Shock
    Rate Shock
    Risk-Off
    """
    factor_returns = (
        _validate_factor_returns(
            factor_returns
        )
    )

    factors = factor_returns.columns

    scenarios = {}

    # Base case.
    scenarios["Base"] = np.zeros(
        len(factors)
    )

    # Generic equity shock.
    scenarios["Equity Crash"] = (
        np.full(
            len(factors),
            -0.10,
        )
    )

    # Generic inflation shock.
    scenarios["Inflation Shock"] = (
        np.full(
            len(factors),
            0.05,
        )
    )

    # Generic rate shock.
    scenarios["Rate Shock"] = (
        np.full(
            len(factors),
            -0.05,
        )
    )

    # Generic risk-off shock.
    scenarios["Risk-Off"] = (
        np.full(
            len(factors),
            -0.15,
        )
    )

    results = []

    for name, shocks in scenarios.items():

        result = run_factor_scenario(
            weights,
            asset_returns,
            factor_returns,
            shocks,
        )

        results.append(
            {
                "scenario": name,
                "portfolio_return":
                    result[
                        "portfolio_return"
                    ],
                "portfolio_loss":
                    result[
                        "portfolio_loss"
                    ],
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# FACTOR STRESS SUMMARY
# ============================================================

def factor_stress_summary(
    weights,
    asset_returns,
    factor_returns,
    factor_shocks,
):
    """
    Generate a complete factor stress summary.
    """
    exposure = portfolio_factor_exposure(
        weights,
        asset_returns,
        factor_returns,
    )

    impact = portfolio_factor_impact(
        weights,
        asset_returns,
        factor_returns,
        factor_shocks,
    )

    return {
        "factor_exposure": exposure,
        "factor_impact": impact,
        "portfolio_return":
            float(impact.sum()),
        "portfolio_loss":
            float(-impact.sum()),
    }