from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    returns = pd.DataFrame(returns, dtype=float)

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if not np.isfinite(returns.values).all():
        raise ValueError("returns contain non-finite values.")

    return returns


def _validate_weights(weights, assets):
    weights = pd.Series(weights, dtype=float)

    if weights.empty:
        raise ValueError("weights cannot be empty.")

    if not np.isfinite(weights.values).all():
        raise ValueError("weights contain non-finite values.")

    if isinstance(weights.index, pd.RangeIndex):
        if len(weights) != len(assets):
            raise ValueError(
                "weights length must match number of assets."
            )
        weights.index = assets
    else:
        missing = [
            asset for asset in assets
            if asset not in weights.index
        ]

        if missing:
            raise ValueError(
                f"weights missing assets: {missing}"
            )

        weights = weights.loc[assets]

    if (weights.values < 0).any():
        raise ValueError(
            "negative weights are not supported."
        )

    total = float(weights.sum())

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    return weights / total


def _validate_positive_series(values, name):
    values = pd.Series(values, dtype=float)

    if values.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(values.values).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if (values.values <= 0).any():
        raise ValueError(
            f"{name} must contain positive values."
        )

    return values


# ============================================================
# BID-ASK SPREAD
# ============================================================

def apply_spread_shock(
    returns,
    spreads,
    spread_multiplier=2.0,
):
    """
    Apply a transaction-cost impact caused by
    bid-ask spread widening.

    Returns are reduced by half the stressed
    spread because the portfolio pays the spread
    when crossing the market.
    """
    returns = _validate_returns(returns)

    spreads = _validate_positive_series(
        spreads,
        "spreads",
    )

    if spread_multiplier <= 0:
        raise ValueError(
            "spread_multiplier must be positive."
        )

    spreads = spreads.reindex(
        returns.columns
    )

    if spreads.isna().any():
        raise ValueError(
            "spreads must contain every asset."
        )

    transaction_cost = (
        spreads * spread_multiplier / 2.0
    )

    stressed_returns = (
        returns - transaction_cost
    )

    return stressed_returns


def spread_cost(
    weights,
    spreads,
    spread_multiplier=2.0,
):
    """
    Calculate weighted portfolio transaction cost
    from bid-ask spreads.
    """
    spreads = _validate_positive_series(
        spreads,
        "spreads",
    )

    weights = _validate_weights(
        weights,
        spreads.index,
    )

    if spread_multiplier <= 0:
        raise ValueError(
            "spread_multiplier must be positive."
        )

    costs = (
        spreads
        * spread_multiplier
        / 2.0
    )

    return float(
        weights @ costs
    )


# ============================================================
# MARKET IMPACT
# ============================================================

def calculate_market_impact(
    weights,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Estimate market impact from portfolio exposure.

    Impact increases when liquidity decreases.

    liquidity should be a positive liquidity measure
    such as average traded value or normalized depth.
    """
    liquidity = _validate_positive_series(
        liquidity,
        "liquidity",
    )

    weights = _validate_weights(
        weights,
        liquidity.index,
    )

    if impact_coefficient < 0:
        raise ValueError(
            "impact_coefficient cannot be negative."
        )

    asset_impact = (
        impact_coefficient
        * weights.abs()
        / np.sqrt(liquidity)
    )

    asset_impact.name = "market_impact"

    return asset_impact


def portfolio_market_impact(
    weights,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Calculate total portfolio market impact.
    """
    impact = calculate_market_impact(
        weights,
        liquidity,
        impact_coefficient,
    )

    return float(
        impact.sum()
    )


# ============================================================
# LIQUIDITY SHOCK
# ============================================================

def apply_liquidity_shock(
    returns,
    weights,
    liquidity,
    liquidity_multiplier=0.50,
    impact_coefficient=0.10,
):
    """
    Apply a liquidity deterioration scenario.

    liquidity_multiplier=0.50 means available liquidity
    falls to 50% of its original level.
    """
    returns = _validate_returns(
        returns
    )

    liquidity = _validate_positive_series(
        liquidity,
        "liquidity",
    )

    if not 0 < liquidity_multiplier <= 1:
        raise ValueError(
            "liquidity_multiplier must be "
            "between 0 and 1."
        )

    liquidity = liquidity.reindex(
        returns.columns
    )

    if liquidity.isna().any():
        raise ValueError(
            "liquidity must contain every asset."
        )

    stressed_liquidity = (
        liquidity
        * liquidity_multiplier
    )

    impact = calculate_market_impact(
        weights,
        stressed_liquidity,
        impact_coefficient,
    )

    stressed_returns = (
        returns - impact
    )

    return stressed_returns


# ============================================================
# LIQUIDATION COST
# ============================================================

def liquidation_cost(
    weights,
    liquidity,
    liquidation_fraction=1.0,
    impact_coefficient=0.10,
):
    """
    Estimate the cost of liquidating a fraction
    of a portfolio.
    """
    if not 0 < liquidation_fraction <= 1:
        raise ValueError(
            "liquidation_fraction must be "
            "between 0 and 1."
        )

    liquidity = _validate_positive_series(
        liquidity,
        "liquidity",
    )

    weights = _validate_weights(
        weights,
        liquidity.index,
    )

    if impact_coefficient < 0:
        raise ValueError(
            "impact_coefficient cannot be negative."
        )

    liquidation_weights = (
        weights * liquidation_fraction
    )

    impact = (
        impact_coefficient
        * liquidation_weights
        / np.sqrt(liquidity)
    )

    return float(
        impact.sum()
    )


# ============================================================
# LIQUIDITY-ADJUSTED RETURN
# ============================================================

def liquidity_adjusted_portfolio_return(
    weights,
    returns,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Calculate portfolio return after estimated
    liquidity costs.
    """
    returns = _validate_returns(
        returns
    )

    weights = _validate_weights(
        weights,
        returns.columns,
    )

    liquidity = _validate_positive_series(
        liquidity,
        "liquidity",
    )

    liquidity = liquidity.reindex(
        returns.columns
    )

    if liquidity.isna().any():
        raise ValueError(
            "liquidity must contain every asset."
        )

    portfolio_returns = (
        returns @ weights
    )

    impact = calculate_market_impact(
        weights,
        liquidity,
        impact_coefficient,
    )

    total_impact = float(
        impact.sum()
    )

    return float(
        portfolio_returns.mean()
        - total_impact
    )


# ============================================================
# STANDARD LIQUIDITY SCENARIOS
# ============================================================

def run_liquidity_scenario(
    weights,
    returns,
    liquidity,
    liquidity_multiplier,
    impact_coefficient=0.10,
):
    """
    Run one liquidity stress scenario.
    """
    returns = _validate_returns(
        returns
    )

    liquidity = _validate_positive_series(
        liquidity,
        "liquidity",
    )

    if not 0 < liquidity_multiplier <= 1:
        raise ValueError(
            "liquidity_multiplier must be "
            "between 0 and 1."
        )

    stressed_returns = apply_liquidity_shock(
        returns,
        weights,
        liquidity,
        liquidity_multiplier,
        impact_coefficient,
    )

    weights = _validate_weights(
        weights,
        returns.columns,
    )

    portfolio_return = float(
        (stressed_returns @ weights).mean()
    )

    return {
        "liquidity_multiplier":
            float(liquidity_multiplier),
        "portfolio_return":
            portfolio_return,
        "portfolio_loss":
            -portfolio_return,
    }


def run_standard_liquidity_scenarios(
    weights,
    returns,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Run standard liquidity deterioration scenarios.
    """
    scenarios = {
        "Normal Liquidity": 1.00,
        "Mild Liquidity Stress": 0.75,
        "Moderate Liquidity Stress": 0.50,
        "Severe Liquidity Stress": 0.25,
        "Extreme Liquidity Stress": 0.10,
    }

    results = []

    for name, multiplier in scenarios.items():

        result = run_liquidity_scenario(
            weights,
            returns,
            liquidity,
            multiplier,
            impact_coefficient,
        )

        result["scenario"] = name

        results.append(result)

    return pd.DataFrame(results)[
        [
            "scenario",
            "liquidity_multiplier",
            "portfolio_return",
            "portfolio_loss",
        ]
    ]


# ============================================================
# LIQUIDITY STRESS SUMMARY
# ============================================================

def liquidity_stress_summary(
    weights,
    returns,
    liquidity,
    impact_coefficient=0.10,
):
    """
    Produce a complete liquidity stress summary.
    """
    scenarios = run_standard_liquidity_scenarios(
        weights,
        returns,
        liquidity,
        impact_coefficient,
    )

    worst_index = (
        scenarios[
            "portfolio_return"
        ].idxmin()
    )

    worst_case = (
        scenarios.loc[
            worst_index
        ].to_dict()
    )

    return {
        "scenarios": scenarios,
        "worst_case": worst_case,
        "portfolio_market_impact":
            portfolio_market_impact(
                weights,
                liquidity,
                impact_coefficient,
            ),
        "liquidation_cost":
            liquidation_cost(
                weights,
                liquidity,
                liquidation_fraction=1.0,
                impact_coefficient=impact_coefficient,
            ),
    }