"""
POR-Dashboard
Stress Testing Engine
=====================

Portfolio stress-testing framework.

The engine supports:

1. Market Crash
2. Severe Crash
3. Volatility Spike
4. Interest Rate Shock
5. Sector Shock
6. Single-Stock Crash
7. Correlation Spike
8. Liquidity / Exposure Shock
9. Historical Crisis
10. Custom Shock

The module is intentionally independent of Streamlit.
It returns dictionaries/DataFrames that can be consumed by
the dashboard, notebooks, tests, or other applications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_weights(
    weights: pd.Series | dict,
) -> pd.Series:
    """Validate portfolio weights."""

    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    elif not isinstance(weights, pd.Series):
        weights = pd.Series(
            weights,
            dtype=float,
        )

    weights = weights.astype(float)

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    if not np.isfinite(
        weights.values
    ).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    if (weights < 0).any():
        raise ValueError(
            "weights cannot contain negative values."
        )

    total = weights.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "weights must have a positive finite sum."
        )

    weights = weights / total

    return weights


def _validate_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Validate return matrix."""

    if not isinstance(
        returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    result = returns.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if result.isna().any().any():
        raise ValueError(
            "returns contain missing or non-numeric values."
        )

    if not np.isfinite(
        result.values
    ).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    return result


def _validate_covariance(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Validate covariance matrix."""

    if not isinstance(
        covariance,
        pd.DataFrame,
    ):
        raise TypeError(
            "covariance must be a pandas DataFrame."
        )

    if covariance.empty:
        raise ValueError(
            "covariance cannot be empty."
        )

    if (
        covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance must be square."
        )

    covariance = covariance.astype(float)

    if not np.isfinite(
        covariance.values
    ).all():
        raise ValueError(
            "covariance contains non-finite values."
        )

    if not np.allclose(
        covariance.values,
        covariance.values.T,
        atol=1e-10,
    ):
        raise ValueError(
            "covariance must be symmetric."
        )

    return covariance


def _validate_probability(
    value: float,
    name: str,
) -> float:
    """Validate probability/confidence parameter."""

    value = float(value)

    if not 0.0 < value < 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _portfolio_return(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> float:
    """Calculate weighted portfolio return."""

    aligned = pd.concat(
        [
            weights.rename("Weight"),
            asset_returns.rename("Return"),
        ],
        axis=1,
    ).fillna(0.0)

    return float(
        (
            aligned["Weight"]
            * aligned["Return"]
        ).sum()
    )


# ============================================================
# CORE STRESS RESULT
# ============================================================

def _build_stress_result(
    scenario: str,
    weights: pd.Series,
    stressed_returns: pd.Series,
    base_nav: float = 1.0,
    metadata: dict | None = None,
) -> dict:
    """
    Build standardized stress-test output.
    """

    weights = _validate_weights(
        weights
    )

    stressed_returns = stressed_returns.reindex(
        weights.index
    ).fillna(0.0)

    asset_losses = (
        weights
        * stressed_returns
    )

    portfolio_return = float(
        asset_losses.sum()
    )

    stressed_nav = (
        float(base_nav)
        * (1.0 + portfolio_return)
    )

    result = {
        "scenario": scenario,
        "portfolio_return": portfolio_return,
        "portfolio_loss": portfolio_return,
        "stressed_nav": stressed_nav,
        "asset_returns": stressed_returns,
        "asset_losses": asset_losses,
        "weights": weights.copy(),
    }

    if metadata:
        result.update(
            metadata
        )

    return result


# ============================================================
# 1. MARKET CRASH
# ============================================================

def market_crash(
    weights: pd.Series | dict,
    crash_return: float = -0.20,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply the same market shock to every asset.

    Example
    -------
    crash_return=-0.20

    means every asset falls by 20%.
    """

    weights = _validate_weights(
        weights
    )

    crash_return = float(
        crash_return
    )

    if crash_return > 0:
        raise ValueError(
            "crash_return should be zero or negative."
        )

    stressed_returns = pd.Series(
        crash_return,
        index=weights.index,
        dtype=float,
    )

    return _build_stress_result(
        scenario="Market Crash",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "shock": crash_return,
        },
    )


# ============================================================
# 2. SEVERE CRASH
# ============================================================

def severe_crash(
    weights: pd.Series | dict,
    crash_return: float = -0.40,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply a severe market-wide shock.
    """

    weights = _validate_weights(
        weights
    )

    crash_return = float(
        crash_return
    )

    if crash_return > 0:
        raise ValueError(
            "crash_return should be zero or negative."
        )

    stressed_returns = pd.Series(
        crash_return,
        index=weights.index,
        dtype=float,
    )

    return _build_stress_result(
        scenario="Severe Crash",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "shock": crash_return,
        },
    )


# ============================================================
# 3. VOLATILITY SPIKE
# ============================================================

def volatility_spike(
    weights: pd.Series | dict,
    returns: pd.DataFrame,
    multiplier: float = 2.0,
    horizon_days: int = 1,
    random_state: int = 42,
    base_nav: float = 1.0,
) -> dict:
    """
    Stress portfolio using amplified historical return shocks.

    The latest historical return is multiplied by the supplied
    volatility multiplier.

    This is deliberately deterministic and reproducible.
    """

    weights = _validate_weights(
        weights
    )

    returns = _validate_returns(
        returns
    )

    if multiplier <= 0:
        raise ValueError(
            "multiplier must be positive."
        )

    if horizon_days < 1:
        raise ValueError(
            "horizon_days must be at least 1."
        )

    common_assets = [
        asset
        for asset in weights.index
        if asset in returns.columns
    ]

    if not common_assets:
        raise ValueError(
            "No common assets between weights and returns."
        )

    historical = returns[
        common_assets
    ]

    # Use the latest observed return as the shock seed.
    latest_returns = historical.iloc[-1]

    stressed_returns = (
        latest_returns
        * float(multiplier)
    )

    stressed_returns = stressed_returns.clip(
        lower=-1.0
    )

    return _build_stress_result(
        scenario="Volatility Spike",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "multiplier": float(
                multiplier
            ),
            "horizon_days": int(
                horizon_days
            ),
            "random_state": random_state,
        },
    )


# ============================================================
# 4. INTEREST RATE SHOCK
# ============================================================

def interest_rate_shock(
    weights: pd.Series | dict,
    shock: float = -0.05,
    rate_sensitive_assets: list[str] | None = None,
    asset_sensitivities: pd.Series | dict | None = None,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply an interest-rate shock.

    If asset_sensitivities are supplied:

        stressed return
        = shock × sensitivity

    Otherwise, assets listed in rate_sensitive_assets receive
    the full shock and all other assets receive zero.
    """

    weights = _validate_weights(
        weights
    )

    shock = float(
        shock
    )

    if shock > 0:
        raise ValueError(
            "shock should be zero or negative."
        )

    stressed_returns = pd.Series(
        0.0,
        index=weights.index,
    )

    if asset_sensitivities is not None:

        if isinstance(
            asset_sensitivities,
            dict,
        ):
            asset_sensitivities = pd.Series(
                asset_sensitivities,
                dtype=float,
            )

        asset_sensitivities = (
            pd.Series(
                asset_sensitivities,
                dtype=float,
            )
        )

        sensitivities = (
            asset_sensitivities
            .reindex(
                weights.index
            )
            .fillna(0.0)
        )

        stressed_returns = (
            shock
            * sensitivities
        )

    elif rate_sensitive_assets is not None:

        for asset in rate_sensitive_assets:

            if asset in stressed_returns.index:
                stressed_returns.loc[asset] = (
                    shock
                )

    else:

        stressed_returns[:] = shock

    stressed_returns = stressed_returns.clip(
        lower=-1.0
    )

    return _build_stress_result(
        scenario="Interest Rate Shock",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "shock": shock,
            "rate_sensitive_assets": (
                rate_sensitive_assets
            ),
        },
    )


# ============================================================
# 5. SECTOR SHOCK
# ============================================================

def sector_shock(
    weights: pd.Series | dict,
    sector_data: pd.Series | dict,
    sector: str,
    shock: float = -0.25,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply a shock to all assets belonging to one sector.
    """

    weights = _validate_weights(
        weights
    )

    if isinstance(
        sector_data,
        dict,
    ):
        sector_data = pd.Series(
            sector_data
        )

    sector_data = pd.Series(
        sector_data
    )

    sector = str(
        sector
    )

    shock = float(
        shock
    )

    if shock > 0:
        raise ValueError(
            "shock should be zero or negative."
        )

    stressed_returns = pd.Series(
        0.0,
        index=weights.index,
    )

    for asset in weights.index:

        if (
            asset in sector_data.index
            and str(
                sector_data.loc[asset]
            ) == sector
        ):

            stressed_returns.loc[asset] = (
                shock
            )

    return _build_stress_result(
        scenario="Sector Shock",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "sector": sector,
            "shock": shock,
        },
    )


# ============================================================
# 6. SINGLE STOCK CRASH
# ============================================================

def single_stock_crash(
    weights: pd.Series | dict,
    ticker: str,
    crash_return: float = -0.50,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply a crash to one stock only.
    """

    weights = _validate_weights(
        weights
    )

    if ticker not in weights.index:
        raise ValueError(
            f"{ticker} is not present in portfolio."
        )

    crash_return = float(
        crash_return
    )

    if crash_return > 0:
        raise ValueError(
            "crash_return should be zero or negative."
        )

    stressed_returns = pd.Series(
        0.0,
        index=weights.index,
    )

    stressed_returns.loc[ticker] = (
        crash_return
    )

    return _build_stress_result(
        scenario="Single Stock Crash",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "ticker": ticker,
            "shock": crash_return,
        },
    )


# ============================================================
# 7. CORRELATION SPIKE
# ============================================================

def correlation_spike(
    weights: pd.Series | dict,
    covariance: pd.DataFrame,
    correlation_target: float = 0.90,
    shock_scale: float = 1.0,
    base_nav: float = 1.0,
) -> dict:
    """
    Stress portfolio correlation structure.

    The covariance matrix is transformed so that all off-diagonal
    correlations move toward correlation_target while individual
    asset volatilities remain unchanged.

    The function returns stressed portfolio volatility rather than
    a single deterministic return shock.
    """

    weights = _validate_weights(
        weights
    )

    covariance = _validate_covariance(
        covariance
    )

    correlation_target = float(
        correlation_target
    )

    if not -1.0 <= correlation_target <= 1.0:
        raise ValueError(
            "correlation_target must be between -1 and 1."
        )

    assets = [
        asset
        for asset in weights.index
        if asset in covariance.columns
    ]

    if not assets:
        raise ValueError(
            "No common assets between weights and covariance."
        )

    covariance = covariance.loc[
        assets,
        assets,
    ]

    weights_aligned = (
        weights.reindex(
            assets
        )
    )

    volatility = np.sqrt(
        np.maximum(
            np.diag(
                covariance.values
            ),
            0.0,
        )
    )

    stressed_correlation = np.full(
        (
            len(assets),
            len(assets),
        ),
        correlation_target,
        dtype=float,
    )

    np.fill_diagonal(
        stressed_correlation,
        1.0,
    )

    stressed_covariance = (
        np.outer(
            volatility,
            volatility,
        )
        * stressed_correlation
    )

    portfolio_variance = (
        weights_aligned.values
        @ stressed_covariance
        @ weights_aligned.values
    )

    stressed_volatility = (
        np.sqrt(
            max(
                portfolio_variance,
                0.0,
            )
        )
        * float(shock_scale)
    )

    original_variance = (
        weights_aligned.values
        @ covariance.loc[
            assets,
            assets,
        ].values
        @ weights_aligned.values
    )

    original_volatility = np.sqrt(
        max(
            original_variance,
            0.0,
        )
    )

    volatility_change = (
        stressed_volatility
        - original_volatility
    )

    return {
        "scenario": "Correlation Spike",
        "portfolio_return": np.nan,
        "portfolio_loss": np.nan,
        "stressed_nav": np.nan,
        "weights": weights_aligned,
        "original_volatility": float(
            original_volatility
        ),
        "stressed_volatility": float(
            stressed_volatility
        ),
        "volatility_change": float(
            volatility_change
        ),
        "correlation_target": (
            correlation_target
        ),
        "stressed_covariance": pd.DataFrame(
            stressed_covariance,
            index=assets,
            columns=assets,
        ),
    }


# ============================================================
# 8. LIQUIDITY / EXPOSURE SHOCK
# ============================================================

def liquidity_exposure_shock(
    weights: pd.Series | dict,
    liquidity_scaling: float = 0.50,
    affected_assets: list[str] | None = None,
    base_nav: float = 1.0,
) -> dict:
    """
    Model reduced effective exposure during a liquidity shock.

    affected_assets have their effective exposure scaled by
    liquidity_scaling.

    The remaining exposure is treated as cash.
    """

    weights = _validate_weights(
        weights
    )

    liquidity_scaling = float(
        liquidity_scaling
    )

    if not 0.0 < liquidity_scaling <= 1.0:
        raise ValueError(
            "liquidity_scaling must be between 0 and 1."
        )

    if affected_assets is None:
        affected_assets = list(
            weights.index
        )

    effective_weights = weights.copy()

    for asset in affected_assets:

        if asset in effective_weights.index:

            effective_weights.loc[asset] *= (
                liquidity_scaling
            )

    removed_exposure = (
        weights
        - effective_weights
    )

    return {
        "scenario": "Liquidity / Exposure Shock",
        "portfolio_return": np.nan,
        "portfolio_loss": np.nan,
        "stressed_nav": np.nan,
        "original_weights": weights,
        "effective_weights": effective_weights,
        "removed_exposure": removed_exposure,
        "effective_gross_exposure": float(
            effective_weights.sum()
        ),
        "cash_exposure": float(
            removed_exposure.sum()
        ),
        "liquidity_scaling": (
            liquidity_scaling
        ),
    }


# ============================================================
# 9. HISTORICAL CRISIS
# ============================================================

def historical_crisis(
    weights: pd.Series | dict,
    returns: pd.DataFrame,
    crisis_start: str | pd.Timestamp,
    crisis_end: str | pd.Timestamp,
    base_nav: float = 1.0,
) -> dict:
    """
    Replay an actual historical crisis period.

    Returns both:
        - crisis-period portfolio return
        - worst single-period portfolio shock
        - crisis NAV path
    """

    weights = _validate_weights(
        weights
    )

    returns = _validate_returns(
        returns
    )

    start = pd.Timestamp(
        crisis_start
    )

    end = pd.Timestamp(
        crisis_end
    )

    if start > end:
        raise ValueError(
            "crisis_start must be before crisis_end."
        )

    selected_assets = [
        asset
        for asset in weights.index
        if asset in returns.columns
    ]

    if not selected_assets:
        raise ValueError(
            "No common assets between weights and returns."
        )

    crisis_returns = returns.loc[
        start:end,
        selected_assets,
    ].copy()

    if crisis_returns.empty:
        raise ValueError(
            "No return observations exist in the crisis period."
        )

    aligned_weights = (
        weights.reindex(
            selected_assets
        )
    )

    portfolio_returns = (
        crisis_returns
        .dot(
            aligned_weights
        )
    )

    crisis_nav = (
        float(base_nav)
        * (
            1.0
            + portfolio_returns
        ).cumprod()
    )

    crisis_return = (
        crisis_nav.iloc[-1]
        / float(base_nav)
        - 1.0
    )

    running_max = (
        crisis_nav.cummax()
    )

    drawdown = (
        crisis_nav
        / running_max
        - 1.0
    )

    worst_period = (
        portfolio_returns.min()
    )

    worst_period_date = (
        portfolio_returns.idxmin()
    )

    return {
        "scenario": "Historical Crisis",
        "portfolio_return": float(
            crisis_return
        ),
        "portfolio_loss": float(
            crisis_return
        ),
        "stressed_nav": float(
            crisis_nav.iloc[-1]
        ),
        "crisis_start": start,
        "crisis_end": end,
        "worst_period_return": float(
            worst_period
        ),
        "worst_period_date": (
            worst_period_date
        ),
        "maximum_drawdown": float(
            drawdown.min()
        ),
        "portfolio_returns": portfolio_returns,
        "crisis_nav": crisis_nav,
        "drawdown": drawdown,
    }


# ============================================================
# 10. CUSTOM SHOCK
# ============================================================

def custom_shock(
    weights: pd.Series | dict,
    shocks: pd.Series | dict,
    base_nav: float = 1.0,
) -> dict:
    """
    Apply user-defined return shocks to individual assets.

    Example
    -------
    {
        "RELIANCE": -0.20,
        "TCS": -0.10,
        "INFY": 0.05
    }
    """

    weights = _validate_weights(
        weights
    )

    if isinstance(
        shocks,
        dict,
    ):
        shocks = pd.Series(
            shocks,
            dtype=float,
        )

    shocks = pd.Series(
        shocks,
        dtype=float,
    )

    if not np.isfinite(
        shocks.values
    ).all():
        raise ValueError(
            "shocks contain non-finite values."
        )

    if (shocks < -1.0).any():
        raise ValueError(
            "Asset shock cannot be below -100%."
        )

    stressed_returns = (
        shocks.reindex(
            weights.index
        ).fillna(0.0)
    )

    return _build_stress_result(
        scenario="Custom Shock",
        weights=weights,
        stressed_returns=stressed_returns,
        base_nav=base_nav,
        metadata={
            "shocks": stressed_returns.copy(),
        },
    )


# ============================================================
# STRESS SUMMARY
# ============================================================

def stress_summary(
    stress_result: dict,
) -> pd.Series:
    """
    Convert a stress result into a dashboard-friendly summary.
    """

    if not isinstance(
        stress_result,
        dict,
    ):
        raise TypeError(
            "stress_result must be a dictionary."
        )

    summary = {
        "Scenario": stress_result.get(
            "scenario"
        ),
        "Portfolio Return": stress_result.get(
            "portfolio_return"
        ),
        "Portfolio Loss": stress_result.get(
            "portfolio_loss"
        ),
        "Stressed NAV": stress_result.get(
            "stressed_nav"
        ),
    }

    if (
        "maximum_drawdown"
        in stress_result
    ):
        summary[
            "Maximum Drawdown"
        ] = stress_result[
            "maximum_drawdown"
        ]

    if (
        "original_volatility"
        in stress_result
    ):
        summary[
            "Original Volatility"
        ] = stress_result[
            "original_volatility"
        ]

    if (
        "stressed_volatility"
        in stress_result
    ):
        summary[
            "Stressed Volatility"
        ] = stress_result[
            "stressed_volatility"
        ]

    return pd.Series(
        summary,
        dtype=object,
    )


# ============================================================
# ASSET LOSS CONTRIBUTION
# ============================================================

def asset_loss_contribution(
    stress_result: dict,
) -> pd.Series:
    """
    Return each asset's contribution to portfolio stress loss.
    """

    if (
        "asset_losses"
        not in stress_result
    ):
        raise ValueError(
            "Stress result does not contain asset_losses."
        )

    losses = stress_result[
        "asset_losses"
    ]

    if isinstance(
        losses,
        dict,
    ):
        losses = pd.Series(
            losses,
            dtype=float,
        )

    return pd.Series(
        losses,
        dtype=float,
    ).sort_values(
        ascending=True
    )


# ============================================================
# MULTI-SCENARIO ANALYSIS
# ============================================================

def compare_stress_scenarios(
    results: dict[str, dict],
) -> pd.DataFrame:
    """
    Compare multiple stress-test results.

    Parameters
    ----------
    results : dict
        Scenario name -> stress result.
    """

    if not results:
        raise ValueError(
            "results cannot be empty."
        )

    rows = []

    for scenario, result in results.items():

        row = {
            "Scenario": scenario,
            "Portfolio Return": result.get(
                "portfolio_return",
                np.nan,
            ),
            "Portfolio Loss": result.get(
                "portfolio_loss",
                np.nan,
            ),
            "Stressed NAV": result.get(
                "stressed_nav",
                np.nan,
            ),
            "Maximum Drawdown": result.get(
                "maximum_drawdown",
                np.nan,
            ),
            "Original Volatility": result.get(
                "original_volatility",
                np.nan,
            ),
            "Stressed Volatility": result.get(
                "stressed_volatility",
                np.nan,
            ),
        }

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .set_index("Scenario")
    )


# ============================================================
# SCENARIO REGISTRY
# ============================================================

STRESS_SCENARIOS = {
    "Market Crash": market_crash,
    "Severe Crash": severe_crash,
    "Volatility Spike": volatility_spike,
    "Interest Rate Shock": interest_rate_shock,
    "Sector Shock": sector_shock,
    "Single Stock Crash": single_stock_crash,
    "Correlation Spike": correlation_spike,
    "Liquidity / Exposure Shock": liquidity_exposure_shock,
    "Historical Crisis": historical_crisis,
    "Custom Shock": custom_shock,
}


# ============================================================
# REGISTRY FUNCTIONS
# ============================================================

def list_stress_scenarios() -> list[str]:
    """
    Return available stress scenarios.
    """

    return list(
        STRESS_SCENARIOS.keys()
    )


def get_stress_scenario(
    name: str,
):
    """
    Return stress-test function by name.
    """

    if name not in STRESS_SCENARIOS:
        raise ValueError(
            f"Unknown stress scenario: {name}"
        )

    return STRESS_SCENARIOS[
        name
    ]


# ============================================================
# GENERIC SCENARIO RUNNER
# ============================================================

def run_stress_test(
    scenario: str,
    weights: pd.Series | dict,
    **kwargs,
) -> dict:
    """
    Run a stress scenario through the registry.

    Example
    -------
    result = run_stress_test(
        "Market Crash",
        weights,
        crash_return=-0.25,
    )
    """

    engine = get_stress_scenario(
        scenario
    )

    return engine(
        weights,
        **kwargs,
    )