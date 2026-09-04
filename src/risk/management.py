"""
POR-Dashboard
Risk Management
==============

Portfolio risk-management controls.

Responsibilities:
    - Position and exposure limits
    - Leverage controls
    - Volatility targeting and scaling
    - Drawdown controls
    - Risk contribution controls
    - Risk budgeting
    - VaR and CVaR controls
    - Beta management
    - Correlation and diversification controls
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_weights(
    weights: np.ndarray | pd.Series,
) -> np.ndarray:
    """Validate portfolio weights."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    if len(weights) == 0:
        raise ValueError("weights cannot be empty")

    if not np.isfinite(weights).all():
        raise ValueError("weights contain invalid values")

    return weights


def _validate_returns(
    returns: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Validate return data."""
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns contain invalid values")

    return returns


def _validate_covariance(
    covariance: np.ndarray | pd.DataFrame,
    n_assets: int,
) -> np.ndarray:
    """Validate covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2:
        raise ValueError("covariance must be two-dimensional")

    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            "covariance dimensions must match weights"
        )

    if not np.isfinite(covariance).all():
        raise ValueError(
            "covariance contains invalid values"
        )

    if not np.allclose(covariance, covariance.T):
        raise ValueError(
            "covariance matrix must be symmetric"
        )

    return covariance


def _validate_positive_parameter(
    value: float,
    name: str,
) -> None:
    """Validate a positive numeric parameter."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )


def _normalize_long_only(
    weights: np.ndarray | pd.Series,
) -> np.ndarray:
    """Normalize long-only weights."""
    weights = _validate_weights(weights)

    if np.any(weights < 0):
        raise ValueError(
            "long-only weights cannot be negative"
        )

    total = weights.sum()

    if np.isclose(total, 0):
        raise ValueError(
            "weights cannot sum to zero"
        )

    return weights / total


# ---------------------------------------------------------------------
# Basic Risk Calculations
# ---------------------------------------------------------------------

def _portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate portfolio volatility."""
    variance = weights @ covariance @ weights

    if variance < 0 and not np.isclose(variance, 0):
        raise ValueError(
            "portfolio variance cannot be negative"
        )

    return float(np.sqrt(max(variance, 0.0)))


def _scale_exposure(
    weights: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale portfolio exposure."""
    return weights * scale


# ---------------------------------------------------------------------
# Position / Exposure Controls
# ---------------------------------------------------------------------

def maximum_position_weight(
    weights: np.ndarray | pd.Series,
    maximum_weight: float,
) -> np.ndarray:
    """Cap individual position weights."""
    _validate_positive_parameter(
        maximum_weight,
        "maximum_weight",
    )

    weights = _validate_weights(weights)

    return np.clip(
        weights,
        -maximum_weight,
        maximum_weight,
    )


def maximum_group_exposure(
    weights: np.ndarray | pd.Series,
    groups: pd.Series | list[str],
    maximum_exposure: float,
) -> np.ndarray:
    """Limit exposure to each group."""
    _validate_positive_parameter(
        maximum_exposure,
        "maximum_exposure",
    )

    weights = _validate_weights(weights)
    groups = pd.Series(groups).reset_index(drop=True)

    if len(weights) != len(groups):
        raise ValueError(
            "groups length must match weights"
        )

    adjusted = weights.copy()

    for group in groups.unique():
        indices = np.where(
            groups.to_numpy() == group
        )[0]

        exposure = np.sum(
            np.abs(adjusted[indices])
        )

        if exposure > maximum_exposure:
            scale = maximum_exposure / exposure
            adjusted[indices] *= scale

    return adjusted


def maximum_industry_exposure(
    weights: np.ndarray | pd.Series,
    industries: pd.Series | list[str],
    maximum_exposure: float,
) -> np.ndarray:
    """Limit exposure to each industry."""
    return maximum_group_exposure(
        weights,
        industries,
        maximum_exposure,
    )


def maximum_sector_exposure(
    weights: np.ndarray | pd.Series,
    sectors: pd.Series | list[str],
    maximum_exposure: float,
) -> np.ndarray:
    """Limit exposure to each sector."""
    return maximum_group_exposure(
        weights,
        sectors,
        maximum_exposure,
    )


def control_gross_net_exposure(
    weights: np.ndarray | pd.Series,
    maximum_gross: float,
    maximum_net: float | None = None,
) -> np.ndarray:
    """Control gross and net portfolio exposure."""
    _validate_positive_parameter(
        maximum_gross,
        "maximum_gross",
    )

    weights = _validate_weights(weights)

    gross = np.sum(np.abs(weights))

    if gross > maximum_gross:
        weights = weights * (
            maximum_gross / gross
        )

    if maximum_net is not None:
        _validate_positive_parameter(
            maximum_net,
            "maximum_net",
        )

        net = abs(np.sum(weights))

        if net > maximum_net and net > 0:
            weights = weights * (
                maximum_net / net
            )

    return weights


def control_leverage(
    weights: np.ndarray | pd.Series,
    maximum_leverage: float,
) -> np.ndarray:
    """Limit gross portfolio leverage."""
    _validate_positive_parameter(
        maximum_leverage,
        "maximum_leverage",
    )

    weights = _validate_weights(weights)

    gross = np.sum(np.abs(weights))

    if gross <= maximum_leverage:
        return weights.copy()

    return weights * (
        maximum_leverage / gross
    )


# ---------------------------------------------------------------------
# Volatility Controls
# ---------------------------------------------------------------------

def volatility_targeting(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    target_volatility: float,
    max_leverage: float | None = None,
) -> np.ndarray:
    """Scale portfolio toward a target volatility."""
    _validate_positive_parameter(
        target_volatility,
        "target_volatility",
    )

    weights = _validate_weights(weights)

    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    current_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if current_volatility == 0:
        return weights.copy()

    scale = (
        target_volatility
        / current_volatility
    )

    if max_leverage is not None:
        _validate_positive_parameter(
            max_leverage,
            "max_leverage",
        )
        scale = min(scale, max_leverage)

    return _scale_exposure(
        weights,
        scale,
    )


def volatility_limit(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    maximum_volatility: float,
) -> np.ndarray:
    """Reduce exposure when volatility exceeds a limit."""
    _validate_positive_parameter(
        maximum_volatility,
        "maximum_volatility",
    )

    weights = _validate_weights(weights)

    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    current_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if current_volatility <= maximum_volatility:
        return weights.copy()

    scale = (
        maximum_volatility
        / current_volatility
    )

    return _scale_exposure(
        weights,
        scale,
    )


def dynamic_volatility_scaling(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    target_volatility: float,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> np.ndarray:
    """Dynamically scale exposure according to volatility."""
    _validate_positive_parameter(
        target_volatility,
        "target_volatility",
    )

    if min_scale < 0:
        raise ValueError(
            "min_scale cannot be negative"
        )

    if max_scale < min_scale:
        raise ValueError(
            "max_scale must be >= min_scale"
        )

    weights = _validate_weights(weights)

    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    current_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if current_volatility == 0:
        scale = max_scale
    else:
        scale = (
            target_volatility
            / current_volatility
        )

        scale = np.clip(
            scale,
            min_scale,
            max_scale,
        )

    return weights * scale


def ewma_volatility(
    returns: pd.DataFrame | np.ndarray,
    span: int = 90,
) -> pd.Series:
    """Calculate EWMA volatility for each asset."""
    if span <= 0:
        raise ValueError(
            "span must be positive"
        )

    returns = _validate_returns(returns)

    return (
        returns
        .ewm(span=span)
        .std()
        .iloc[-1]
    )


def ewma_risk_scaling(
    weights: np.ndarray | pd.Series,
    returns: pd.DataFrame | np.ndarray,
    target_volatility: float,
    span: int = 90,
    max_leverage: float | None = None,
) -> np.ndarray:
    """Scale portfolio exposure using EWMA volatility."""
    _validate_positive_parameter(
        target_volatility,
        "target_volatility",
    )

    weights = _validate_weights(weights)
    returns = _validate_returns(returns)

    if len(weights) != returns.shape[1]:
        raise ValueError(
            "weights must match number of assets"
        )

    asset_volatility = ewma_volatility(
        returns,
        span,
    )

    covariance = returns.cov().to_numpy()

    if not np.isfinite(covariance).all():
        raise ValueError(
            "unable to calculate covariance"
        )

    current_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if current_volatility == 0:
        return weights.copy()

    scale = (
        target_volatility
        / current_volatility
    )

    if max_leverage is not None:
        _validate_positive_parameter(
            max_leverage,
            "max_leverage",
        )
        scale = min(scale, max_leverage)

    return weights * scale


# ---------------------------------------------------------------------
# Drawdown Controls
# ---------------------------------------------------------------------

def calculate_drawdown(
    portfolio_returns: pd.Series | np.ndarray,
) -> pd.Series:
    """Calculate portfolio drawdown from returns."""
    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    )

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    if not np.isfinite(
        returns.to_numpy()
    ).all():
        raise ValueError(
            "portfolio_returns contain invalid values"
        )

    wealth = (1 + returns).cumprod()
    high_water_mark = wealth.cummax()

    return (
        wealth / high_water_mark
    ) - 1


def maximum_drawdown_control(
    weights: np.ndarray | pd.Series,
    portfolio_returns: pd.Series | np.ndarray,
    maximum_drawdown: float,
) -> np.ndarray:
    """Reduce exposure when maximum drawdown is breached."""
    _validate_positive_parameter(
        maximum_drawdown,
        "maximum_drawdown",
    )

    weights = _validate_weights(weights)

    drawdown = calculate_drawdown(
        portfolio_returns
    )

    current_drawdown = abs(
        drawdown.min()
    )

    if current_drawdown <= maximum_drawdown:
        return weights.copy()

    scale = (
        maximum_drawdown
        / current_drawdown
    )

    return weights * scale


def high_water_mark_derisking(
    weights: np.ndarray | pd.Series,
    portfolio_returns: pd.Series | np.ndarray,
    drawdown_threshold: float,
) -> np.ndarray:
    """De-risk portfolio when drawdown exceeds threshold."""
    _validate_positive_parameter(
        drawdown_threshold,
        "drawdown_threshold",
    )

    weights = _validate_weights(weights)

    drawdown = calculate_drawdown(
        portfolio_returns
    )

    current_drawdown = abs(
        drawdown.iloc[-1]
    )

    if current_drawdown <= drawdown_threshold:
        return weights.copy()

    scale = (
        drawdown_threshold
        / current_drawdown
    )

    return weights * min(scale, 1.0)


def trend_derisking(
    weights: np.ndarray | pd.Series,
    portfolio_returns: pd.Series | np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """Reduce exposure when recent trend is negative."""
    if lookback <= 0:
        raise ValueError(
            "lookback must be positive"
        )

    weights = _validate_weights(weights)

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    )

    if len(returns) < lookback:
        return weights.copy()

    cumulative_return = (
        (1 + returns.iloc[-lookback:]).prod()
        - 1
    )

    if cumulative_return >= 0:
        return weights.copy()

    return weights * 0.5


# ---------------------------------------------------------------------
# Risk Contribution Controls
# ---------------------------------------------------------------------

def calculate_risk_contributions(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Calculate percentage risk contribution."""
    weights = _validate_weights(weights)

    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    portfolio_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if portfolio_volatility == 0:
        return np.zeros(len(weights))

    marginal = (
        covariance @ weights
    ) / portfolio_volatility

    component = weights * marginal

    total = component.sum()

    if np.isclose(total, 0):
        return np.zeros(len(weights))

    return component / total


def risk_contribution_constraint(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    maximum_risk_contribution: float,
) -> np.ndarray:
    """Limit excessive individual risk contributions."""
    _validate_positive_parameter(
        maximum_risk_contribution,
        "maximum_risk_contribution",
    )

    weights = _validate_weights(weights)

    contributions = calculate_risk_contributions(
        weights,
        covariance,
    )

    adjusted = weights.copy()

    for i, contribution in enumerate(contributions):
        if contribution > maximum_risk_contribution:
            adjusted[i] *= (
                maximum_risk_contribution
                / contribution
            )

    return adjusted


def risk_budgeting(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    risk_budget: np.ndarray | pd.Series,
) -> np.ndarray:
    """Scale weights toward specified risk budgets."""
    weights = _validate_weights(weights)

    risk_budget = np.asarray(
        risk_budget,
        dtype=float,
    )

    if len(risk_budget) != len(weights):
        raise ValueError(
            "risk_budget must match weights"
        )

    if np.any(risk_budget < 0):
        raise ValueError(
            "risk_budget cannot contain negative values"
        )

    if np.isclose(risk_budget.sum(), 0):
        raise ValueError(
            "risk_budget cannot sum to zero"
        )

    risk_budget = (
        risk_budget
        / risk_budget.sum()
    )

    contributions = calculate_risk_contributions(
        weights,
        covariance,
    )

    if np.isclose(contributions.sum(), 0):
        return weights.copy()

    scaling = np.divide(
        risk_budget,
        np.maximum(
            contributions,
            1e-12,
        ),
    )

    scaling = np.sqrt(scaling)

    return weights * scaling


def risk_concentration(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate concentration of risk contributions."""
    contributions = calculate_risk_contributions(
        weights,
        covariance,
    )

    return float(
        np.sum(contributions ** 2)
    )


# ---------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------

def historical_var(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical Value at Risk."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    return float(
        -np.quantile(
            returns,
            1 - confidence_level,
        )
    )


def parametric_var(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate parametric Gaussian Value at Risk."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    mean = returns.mean()
    std = returns.std()

    from statistics import NormalDist

    z_score = NormalDist().inv_cdf(
        1 - confidence_level
    )

    return float(
        -(mean + z_score * std)
    )


def monte_carlo_var(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    simulations: int = 10_000,
    random_state: int | None = 42,
) -> float:
    """Estimate Value at Risk using Monte Carlo simulation."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    if simulations <= 0:
        raise ValueError(
            "simulations must be positive"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    mean = returns.mean()
    std = returns.std()

    rng = np.random.default_rng(
        random_state
    )

    simulated_returns = rng.normal(
        mean,
        std,
        simulations,
    )

    return float(
        -np.quantile(
            simulated_returns,
            1 - confidence_level,
        )
    )


def var_constraint(
    weights: np.ndarray | pd.Series,
    portfolio_returns: pd.Series | np.ndarray,
    maximum_var: float,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Reduce exposure when VaR exceeds a limit."""
    _validate_positive_parameter(
        maximum_var,
        "maximum_var",
    )

    weights = _validate_weights(weights)

    var = historical_var(
        portfolio_returns,
        confidence_level,
    )

    if var <= maximum_var:
        return weights.copy()

    return weights * (
        maximum_var / var
    )


# ---------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------

def historical_cvar(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical Conditional VaR."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    var_threshold = np.quantile(
        returns,
        1 - confidence_level,
    )

    tail_returns = returns[
        returns <= var_threshold
    ]

    if tail_returns.empty:
        return historical_var(
            returns,
            confidence_level,
        )

    return float(
        -tail_returns.mean()
    )


def parametric_cvar(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Gaussian Conditional VaR."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    mean = returns.mean()
    std = returns.std()

    from statistics import NormalDist

    normal = NormalDist()

    z = normal.inv_cdf(
        1 - confidence_level
    )

    pdf = np.exp(
        -0.5 * z ** 2
    ) / np.sqrt(2 * np.pi)

    cvar = -(
        mean
        - std
        * pdf
        / (1 - confidence_level)
    )

    return float(cvar)


def monte_carlo_cvar(
    portfolio_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    simulations: int = 10_000,
    random_state: int | None = 42,
) -> float:
    """Estimate CVaR using Monte Carlo simulation."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    if simulations <= 0:
        raise ValueError(
            "simulations must be positive"
        )

    returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    mean = returns.mean()
    std = returns.std()

    rng = np.random.default_rng(
        random_state
    )

    simulated_returns = rng.normal(
        mean,
        std,
        simulations,
    )

    threshold = np.quantile(
        simulated_returns,
        1 - confidence_level,
    )

    tail = simulated_returns[
        simulated_returns <= threshold
    ]

    return float(
        -tail.mean()
    )


def cvar_constraint(
    weights: np.ndarray | pd.Series,
    portfolio_returns: pd.Series | np.ndarray,
    maximum_cvar: float,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Reduce exposure when CVaR exceeds a limit."""
    _validate_positive_parameter(
        maximum_cvar,
        "maximum_cvar",
    )

    weights = _validate_weights(weights)

    cvar = historical_cvar(
        portfolio_returns,
        confidence_level,
    )

    if cvar <= maximum_cvar:
        return weights.copy()

    return weights * (
        maximum_cvar / cvar
    )


# ---------------------------------------------------------------------
# Beta Management
# ---------------------------------------------------------------------

def portfolio_beta(
    weights: np.ndarray | pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta."""
    weights = _validate_weights(weights)

    asset_returns = _validate_returns(
        asset_returns
    )

    benchmark_returns = pd.Series(
        benchmark_returns,
        dtype=float,
    )

    if len(weights) != asset_returns.shape[1]:
        raise ValueError(
            "weights must match number of assets"
        )

    aligned = pd.concat(
        [
            asset_returns,
            benchmark_returns.rename(
                "Benchmark"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "no overlapping observations"
        )

    assets = aligned.iloc[:, :-1]
    benchmark = aligned["Benchmark"]

    benchmark_variance = benchmark.var()

    if np.isclose(benchmark_variance, 0):
        raise ValueError(
            "benchmark variance cannot be zero"
        )

    betas = assets.apply(
        lambda column:
        column.cov(benchmark)
        / benchmark_variance
    )

    return float(
        weights @ betas.to_numpy()
    )


def beta_management(
    weights: np.ndarray | pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    maximum_beta: float,
) -> np.ndarray:
    """Reduce portfolio exposure when beta exceeds a limit."""
    _validate_positive_parameter(
        maximum_beta,
        "maximum_beta",
    )

    weights = _validate_weights(weights)

    beta = portfolio_beta(
        weights,
        asset_returns,
        benchmark_returns,
    )

    if abs(beta) <= maximum_beta:
        return weights.copy()

    if np.isclose(beta, 0):
        return weights.copy()

    scale = (
        maximum_beta
        / abs(beta)
    )

    return weights * scale


# ---------------------------------------------------------------------
# Correlation / Diversification
# ---------------------------------------------------------------------

def average_pairwise_correlation(
    returns: pd.DataFrame,
) -> float:
    """Calculate average pairwise correlation."""
    returns = _validate_returns(
        returns
    )

    correlation = returns.corr().to_numpy()

    n_assets = correlation.shape[0]

    if n_assets < 2:
        return 0.0

    values = correlation[
        np.triu_indices(
            n_assets,
            k=1,
        )
    ]

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.mean(values)
    )


def diversification_ratio(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate portfolio diversification ratio."""
    weights = _validate_weights(weights)

    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    asset_volatility = np.sqrt(
        np.maximum(
            np.diag(covariance),
            0,
        )
    )

    portfolio_volatility = _portfolio_volatility(
        weights,
        covariance,
    )

    if portfolio_volatility == 0:
        return 0.0

    return float(
        np.sum(
            np.abs(weights)
            * asset_volatility
        )
        / portfolio_volatility
    )


def effective_number_of_assets(
    weights: np.ndarray | pd.Series,
) -> float:
    """Calculate effective number of assets."""
    weights = _validate_weights(weights)

    absolute_weights = np.abs(weights)

    total = absolute_weights.sum()

    if np.isclose(total, 0):
        return 0.0

    normalized = (
        absolute_weights / total
    )

    concentration = np.sum(
        normalized ** 2
    )

    if np.isclose(concentration, 0):
        return 0.0

    return float(
        1 / concentration
    )


def correlation_diversification_control(
    weights: np.ndarray | pd.Series,
    returns: pd.DataFrame,
    maximum_correlation: float,
) -> np.ndarray:
    """Reduce exposure when average correlation is too high."""
    if not -1 <= maximum_correlation <= 1:
        raise ValueError(
            "maximum_correlation must be between -1 and 1"
        )

    weights = _validate_weights(weights)

    correlation = average_pairwise_correlation(
        returns
    )

    if correlation <= maximum_correlation:
        return weights.copy()

    scale = (
        maximum_correlation
        / correlation
    )

    return weights * min(scale, 1.0)


# ---------------------------------------------------------------------
# Engine Registry
# ---------------------------------------------------------------------

def list_risk_management_engines() -> list[str]:
    """Return available risk-management engines."""
    return [
        "maximum_position_weight",
        "maximum_group_exposure",
        "maximum_industry_exposure",
        "maximum_sector_exposure",
        "control_gross_net_exposure",
        "control_leverage",
        "volatility_targeting",
        "volatility_limit",
        "dynamic_volatility_scaling",
        "ewma_volatility",
        "ewma_risk_scaling",
        "calculate_drawdown",
        "maximum_drawdown_control",
        "high_water_mark_derisking",
        "trend_derisking",
        "calculate_risk_contributions",
        "risk_contribution_constraint",
        "risk_budgeting",
        "risk_concentration",
        "historical_var",
        "parametric_var",
        "monte_carlo_var",
        "var_constraint",
        "historical_cvar",
        "parametric_cvar",
        "monte_carlo_cvar",
        "cvar_constraint",
        "portfolio_beta",
        "beta_management",
        "average_pairwise_correlation",
        "diversification_ratio",
        "effective_number_of_assets",
        "correlation_diversification_control",
    ]


def get_risk_management_engine(
    name: str,
):
    """Return a risk-management function by name."""
    engines = {
        engine_name: globals()[engine_name]
        for engine_name
        in list_risk_management_engines()
    }

    if name not in engines:
        raise ValueError(
            f"Unknown risk management engine: {name}"
        )

    return engines[name]