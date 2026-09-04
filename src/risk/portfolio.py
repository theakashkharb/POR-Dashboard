"""
POR-Dashboard
Portfolio Risk
==============

Portfolio-level risk calculations.

Responsibilities:
    - Portfolio variance and volatility
    - Marginal and component risk
    - Risk contribution
    - Concentration
    - Diversification
    - Beta and correlation
    - Industry exposure
    - Portfolio risk summaries
    - Portfolio comparisons
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
    """Validate and return portfolio weights as a 1D numpy array."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    if len(weights) == 0:
        raise ValueError("weights cannot be empty")

    if not np.isfinite(weights).all():
        raise ValueError("weights contain invalid values")

    return weights


def _validate_covariance(
    covariance: np.ndarray | pd.DataFrame,
    n_assets: int,
) -> np.ndarray:
    """Validate covariance matrix dimensions and values."""
    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2:
        raise ValueError("covariance must be two-dimensional")

    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            "covariance dimensions must match number of weights"
        )

    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains invalid values")

    if not np.allclose(covariance, covariance.T):
        raise ValueError("covariance matrix must be symmetric")

    return covariance


def _validate_returns(
    returns: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Validate return data and return it as a DataFrame."""
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    numeric_data = returns.select_dtypes(include=np.number)

    if numeric_data.shape != returns.shape:
        raise ValueError(
            "returns must contain only numeric columns"
        )

    if not np.isfinite(numeric_data.to_numpy()).all():
        raise ValueError("returns contain invalid values")

    return returns


# ---------------------------------------------------------------------
# Basic Portfolio Risk
# ---------------------------------------------------------------------

def portfolio_variance(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate portfolio variance."""
    weights = _validate_weights(weights)
    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    variance = weights @ covariance @ weights

    if variance < 0 and not np.isclose(variance, 0):
        raise ValueError(
            "portfolio variance cannot be negative"
        )

    return float(max(variance, 0.0))


def portfolio_volatility(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate portfolio volatility."""
    return float(
        np.sqrt(
            portfolio_variance(
                weights,
                covariance,
            )
        )
    )


# ---------------------------------------------------------------------
# Marginal / Component Risk
# ---------------------------------------------------------------------

def marginal_risk(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Calculate marginal contribution to portfolio volatility.

    MCR_i = (Σw)_i / σ_p
    """
    weights = _validate_weights(weights)
    covariance = _validate_covariance(
        covariance,
        len(weights),
    )

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    if volatility == 0:
        return np.zeros(len(weights))

    return (covariance @ weights) / volatility


def component_risk(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Calculate component contribution to portfolio volatility.

    Component Risk_i = w_i × Marginal Risk_i
    """
    weights = _validate_weights(weights)

    marginal = marginal_risk(
        weights,
        covariance,
    )

    return weights * marginal


def risk_contribution(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Calculate percentage contribution of each asset
    to total portfolio risk.
    """
    component = component_risk(
        weights,
        covariance,
    )

    total_risk = component.sum()

    if np.isclose(total_risk, 0):
        return np.zeros(len(weights))

    return component / total_risk


def risk_contribution_table(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    asset_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return a table containing portfolio risk contributions."""
    weights = _validate_weights(weights)

    if asset_names is None:
        asset_names = [
            f"Asset_{i}"
            for i in range(len(weights))
        ]

    if len(asset_names) != len(weights):
        raise ValueError(
            "asset_names length must match number of weights"
        )

    marginal = marginal_risk(
        weights,
        covariance,
    )

    component = component_risk(
        weights,
        covariance,
    )

    contribution = risk_contribution(
        weights,
        covariance,
    )

    return pd.DataFrame(
        {
            "Asset": asset_names,
            "Weight": weights,
            "Marginal_Risk": marginal,
            "Component_Risk": component,
            "Risk_Contribution": contribution,
        }
    )


# ---------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------

def weight_concentration(
    weights: np.ndarray | pd.Series,
) -> float:
    """Calculate Herfindahl-style weight concentration."""
    weights = _validate_weights(weights)

    return float(
        np.sum(weights ** 2)
    )


def risk_concentration(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """Calculate concentration of portfolio risk contributions."""
    contributions = risk_contribution(
        weights,
        covariance,
    )

    return float(
        np.sum(contributions ** 2)
    )


# ---------------------------------------------------------------------
# Diversification
# ---------------------------------------------------------------------

def diversification_ratio(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
) -> float:
    """
    Calculate portfolio diversification ratio.

    DR =
        weighted average individual volatility
        --------------------------------------
             portfolio volatility
    """
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

    portfolio_vol = portfolio_volatility(
        weights,
        covariance,
    )

    if portfolio_vol == 0:
        return 0.0

    return float(
        np.sum(
            np.abs(weights) * asset_volatility
        )
        / portfolio_vol
    )


# ---------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------

def portfolio_beta(
    weights: np.ndarray | pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta relative to a benchmark."""
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

    if not np.isfinite(
        benchmark_returns.to_numpy()
    ).all():
        raise ValueError(
            "benchmark returns contain invalid values"
        )

    aligned = pd.concat(
        [
            asset_returns,
            benchmark_returns.rename("Benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "no overlapping observations between "
            "assets and benchmark"
        )

    asset_data = aligned.iloc[:, :-1]
    benchmark = aligned["Benchmark"]

    benchmark_variance = benchmark.var()

    if np.isclose(benchmark_variance, 0):
        raise ValueError(
            "benchmark variance cannot be zero"
        )

    asset_betas = asset_data.apply(
        lambda column:
        column.cov(benchmark)
        / benchmark_variance
    )

    return float(
        weights @ asset_betas.to_numpy()
    )


# ---------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------

def portfolio_correlation(
    weights: np.ndarray | pd.Series,
    returns: pd.DataFrame,
) -> float:
    """
    Calculate weighted average pairwise correlation.

    Absolute weight products are used so long/short
    positions do not cancel each other out.
    """
    weights = _validate_weights(weights)

    returns = _validate_returns(
        returns
    )

    if len(weights) != returns.shape[1]:
        raise ValueError(
            "weights must match number of assets"
        )

    correlation = returns.corr().to_numpy()

    weighted_sum = 0.0
    total_weight = 0.0

    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            weight = abs(
                weights[i] * weights[j]
            )

            if np.isfinite(correlation[i, j]):
                weighted_sum += (
                    weight * correlation[i, j]
                )

                total_weight += weight

    if total_weight == 0:
        return 0.0

    return float(
        weighted_sum / total_weight
    )


def average_pairwise_correlation(
    returns: pd.DataFrame,
) -> float:
    """Calculate average pairwise asset correlation."""
    returns = _validate_returns(
        returns
    )

    correlation = returns.corr()

    n_assets = correlation.shape[0]

    if n_assets < 2:
        return 0.0

    values = correlation.to_numpy()

    upper_triangle = values[
        np.triu_indices(
            n_assets,
            k=1,
        )
    ]

    upper_triangle = upper_triangle[
        np.isfinite(upper_triangle)
    ]

    if len(upper_triangle) == 0:
        return 0.0

    return float(
        np.mean(upper_triangle)
    )


# ---------------------------------------------------------------------
# Industry Exposure
# ---------------------------------------------------------------------

def industry_exposure(
    weights: np.ndarray | pd.Series,
    industries: pd.Series | list[str],
) -> pd.Series:
    """Aggregate portfolio weights by industry."""
    weights = _validate_weights(weights)

    industries = pd.Series(
        industries
    ).reset_index(drop=True)

    if len(weights) != len(industries):
        raise ValueError(
            "industries length must match weights"
        )

    return (
        pd.Series(
            weights,
            index=industries,
        )
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )


# ---------------------------------------------------------------------
# Portfolio Risk Summary
# ---------------------------------------------------------------------

def portfolio_risk_summary(
    weights: np.ndarray | pd.Series,
    covariance: np.ndarray | pd.DataFrame,
    returns: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Generate a portfolio-level risk summary."""
    weights = _validate_weights(weights)

    summary = {
        "portfolio_variance": portfolio_variance(
            weights,
            covariance,
        ),
        "portfolio_volatility": portfolio_volatility(
            weights,
            covariance,
        ),
        "weight_concentration": weight_concentration(
            weights,
        ),
        "risk_concentration": risk_concentration(
            weights,
            covariance,
        ),
        "diversification_ratio": diversification_ratio(
            weights,
            covariance,
        ),
    }

    if returns is not None:
        summary[
            "average_pairwise_correlation"
        ] = average_pairwise_correlation(
            returns
        )

        summary[
            "portfolio_correlation"
        ] = portfolio_correlation(
            weights,
            returns,
        )

    if (
        benchmark_returns is not None
        and returns is not None
    ):
        summary[
            "portfolio_beta"
        ] = portfolio_beta(
            weights,
            returns,
            benchmark_returns,
        )

    return summary


# ---------------------------------------------------------------------
# Portfolio Comparison
# ---------------------------------------------------------------------

def compare_portfolio_risk(
    portfolios: dict[
        str,
        np.ndarray | pd.Series
    ],
    covariance: np.ndarray | pd.DataFrame,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare risk characteristics across portfolios."""
    results = []

    for name, weights in portfolios.items():
        weights = _validate_weights(
            weights
        )

        row = {
            "Portfolio": name,
            "Volatility": portfolio_volatility(
                weights,
                covariance,
            ),
            "Weight_Concentration": weight_concentration(
                weights,
            ),
            "Risk_Concentration": risk_concentration(
                weights,
                covariance,
            ),
            "Diversification_Ratio": diversification_ratio(
                weights,
                covariance,
            ),
        }

        if returns is not None:
            row[
                "Average_Correlation"
            ] = average_pairwise_correlation(
                returns
            )

            row[
                "Portfolio_Correlation"
            ] = portfolio_correlation(
                weights,
                returns,
            )

        results.append(row)

    return (
        pd.DataFrame(results)
        .set_index("Portfolio")
    )