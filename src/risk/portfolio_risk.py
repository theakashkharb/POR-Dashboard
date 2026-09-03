"""
Portfolio Risk Analytics Engine
POR-Dashboard

Purpose
-------
Measure portfolio-level risk for each optimized portfolio.

The module is deliberately independent from:
    - optimization
    - backtesting
    - performance ranking

It accepts portfolio weights, covariance, asset returns and optional
benchmark / industry information and returns reusable risk analytics.

Dependencies:
    numpy
    pandas
"""

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_weights(weights):
    """Validate and normalize portfolio weights."""
    weights = pd.Series(weights, dtype=float).dropna()

    if weights.empty:
        raise ValueError("weights cannot be empty.")

    if not np.isfinite(weights.values).all():
        raise ValueError("weights contain non-finite values.")

    if (weights < -1e-12).any():
        raise ValueError("Portfolio Risk Engine supports long-only weights.")

    total = float(weights.sum())

    if total <= 0:
        raise ValueError("weights must have a positive sum.")

    weights = weights / total

    return weights


def _validate_covariance(covariance, assets):
    """Validate covariance matrix and align it to portfolio assets."""
    covariance = pd.DataFrame(covariance, dtype=float)

    missing_rows = [asset for asset in assets if asset not in covariance.index]
    missing_cols = [asset for asset in assets if asset not in covariance.columns]

    if missing_rows or missing_cols:
        raise ValueError("covariance does not contain all portfolio assets.")

    covariance = covariance.loc[assets, assets]

    if not np.isfinite(covariance.values).all():
        raise ValueError("covariance contains non-finite values.")

    covariance = (covariance + covariance.T) / 2.0

    return covariance


def _validate_returns(returns, assets):
    """Validate and align an asset return matrix."""
    returns = pd.DataFrame(returns, dtype=float)

    missing = [asset for asset in assets if asset not in returns.columns]

    if missing:
        raise ValueError("returns do not contain all portfolio assets.")

    returns = returns.loc[:, assets].dropna(how="all")

    return returns


# ============================================================
# BASIC PORTFOLIO RISK
# ============================================================

def portfolio_variance(weights, covariance):
    """Annualized portfolio variance from an annualized covariance matrix."""
    weights = _validate_weights(weights)
    covariance = _validate_covariance(covariance, weights.index)

    w = weights.values
    sigma = covariance.values

    return float(w @ sigma @ w)


def portfolio_volatility(weights, covariance):
    """Annualized portfolio volatility."""
    variance = portfolio_variance(weights, covariance)

    return float(np.sqrt(max(variance, 0.0)))


# ============================================================
# MARGINAL / COMPONENT RISK
# ============================================================

def marginal_risk(weights, covariance):
    """
    Marginal risk of each asset.

    MCR_i = (Sigma w)_i / portfolio_volatility
    """
    weights = _validate_weights(weights)
    covariance = _validate_covariance(covariance, weights.index)

    w = weights.values
    sigma = covariance.values

    volatility = portfolio_volatility(weights, covariance)

    if volatility <= 1e-16:
        values = np.zeros(len(weights))
    else:
        values = sigma @ w / volatility

    return pd.Series(values, index=weights.index, name="marginal_risk")


def component_risk(weights, covariance):
    """
    Component risk contribution in volatility units.

    CR_i = w_i * MCR_i
    """
    weights = _validate_weights(weights)
    mcr = marginal_risk(weights, covariance)

    values = weights * mcr

    values.name = "component_risk"

    return values


def risk_contribution(weights, covariance):
    """
    Percentage contribution of each asset to total portfolio volatility.
    """
    weights = _validate_weights(weights)
    covariance = _validate_covariance(covariance, weights.index)

    component = component_risk(weights, covariance)
    volatility = portfolio_volatility(weights, covariance)

    if volatility <= 1e-16:
        values = pd.Series(0.0, index=weights.index)
    else:
        values = component / volatility

    values.name = "risk_contribution"

    return values


def risk_contribution_table(weights, covariance):
    """Return a complete asset-level risk contribution table."""
    weights = _validate_weights(weights)
    covariance = _validate_covariance(covariance, weights.index)

    volatility = portfolio_volatility(weights, covariance)
    mcr = marginal_risk(weights, covariance)
    component = component_risk(weights, covariance)
    contribution = risk_contribution(weights, covariance)

    return pd.DataFrame(
        {
            "weight": weights,
            "marginal_risk": mcr,
            "component_risk": component,
            "risk_contribution": contribution,
        }
    )


# ============================================================
# CONCENTRATION
# ============================================================

def weight_concentration(weights):

    if isinstance(weights, dict):

        weights = pd.Series(
            weights,
            dtype=float
        )

    else:

        weights = pd.Series(
            weights,
            dtype=float
        )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    values = weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "weights contain non-finite values."
        )

    if (
        values < 0
    ).any():

        raise ValueError(
            "weights cannot contain negative values."
        )

    total = values.sum()

    if not np.isfinite(total):
        raise ValueError(
            "weights sum must be finite."
        )

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    normalized_weights = (
        values / total
    )

    return float(
        np.sum(
            normalized_weights ** 2
        )
    )

def risk_concentration(weights, covariance):
    """
    Risk concentration based on squared risk contributions.

    Lower concentration generally means risk is distributed
    more evenly across assets.
    """
    contribution = risk_contribution(weights, covariance)

    hhi = float(np.sum(contribution.values ** 2))

    effective_risk_buckets = (
        1.0 / hhi
        if hhi > 1e-16
        else 0.0
    )

    return {
        "risk_contribution_hhi": hhi,
        "effective_risk_buckets": effective_risk_buckets,
        "largest_risk_contribution": float(contribution.max()),
    }


# ============================================================
# DIVERSIFICATION
# ============================================================

def diversification_ratio(weights, covariance):
    """
    Diversification ratio:

        DR = weighted average asset volatility / portfolio volatility
    """
    weights = _validate_weights(weights)
    covariance = _validate_covariance(covariance, weights.index)

    asset_volatility = np.sqrt(
        np.maximum(np.diag(covariance.values), 0.0)
    )

    portfolio_vol = portfolio_volatility(weights, covariance)

    if portfolio_vol <= 1e-16:
        return 0.0

    return float(
        weights.values @ asset_volatility
        / portfolio_vol
    )


# ============================================================
# BETA
# ============================================================

def portfolio_beta(weights, asset_betas):
    """
    Weighted portfolio beta.

    Parameters
    ----------
    weights : Series or dict
    asset_betas : Series or dict
        Asset beta relative to the selected benchmark.
    """
    weights = _validate_weights(weights)
    asset_betas = pd.Series(asset_betas, dtype=float)

    missing = [
        asset for asset in weights.index
        if asset not in asset_betas.index
    ]

    if missing:
        raise ValueError(
            f"asset_betas missing assets: {missing}"
        )

    asset_betas = asset_betas.loc[weights.index]

    return float(weights @ asset_betas)


# ============================================================
# CORRELATION
# ============================================================

def portfolio_correlation(returns):
    """Return the asset correlation matrix."""
    returns = pd.DataFrame(returns, dtype=float)

    return returns.corr()


def average_pairwise_correlation(returns):
    """Average off-diagonal pairwise asset correlation."""
    correlation = portfolio_correlation(returns)

    n = len(correlation)

    if n <= 1:
        return 1.0

    values = correlation.values[
        np.triu_indices(n, k=1)
    ]

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


# ============================================================
# INDUSTRY EXPOSURE
# ============================================================

def industry_exposure(weights, industry_data):
    """
    Aggregate portfolio weights by industry.

    industry_data can be:
        Series indexed by asset
    OR
        DataFrame containing an 'industry' column.
    """
    weights = _validate_weights(weights)

    if industry_data is None:
        return pd.Series(
            dtype=float,
            name="industry_weight"
        )

    if isinstance(industry_data, pd.DataFrame):
        if "industry" not in industry_data.columns:
            raise ValueError(
                "industry_data DataFrame must contain 'industry'."
            )

        industries = industry_data["industry"]

    else:
        industries = pd.Series(industry_data)

    missing = [
        asset for asset in weights.index
        if asset not in industries.index
    ]

    if missing:
        raise ValueError(
            f"industry_data missing assets: {missing}"
        )

    industries = industries.loc[weights.index]

    result = (
        pd.DataFrame(
            {
                "weight": weights,
                "industry": industries
            }
        )
        .groupby("industry")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    result.name = "industry_weight"

    return result


# ============================================================
# COMPLETE PORTFOLIO RISK SUMMARY
# ============================================================

def portfolio_risk_summary(
    weights,
    covariance,
    returns=None,
    asset_betas=None,
    industry_data=None,
):
    """
    Generate a complete portfolio-level risk summary.

    Returns a dictionary containing scalar risk metrics and
    supporting tables.
    """
    weights = _validate_weights(weights)
    covariance = _validate_covariance(
        covariance,
        weights.index
    )

    volatility = portfolio_volatility(
        weights,
        covariance
    )

    weight_concentration_metrics = weight_concentration(
        weights
    )

    risk_concentration_metrics = risk_concentration(
        weights,
        covariance
    )

    result = {
        "portfolio_volatility": volatility,
        "weight_hhi": weight_concentration_metrics[
            "weight_hhi"
        ],
        "effective_number_of_assets":
            weight_concentration_metrics[
                "effective_number_of_assets"
            ],
        "maximum_weight":
            weight_concentration_metrics[
                "maximum_weight"
            ],
        "risk_contribution_hhi":
            risk_concentration_metrics[
                "risk_contribution_hhi"
            ],
        "effective_risk_buckets":
            risk_concentration_metrics[
                "effective_risk_buckets"
            ],
        "largest_risk_contribution":
            risk_concentration_metrics[
                "largest_risk_contribution"
            ],
        "diversification_ratio":
            diversification_ratio(
                weights,
                covariance
            ),
    }

    if returns is not None:
        returns = _validate_returns(
            returns,
            weights.index
        )

        result["average_pairwise_correlation"] = (
            average_pairwise_correlation(returns)
        )

    if asset_betas is not None:
        result["portfolio_beta"] = portfolio_beta(
            weights,
            asset_betas
        )

    if industry_data is not None:
        result["industry_exposure"] = industry_exposure(
            weights,
            industry_data
        )

    result["risk_contribution_table"] = (
        risk_contribution_table(
            weights,
            covariance
        )
    )

    return result


# ============================================================
# COMPARE MULTIPLE PORTFOLIOS
# ============================================================

def compare_portfolio_risk(
    portfolio_weights,
    covariance,
    returns=None,
    asset_betas=None,
    industry_data=None,
):
    """
    Compare portfolio risk across multiple optimization methods.

    portfolio_weights:
        dictionary:
            {
                "equal_weight": weights,
                "minimum_variance": weights,
                ...
            }

    Returns
    -------
    DataFrame
        One row per portfolio method.
    """
    rows = []

    for method, weights in portfolio_weights.items():
        summary = portfolio_risk_summary(
            weights=weights,
            covariance=covariance,
            returns=returns,
            asset_betas=asset_betas,
            industry_data=industry_data,
        )

        row = {
            "Method": method,
            "Portfolio Volatility":
                summary["portfolio_volatility"],
            "Maximum Weight":
                summary["maximum_weight"],
            "Weight HHI":
                summary["weight_hhi"],
            "Effective Assets":
                summary["effective_number_of_assets"],
            "Largest Risk Contribution":
                summary["largest_risk_contribution"],
            "Risk Contribution HHI":
                summary["risk_contribution_hhi"],
            "Effective Risk Buckets":
                summary["effective_risk_buckets"],
            "Diversification Ratio":
                summary["diversification_ratio"],
        }

        if "portfolio_beta" in summary:
            row["Portfolio Beta"] = summary["portfolio_beta"]

        if "average_pairwise_correlation" in summary:
            row["Average Pairwise Correlation"] = (
                summary["average_pairwise_correlation"]
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("Method")
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    assets = [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "TCS.NS",
    ]

    weights = pd.Series(
        {
            "HDFCBANK.NS": 0.20,
            "ICICIBANK.NS": 0.20,
            "INFY.NS": 0.20,
            "RELIANCE.NS": 0.20,
            "TCS.NS": 0.20,
        }
    )

    covariance = pd.DataFrame(
        np.eye(5) * 0.04,
        index=assets,
        columns=assets,
    )

    print("=" * 60)
    print("PORTFOLIO RISK ENGINE TEST")
    print("=" * 60)

    volatility = portfolio_volatility(
        weights,
        covariance
    )

    print(
        f"Portfolio volatility: "
        f"{volatility:.4f}"
    )

    table = risk_contribution_table(
        weights,
        covariance
    )

    print("\nRisk contribution table:")
    print(table)

    assert np.isclose(
        table["risk_contribution"].sum(),
        1.0
    )

    summary = portfolio_risk_summary(
        weights,
        covariance
    )

    assert summary["portfolio_volatility"] > 0
    assert summary["effective_number_of_assets"] > 0
    assert summary["diversification_ratio"] > 0

    print("\nPortfolio risk summary:")
    for key, value in summary.items():
        if not isinstance(value, pd.DataFrame):
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("ALL PORTFOLIO RISK TESTS PASSED")
    print("=" * 60)
