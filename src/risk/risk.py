import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. HISTORICAL VOLATILITY
# ---------------------------------------------------------

def calculate_historical_volatility(
    data,
    window=90,
    annualization=252
):
    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    volatility = (
        data.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .groupby(level=0)
        .last()
        * np.sqrt(annualization)
    )

    return volatility.dropna()


# ---------------------------------------------------------
# 2. EWMA VOLATILITY
# ---------------------------------------------------------

def calculate_ewma_volatility(
    data,
    span=90,
    annualization=252
):
    data = data.sort_values(
        ["Ticker", "Date"]
    ).copy()

    data["Return"] = (
        data.groupby("Ticker")["Close"]
        .pct_change()
    )

    ewma_variance = (
        data.groupby("Ticker")["Return"]
        .ewm(span=span)
        .var()
        .groupby(level=0)
        .last()
    )

    ewma_volatility = (
        np.sqrt(ewma_variance)
        * np.sqrt(annualization)
    )

    return ewma_volatility.dropna()


# ---------------------------------------------------------
# 3. RETURN MATRIX
# ---------------------------------------------------------

def create_return_matrix(data):
    data = data.sort_values("Date").copy()

    returns = (
        data.pivot(
            index="Date",
            columns="Ticker",
            values="Close"
        )
        .pct_change()
        .dropna()
    )

    return returns


# ---------------------------------------------------------
# 4. SAMPLE COVARIANCE
# ---------------------------------------------------------

def calculate_sample_covariance(
    data,
    annualization=252
):
    returns = create_return_matrix(data)

    covariance = (
        returns.cov()
        * annualization
    )

    return covariance


# ---------------------------------------------------------
# 5. LEDOIT-WOLF COVARIANCE
#    MANUAL IMPLEMENTATION
# ---------------------------------------------------------

def calculate_ledoit_wolf_covariance(
    data,
    annualization=252
):
    returns = create_return_matrix(data)

    X = returns.values

    n_observations, n_assets = X.shape

    # Center returns
    X = X - X.mean(axis=0)

    # Sample covariance
    sample_covariance = (
        X.T @ X
    ) / n_observations

    # Target: scaled identity matrix
    average_variance = (
        np.trace(sample_covariance)
        / n_assets
    )

    target = (
        average_variance
        * np.eye(n_assets)
    )

    # Estimate the variance of covariance elements
    sample_covariance_error = np.zeros(
        (n_assets, n_assets)
    )

    for t in range(n_observations):

        outer_product = np.outer(
            X[t],
            X[t]
        )

        deviation = (
            outer_product
            - sample_covariance
        )

        sample_covariance_error += (
            deviation ** 2
        )

    phi = (
        sample_covariance_error
        / n_observations
    ).sum()

    # Distance between sample covariance
    # and shrinkage target
    gamma = np.sum(
        (sample_covariance - target) ** 2
    )

    # Optimal shrinkage intensity
    if gamma == 0:
        shrinkage = 0
    else:
        shrinkage = min(
            max(phi / (n_observations * gamma), 0),
            1
        )

    # Shrunk covariance
    shrunk_covariance = (
        shrinkage * target
        + (1 - shrinkage) * sample_covariance
    )

    covariance = pd.DataFrame(
        shrunk_covariance * annualization,
        index=returns.columns,
        columns=returns.columns
    )

    return covariance


# ---------------------------------------------------------
# 6. CORRELATION
# ---------------------------------------------------------

def calculate_correlation(data):
    returns = create_return_matrix(data)

    return returns.corr()


# ---------------------------------------------------------
# 7. SINGLE-INDEX BETA
# ---------------------------------------------------------

def calculate_beta(
    data,
    benchmark_returns
):
    returns = create_return_matrix(data)

    common_dates = (
        returns.index
        .intersection(
            benchmark_returns.index
        )
    )

    returns = returns.loc[common_dates]

    benchmark_returns = (
        benchmark_returns
        .loc[common_dates]
    )

    benchmark_variance = (
        benchmark_returns.var()
    )

    beta = returns.apply(
        lambda x:
        x.cov(benchmark_returns)
        / benchmark_variance
    )

    return beta


# ---------------------------------------------------------
# 8. PORTFOLIO VOLATILITY
# ---------------------------------------------------------

def calculate_portfolio_volatility(
    weights,
    covariance
):
    weights = np.asarray(weights)

    covariance_matrix = covariance.values

    portfolio_variance = (
        weights
        @ covariance_matrix
        @ weights
    )

    return np.sqrt(
        portfolio_variance
    )


# ---------------------------------------------------------
# 9. MARGINAL RISK
# ---------------------------------------------------------

def calculate_marginal_risk(
    weights,
    covariance
):
    weights = np.asarray(weights)

    covariance_matrix = covariance.values

    portfolio_volatility = (
        calculate_portfolio_volatility(
            weights,
            covariance
        )
    )

    marginal_risk = (
        covariance_matrix @ weights
    ) / portfolio_volatility

    return pd.Series(
        marginal_risk,
        index=covariance.index
    )


# ---------------------------------------------------------
# 10. COMPONENT RISK
# ---------------------------------------------------------

def calculate_component_risk(
    weights,
    covariance
):
    weights = np.asarray(weights)

    marginal_risk = (
        calculate_marginal_risk(
            weights,
            covariance
        )
    )

    component_risk = (
        pd.Series(
            weights,
            index=covariance.index
        )
        * marginal_risk
    )

    return component_risk


# ---------------------------------------------------------
# 11. RISK CONTRIBUTION
# ---------------------------------------------------------

def calculate_risk_contribution(
    weights,
    covariance
):
    component_risk = (
        calculate_component_risk(
            weights,
            covariance
        )
    )

    total_risk = component_risk.sum()

    if total_risk == 0:
        return pd.Series(
            0,
            index=component_risk.index
        )

    risk_contribution = (
        component_risk
        / total_risk
    )

    return risk_contribution