import numpy as np
import pandas as pd

from src.features.returns import create_return_matrix


def calculate_sample_covariance(
    data,
    annualization=252,
):
    """
    Calculate annualized sample covariance matrix
    from historical stock prices.
    """

    returns = create_return_matrix(data)

    return returns.cov() * annualization


def calculate_ledoit_wolf_covariance(
    data,
    annualization=252,
):
    """
    Calculate annualized Ledoit-Wolf shrinkage covariance
    matrix using a manual implementation.
    """

    returns = create_return_matrix(data)

    X = returns.values

    n_observations, n_assets = X.shape

    if n_observations == 0:
        raise ValueError("returns cannot be empty")

    if n_assets == 0:
        raise ValueError("returns must contain at least one asset")

    # Center returns
    X = X - X.mean(axis=0)

    # Sample covariance
    sample_covariance = (
        X.T @ X
    ) / n_observations

    # Shrinkage target:
    # scaled identity matrix
    average_variance = (
        np.trace(sample_covariance)
        / n_assets
    )

    target = (
        average_variance
        * np.eye(n_assets)
    )

    # Estimate covariance estimation error
    sample_covariance_error = np.zeros(
        (n_assets, n_assets)
    )

    for t in range(n_observations):

        outer_product = np.outer(
            X[t],
            X[t],
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
        shrinkage = 0.0
    else:
        shrinkage = min(
            max(
                phi / (
                    n_observations * gamma
                ),
                0.0,
            ),
            1.0,
        )

    # Shrunk covariance
    shrunk_covariance = (
        shrinkage * target
        + (1 - shrinkage)
        * sample_covariance
    )

    return pd.DataFrame(
        shrunk_covariance * annualization,
        index=returns.columns,
        columns=returns.columns,
    )


def calculate_correlation(data):
    """
    Calculate asset correlation matrix
    from historical stock prices.
    """

    returns = create_return_matrix(data)

    return returns.corr()