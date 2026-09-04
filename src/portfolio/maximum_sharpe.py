import numpy as np
import pandas as pd


def maximum_sharpe_weights(
    expected_returns,
    covariance,
    risk_free_rate=0.0,
    long_only=True,
    max_iterations=5000,
    tolerance=1e-10
):
    """
    Calculate maximum-Sharpe portfolio weights.

    Uses projected gradient ascent on the Sharpe ratio.
    No scipy or other optimization library is required.

    Parameters
    ----------
    expected_returns : array-like
        Expected annual returns for each asset.

    covariance : array-like
        Annualized covariance matrix.

    risk_free_rate : float
        Annual risk-free rate.

    long_only : bool
        If True, weights cannot be negative.

    max_iterations : int
        Maximum optimization iterations.

    tolerance : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Portfolio weights.
    """

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    if expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be 1-dimensional"
        )

    if len(expected_returns) == 0:
        raise ValueError(
            "expected_returns cannot be empty"
        )

    if not np.all(np.isfinite(expected_returns)):
        raise ValueError(
            "expected_returns must contain finite values"
        )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be 2-dimensional"
        )

    if (
        covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance matrix must be square"
        )

    if covariance.shape[0] != len(
        expected_returns
    ):
        raise ValueError(
            "expected_returns and covariance "
            "dimensions do not match"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "covariance must contain finite values"
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10
    ):
        raise ValueError(
            "covariance matrix must be symmetric"
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be positive"
        )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive"
        )

    n_assets = len(expected_returns)

    # ---------------------------------------------------------
    # Initial portfolio
    # ---------------------------------------------------------

    weights = np.full(
        n_assets,
        1.0 / n_assets
    )

    excess_returns = (
        expected_returns
        - risk_free_rate
    )

    # ---------------------------------------------------------
    # Simplex projection for long-only portfolios
    # ---------------------------------------------------------

    def project_simplex(vector):

        sorted_vector = np.sort(
            vector
        )[::-1]

        cumulative = np.cumsum(
            sorted_vector
        )

        indices = np.arange(
            1,
            len(vector) + 1
        )

        condition = (
            sorted_vector
            - (
                cumulative - 1.0
            ) / indices
            > 0
        )

        if not np.any(condition):
            return np.full(
                len(vector),
                1.0 / len(vector)
            )

        rho = np.where(
            condition
        )[0][-1]

        threshold = (
            cumulative[rho] - 1.0
        ) / (rho + 1)

        return np.maximum(
            vector - threshold,
            0.0
        )

    # ---------------------------------------------------------
    # Sharpe ratio gradient
    #
    # S = (w'μ - Rf) / sqrt(w'Σw)
    #
    # Gradient:
    #
    # ∇S =
    # μ_excess / σ
    # -
    # S * (Σw / σ²)
    # ---------------------------------------------------------

    def sharpe_gradient(weights):

        portfolio_excess_return = np.dot(
            weights,
            excess_returns
        )

        variance = (
            weights
            @ covariance
            @ weights
        )

        variance = max(
            variance,
            1e-16
        )

        volatility = np.sqrt(
            variance
        )

        sharpe = (
            portfolio_excess_return
            / volatility
        )

        gradient = (
            excess_returns / volatility
            -
            sharpe
            * (
                covariance @ weights
            )
            / variance
        )

        return gradient

    # ---------------------------------------------------------
    # Learning rate
    # ---------------------------------------------------------

    covariance_scale = np.max(
        np.sum(
            np.abs(covariance),
            axis=1
        )
    )

    if covariance_scale <= 1e-15:
        covariance_scale = 1.0

    learning_rate = (
        0.01
        / np.sqrt(covariance_scale)
    )

    # ---------------------------------------------------------
    # Gradient ascent
    # ---------------------------------------------------------

    for _ in range(max_iterations):

        old_weights = weights.copy()

        gradient = sharpe_gradient(
            weights
        )

        # Normalize gradient to prevent
        # extremely large optimization steps.
        gradient_norm = np.linalg.norm(
            gradient
        )

        if gradient_norm > 1e-12:
            gradient = (
                gradient / gradient_norm
            )

        candidate = (
            weights
            + learning_rate * gradient
        )

        if long_only:

            weights = project_simplex(
                candidate
            )

        else:

            # Maintain sum(weights) = 1
            weights = (
                candidate
                - (
                    candidate.sum() - 1.0
                ) / n_assets
            )

        change = np.max(
            np.abs(
                weights - old_weights
            )
        )

        if change < tolerance:
            break

    # ---------------------------------------------------------
    # Final normalization
    # ---------------------------------------------------------

    if long_only:

        weights = np.maximum(
            weights,
            0.0
        )

    weights = (
        weights / weights.sum()
    )

    return weights


def maximum_sharpe_series(
    expected_returns,
    covariance,
    asset_names=None,
    risk_free_rate=0.0,
    long_only=True,
    max_iterations=5000,
    tolerance=1e-10
):
    """
    Calculate maximum-Sharpe weights as a pandas Series.
    """

    if isinstance(
        expected_returns,
        pd.Series
    ):
        if asset_names is None:
            asset_names = (
                expected_returns.index.tolist()
            )

        expected_returns = (
            expected_returns.values
        )

    covariance_array = np.asarray(
        covariance,
        dtype=float
    )

    n_assets = len(
        expected_returns
    )

    if asset_names is None:

        if isinstance(
            covariance,
            pd.DataFrame
        ):
            asset_names = (
                covariance.columns.tolist()
            )

        else:
            asset_names = [
                f"Asset_{i}"
                for i in range(n_assets)
            ]

    if len(asset_names) != n_assets:
        raise ValueError(
            "asset_names must match number of assets"
        )

    weights = maximum_sharpe_weights(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_free_rate=risk_free_rate,
        long_only=long_only,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def maximum_sharpe_dataframe(
    expected_returns,
    covariance,
    asset_names=None,
    risk_free_rate=0.0,
    long_only=True
):
    """
    Return maximum-Sharpe portfolio as a DataFrame.
    """

    weights = maximum_sharpe_series(
        expected_returns=expected_returns,
        covariance=covariance,
        asset_names=asset_names,
        risk_free_rate=risk_free_rate,
        long_only=long_only
    )

    return weights.to_frame()


def calculate_sharpe_ratio(
    weights,
    expected_returns,
    covariance,
    risk_free_rate=0.0
):
    """
    Calculate the Sharpe ratio of a portfolio.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    portfolio_return = np.dot(
        weights,
        expected_returns
    )

    variance = (
        weights
        @ covariance
        @ weights
    )

    volatility = np.sqrt(
        max(variance, 0.0)
    )

    if np.isclose(
        volatility,
        0.0
    ):
        return 0.0

    return float(
        (
            portfolio_return
            - risk_free_rate
        )
        / volatility
    )