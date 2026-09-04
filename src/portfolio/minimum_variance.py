import numpy as np
import pandas as pd


def minimum_variance_weights(
    covariance,
    long_only=True,
    max_iterations=5000,
    tolerance=1e-10
):
    """
    Calculate minimum-variance portfolio weights.

    Uses projected gradient descent, so no external
    optimization library is required.

    Parameters
    ----------
    covariance : pd.DataFrame or array-like
        Asset covariance matrix.

    long_only : bool
        If True, negative weights are not allowed.

    max_iterations : int
        Maximum optimization iterations.

    tolerance : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Portfolio weights.
    """

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be a 2-dimensional matrix"
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance matrix must be square"
        )

    n_assets = covariance.shape[0]

    if n_assets == 0:
        raise ValueError(
            "covariance matrix cannot be empty"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "covariance matrix must contain finite values"
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

    # ---------------------------------------------------------
    # Initial portfolio
    # ---------------------------------------------------------

    weights = np.full(
        n_assets,
        1.0 / n_assets
    )

    # ---------------------------------------------------------
    # Gradient of:
    #
    #     w'Σw
    #
    # is:
    #
    #     2Σw
    # ---------------------------------------------------------

    gradient_matrix = 2.0 * covariance

    # ---------------------------------------------------------
    # Step size
    #
    # Estimate a stable learning rate from the largest
    # absolute row sum of the covariance matrix.
    # ---------------------------------------------------------

    scale = np.max(
        np.sum(
            np.abs(covariance),
            axis=1
        )
    )

    if scale <= 1e-15:
        return weights

    learning_rate = 1.0 / (
        4.0 * scale
    )

    # ---------------------------------------------------------
    # Project weights back onto the simplex:
    #
    #     wi >= 0
    #     Σwi = 1
    #
    # when long_only=True.
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
    # Optimization
    # ---------------------------------------------------------

    for _ in range(max_iterations):

        old_weights = weights.copy()

        gradient = (
            gradient_matrix @ weights
        )

        if long_only:

            candidate = (
                weights
                - learning_rate * gradient
            )

            weights = project_simplex(
                candidate
            )

        else:

            # Equality constraint:
            # sum(weights) = 1
            #
            # Remove the component of the gradient
            # that changes the total weight.

            adjusted_gradient = (
                gradient
                - np.mean(gradient)
            )

            weights = (
                weights
                - learning_rate
                * adjusted_gradient
            )

            weights = (
                weights
                - (
                    weights.sum() - 1.0
                ) / n_assets
            )

        change = np.max(
            np.abs(
                weights - old_weights
            )
        )

        if change < tolerance:
            break

    # Final numerical normalization
    weights = (
        weights / weights.sum()
    )

    if long_only:
        weights = np.maximum(
            weights,
            0.0
        )

        weights = (
            weights / weights.sum()
        )

    return weights


def minimum_variance_series(
    covariance,
    asset_names=None,
    long_only=True,
    max_iterations=5000,
    tolerance=1e-10
):
    """
    Calculate minimum-variance weights and return
    them as a pandas Series.
    """

    covariance_array = np.asarray(
        covariance,
        dtype=float
    )

    n_assets = covariance_array.shape[0]

    if asset_names is None:

        if isinstance(
            covariance,
            pd.DataFrame
        ):
            asset_names = covariance.columns.tolist()

        else:
            asset_names = [
                f"Asset_{i}"
                for i in range(n_assets)
            ]

    if len(asset_names) != n_assets:
        raise ValueError(
            "asset_names must match number of assets"
        )

    weights = minimum_variance_weights(
        covariance=covariance,
        long_only=long_only,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def minimum_variance_dataframe(
    covariance,
    asset_names=None,
    long_only=True
):
    """
    Return minimum-variance portfolio as a DataFrame.
    """

    weights = minimum_variance_series(
        covariance=covariance,
        asset_names=asset_names,
        long_only=long_only
    )

    return weights.to_frame()