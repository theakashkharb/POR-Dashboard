import numpy as np
import pandas as pd


def _validate_covariance(covariance):
    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be a 2-dimensional matrix"
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance matrix must be square"
        )

    if covariance.shape[0] == 0:
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

    return covariance


def _validate_target_risk_budgets(
    target_risk_budgets,
    n_assets
):
    if target_risk_budgets is None:
        return np.full(
            n_assets,
            1.0 / n_assets
        )

    budgets = np.asarray(
        target_risk_budgets,
        dtype=float
    )

    if budgets.ndim != 1:
        raise ValueError(
            "target_risk_budgets must be 1-dimensional"
        )

    if len(budgets) != n_assets:
        raise ValueError(
            "target_risk_budgets must match "
            "number of assets"
        )

    if not np.all(np.isfinite(budgets)):
        raise ValueError(
            "target_risk_budgets must contain "
            "finite values"
        )

    if np.any(budgets <= 0):
        raise ValueError(
            "target_risk_budgets must be positive"
        )

    total = budgets.sum()

    if total <= 0:
        raise ValueError(
            "target_risk_budgets must have "
            "a positive sum"
        )

    return budgets / total


def portfolio_volatility(
    weights,
    covariance
):
    """
    Calculate portfolio volatility.
    """

    variance = (
        weights
        @ covariance
        @ weights
    )

    return np.sqrt(
        max(variance, 0.0)
    )


def risk_contributions(
    weights,
    covariance
):
    """
    Calculate percentage risk contribution
    of each asset.

    Formula:

        RC_i =
        w_i * (Σw)_i / portfolio_variance
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    covariance = _validate_covariance(
        covariance
    )

    if len(weights) != covariance.shape[0]:
        raise ValueError(
            "weights and covariance dimensions "
            "do not match"
        )

    marginal_contribution = (
        covariance @ weights
    )

    contribution = (
        weights
        * marginal_contribution
    )

    total_contribution = (
        contribution.sum()
    )

    if np.isclose(
        total_contribution,
        0.0
    ):
        return np.zeros(
            len(weights)
        )

    return (
        contribution
        / total_contribution
    )


def risk_parity_weights(
    covariance,
    target_risk_budgets=None,
    long_only=True,
    max_iterations=10000,
    tolerance=1e-10
):
    """
    Calculate risk-parity portfolio weights.

    By default, every asset receives equal risk:

        RC_i = 1 / N

    The algorithm uses multiplicative updates
    and therefore requires only NumPy.

    Parameters
    ----------
    covariance : array-like
        Covariance matrix.

    target_risk_budgets : array-like, optional
        Desired percentage risk contribution
        for each asset.

    long_only : bool
        If True, all weights remain non-negative.

    max_iterations : int
        Maximum number of iterations.

    tolerance : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Portfolio weights.
    """

    covariance = _validate_covariance(
        covariance
    )

    n_assets = covariance.shape[0]

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be positive"
        )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive"
        )

    target = _validate_target_risk_budgets(
        target_risk_budgets,
        n_assets
    )

    # ---------------------------------------------------------
    # Initial weights
    # ---------------------------------------------------------

    diagonal = np.diag(covariance)

    if np.any(diagonal < 0):
        raise ValueError(
            "covariance diagonal cannot be negative"
        )

    asset_volatility = np.sqrt(
        np.maximum(
            diagonal,
            1e-16
        )
    )

    weights = 1.0 / asset_volatility

    weights = (
        weights / weights.sum()
    )

    # ---------------------------------------------------------
    # Multiplicative risk-budgeting updates
    # ---------------------------------------------------------

    for _ in range(max_iterations):

        old_weights = weights.copy()

        portfolio_variance = (
            weights
            @ covariance
            @ weights
        )

        if portfolio_variance <= 1e-16:
            break

        marginal_risk = (
            covariance @ weights
        )

        contribution = (
            weights
            * marginal_risk
        )

        contribution = np.maximum(
            contribution,
            1e-16
        )

        contribution_share = (
            contribution
            / contribution.sum()
        )

        # Increase assets contributing too little
        # and decrease assets contributing too much.
        adjustment = (
            target
            / contribution_share
        )

        adjustment = np.sqrt(
            adjustment
        )

        weights = (
            weights * adjustment
        )

        if long_only:
            weights = np.maximum(
                weights,
                1e-16
            )

        else:
            # Risk parity is primarily designed
            # for positive portfolio weights.
            # Keep the implementation stable.
            weights = np.maximum(
                weights,
                1e-16
            )

        weights = (
            weights / weights.sum()
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

    weights = np.maximum(
        weights,
        0.0
    )

    weights = (
        weights / weights.sum()
    )

    return weights


def risk_parity_series(
    covariance,
    asset_names=None,
    target_risk_budgets=None,
    long_only=True,
    max_iterations=10000,
    tolerance=1e-10
):
    """
    Calculate risk-parity weights as a pandas Series.
    """

    covariance_array = _validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

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
            "asset_names must match "
            "number of assets"
        )

    weights = risk_parity_weights(
        covariance=covariance,
        target_risk_budgets=target_risk_budgets,
        long_only=long_only,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def risk_parity_dataframe(
    covariance,
    asset_names=None,
    target_risk_budgets=None,
    long_only=True
):
    """
    Return risk-parity portfolio as a DataFrame.
    """

    weights = risk_parity_series(
        covariance=covariance,
        asset_names=asset_names,
        target_risk_budgets=target_risk_budgets,
        long_only=long_only
    )

    return weights.to_frame()


def validate_risk_parity(
    weights,
    covariance,
    target_risk_budgets=None,
    tolerance=1e-4
):
    """
    Check whether a portfolio approximately satisfies
    the requested risk budgets.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    covariance = _validate_covariance(
        covariance
    )

    if len(weights) != covariance.shape[0]:
        raise ValueError(
            "weights and covariance dimensions "
            "do not match"
        )

    target = _validate_target_risk_budgets(
        target_risk_budgets,
        len(weights)
    )

    actual = risk_contributions(
        weights,
        covariance
    )

    return bool(
        np.allclose(
            actual,
            target,
            atol=tolerance
        )
    )