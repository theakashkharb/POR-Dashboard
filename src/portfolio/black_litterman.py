import numpy as np
import pandas as pd


def validate_covariance(covariance):
    """Validate covariance matrix."""

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be 2-dimensional"
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


def validate_weights(weights, n_assets):
    """Validate portfolio weights."""

    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim != 1:
        raise ValueError(
            "weights must be 1-dimensional"
        )

    if len(weights) != n_assets:
        raise ValueError(
            "weights must match number of assets"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    return weights


def market_implied_returns(
    covariance,
    market_weights,
    risk_aversion=2.5
):
    """
    Calculate Black-Litterman equilibrium returns.

    Formula:

        Π = δΣw

    where:

        Π = equilibrium returns
        δ = risk-aversion coefficient
        Σ = covariance matrix
        w = market-cap weights
    """

    covariance = validate_covariance(
        covariance
    )

    n_assets = covariance.shape[0]

    market_weights = validate_weights(
        market_weights,
        n_assets
    )

    if risk_aversion <= 0:
        raise ValueError(
            "risk_aversion must be positive"
        )

    weight_sum = market_weights.sum()

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-8
    ):
        raise ValueError(
            "market_weights must sum to 1"
        )

    implied_returns = (
        risk_aversion
        * covariance
        @ market_weights
    )

    return implied_returns


def posterior_returns(
    prior_returns,
    covariance,
    views,
    view_matrix,
    tau=0.05,
    view_uncertainty=None
):
    """
    Calculate Black-Litterman posterior expected returns.

    Uses the standard Black-Litterman formula:

        μ_BL =
        [(τΣ)^-1 + P'Ω^-1P]^-1
        [(τΣ)^-1Π + P'Ω^-1Q]

    Parameters
    ----------
    prior_returns : array-like
        Equilibrium/prior expected returns Π.

    covariance : array-like
        Asset covariance matrix Σ.

    views : array-like
        Investor views Q.

    view_matrix : array-like
        View matrix P.

    tau : float
        Uncertainty in equilibrium returns.

    view_uncertainty : array-like, optional
        Diagonal of Ω. If omitted, a reasonable
        variance-based value is generated.
    """

    covariance = validate_covariance(
        covariance
    )

    n_assets = covariance.shape[0]

    prior_returns = np.asarray(
        prior_returns,
        dtype=float
    )

    views = np.asarray(
        views,
        dtype=float
    )

    view_matrix = np.asarray(
        view_matrix,
        dtype=float
    )

    if prior_returns.ndim != 1:
        raise ValueError(
            "prior_returns must be 1-dimensional"
        )

    if len(prior_returns) != n_assets:
        raise ValueError(
            "prior_returns must match number of assets"
        )

    if views.ndim != 1:
        raise ValueError(
            "views must be 1-dimensional"
        )

    if view_matrix.ndim != 2:
        raise ValueError(
            "view_matrix must be 2-dimensional"
        )

    n_views = len(views)

    if view_matrix.shape != (
        n_views,
        n_assets
    ):
        raise ValueError(
            "view_matrix dimensions do not match "
            "views and assets"
        )

    if not np.all(
        np.isfinite(prior_returns)
    ):
        raise ValueError(
            "prior_returns must contain finite values"
        )

    if not np.all(
        np.isfinite(views)
    ):
        raise ValueError(
            "views must contain finite values"
        )

    if not np.all(
        np.isfinite(view_matrix)
    ):
        raise ValueError(
            "view_matrix must contain finite values"
        )

    if tau <= 0:
        raise ValueError(
            "tau must be positive"
        )

    # ---------------------------------------------------------
    # Prior covariance
    # ---------------------------------------------------------

    prior_covariance = (
        tau * covariance
    )

    # ---------------------------------------------------------
    # View uncertainty Ω
    # ---------------------------------------------------------

    if view_uncertainty is None:

        view_uncertainty = np.diag(
            view_matrix
            @ prior_covariance
            @ view_matrix.T
        )

        view_uncertainty = np.maximum(
            view_uncertainty,
            1e-12
        )

    else:

        view_uncertainty = np.asarray(
            view_uncertainty,
            dtype=float
        )

        if view_uncertainty.ndim == 1:

            if len(view_uncertainty) != n_views:
                raise ValueError(
                    "view_uncertainty must match "
                    "number of views"
                )

            if np.any(
                view_uncertainty <= 0
            ):
                raise ValueError(
                    "view_uncertainty must be positive"
                )

        elif view_uncertainty.ndim == 2:

            if view_uncertainty.shape != (
                n_views,
                n_views
            ):
                raise ValueError(
                    "view_uncertainty matrix has "
                    "incorrect dimensions"
                )

            if not np.allclose(
                view_uncertainty,
                view_uncertainty.T,
                atol=1e-10
            ):
                raise ValueError(
                    "view_uncertainty must be symmetric"
                )

        else:

            raise ValueError(
                "view_uncertainty must be a vector "
                "or matrix"
            )

    # ---------------------------------------------------------
    # Ω inverse
    # ---------------------------------------------------------

    if view_uncertainty.ndim == 1:

        omega_inverse = np.diag(
            1.0 / view_uncertainty
        )

    else:

        omega_inverse = np.linalg.pinv(
            view_uncertainty
        )

    # ---------------------------------------------------------
    # Matrix calculations
    # ---------------------------------------------------------

    prior_inverse = np.linalg.pinv(
        prior_covariance
    )

    posterior_precision = (
        prior_inverse
        +
        view_matrix.T
        @ omega_inverse
        @ view_matrix
    )

    posterior_covariance = np.linalg.pinv(
        posterior_precision
    )

    posterior_mean = (
        posterior_covariance
        @ (
            prior_inverse
            @ prior_returns
            +
            view_matrix.T
            @ omega_inverse
            @ views
        )
    )

    return posterior_mean


def black_litterman_returns(
    covariance,
    market_weights,
    views,
    view_matrix,
    risk_aversion=2.5,
    tau=0.05,
    view_uncertainty=None
):
    """
    Complete Black-Litterman expected return calculation.

    Returns posterior expected returns.
    """

    covariance = validate_covariance(
        covariance
    )

    prior_returns = market_implied_returns(
        covariance=covariance,
        market_weights=market_weights,
        risk_aversion=risk_aversion
    )

    posterior = posterior_returns(
        prior_returns=prior_returns,
        covariance=covariance,
        views=views,
        view_matrix=view_matrix,
        tau=tau,
        view_uncertainty=view_uncertainty
    )

    return posterior


def black_litterman_series(
    covariance,
    market_weights,
    views,
    view_matrix,
    asset_names=None,
    risk_aversion=2.5,
    tau=0.05,
    view_uncertainty=None
):
    """
    Return Black-Litterman posterior returns
    as a pandas Series.
    """

    covariance_array = validate_covariance(
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

    returns = black_litterman_returns(
        covariance=covariance,
        market_weights=market_weights,
        views=views,
        view_matrix=view_matrix,
        risk_aversion=risk_aversion,
        tau=tau,
        view_uncertainty=view_uncertainty
    )

    return pd.Series(
        returns,
        index=asset_names,
        name="Expected_Return"
    )


def black_litterman_dataframe(
    covariance,
    market_weights,
    views,
    view_matrix,
    asset_names=None,
    risk_aversion=2.5,
    tau=0.05,
    view_uncertainty=None
):
    """
    Return Black-Litterman expected returns
    as a DataFrame.
    """

    returns = black_litterman_series(
        covariance=covariance,
        market_weights=market_weights,
        views=views,
        view_matrix=view_matrix,
        asset_names=asset_names,
        risk_aversion=risk_aversion,
        tau=tau,
        view_uncertainty=view_uncertainty
    )

    return returns.to_frame()


def black_litterman_portfolio_weights(
    covariance,
    market_weights,
    views,
    view_matrix,
    risk_aversion=2.5,
    tau=0.05,
    risk_free_rate=0.0
):
    """
    Calculate a long-only Black-Litterman portfolio.

    Uses the posterior expected returns and a
    closed-form unconstrained tangency direction,
    followed by long-only projection.
    """

    covariance = validate_covariance(
        covariance
    )

    n_assets = covariance.shape[0]

    market_weights = validate_weights(
        market_weights,
        n_assets
    )

    posterior = black_litterman_returns(
        covariance=covariance,
        market_weights=market_weights,
        views=views,
        view_matrix=view_matrix,
        risk_aversion=risk_aversion,
        tau=tau
    )

    excess_returns = (
        posterior - risk_free_rate
    )

    # ---------------------------------------------------------
    # Tangency direction
    # ---------------------------------------------------------

    raw_weights = (
        np.linalg.pinv(covariance)
        @ excess_returns
    )

    # ---------------------------------------------------------
    # Long-only projection
    # ---------------------------------------------------------

    raw_weights = np.maximum(
        raw_weights,
        0.0
    )

    if np.isclose(
        raw_weights.sum(),
        0.0
    ):
        return market_weights.copy()

    weights = (
        raw_weights
        / raw_weights.sum()
    )

    return weights