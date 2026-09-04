import numpy as np
import pandas as pd

from src.portfolio.equal_weight import (
    equal_weight_series,
)

from src.portfolio.minimum_variance import (
    minimum_variance_series,
)

from src.portfolio.maximum_sharpe import (
    maximum_sharpe_series,
)

from src.portfolio.risk_parity import (
    risk_parity_series,
)

from src.portfolio.hrp import (
    hrp_series,
)

from src.portfolio.black_litterman import (
    black_litterman_portfolio_weights,
)


SUPPORTED_METHODS = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "hrp",
    "black_litterman",
)


def list_portfolio_methods():
    """
    Return all supported portfolio construction methods.
    """

    return list(SUPPORTED_METHODS)


def validate_method(method):
    """
    Validate portfolio construction method.
    """

    if not isinstance(method, str):
        raise TypeError(
            "method must be a string"
        )

    method = method.lower().strip()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported portfolio method: {method}. "
            f"Available methods: {list(SUPPORTED_METHODS)}"
        )

    return method


def validate_covariance(covariance):
    """
    Validate covariance matrix.
    """

    if isinstance(
        covariance,
        pd.DataFrame
    ):
        covariance_array = covariance.values

    else:
        covariance_array = np.asarray(
            covariance,
            dtype=float
        )

    if covariance_array.ndim != 2:
        raise ValueError(
            "covariance must be 2-dimensional"
        )

    if (
        covariance_array.shape[0]
        != covariance_array.shape[1]
    ):
        raise ValueError(
            "covariance matrix must be square"
        )

    if covariance_array.shape[0] == 0:
        raise ValueError(
            "covariance matrix cannot be empty"
        )

    if not np.all(
        np.isfinite(covariance_array)
    ):
        raise ValueError(
            "covariance must contain finite values"
        )

    if not np.allclose(
        covariance_array,
        covariance_array.T,
        atol=1e-10
    ):
        raise ValueError(
            "covariance matrix must be symmetric"
        )

    return covariance_array


def get_asset_names(
    covariance,
    n_assets
):
    """
    Determine asset names from covariance matrix.
    """

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
            "asset names do not match "
            "number of assets"
        )

    return asset_names


def validate_weights(weights):
    """
    Validate portfolio weights.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim != 1:
        raise ValueError(
            "weights must be 1-dimensional"
        )

    if len(weights) == 0:
        raise ValueError(
            "weights cannot be empty"
        )

    if not np.all(
        np.isfinite(weights)
    ):
        raise ValueError(
            "weights must contain finite values"
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-8
    ):
        raise ValueError(
            "portfolio weights must sum to 1"
        )

    return weights


def build_equal_weight(
    covariance
):
    """
    Build equal-weight portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    asset_names = get_asset_names(
        covariance,
        n_assets
    )

    return equal_weight_series(
        asset_names
    )


def build_minimum_variance(
    covariance,
    long_only=True
):
    """
    Build minimum-variance portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    asset_names = get_asset_names(
        covariance,
        n_assets
    )

    return minimum_variance_series(
        covariance=covariance_array,
        asset_names=asset_names,
        long_only=long_only
    )


def build_maximum_sharpe(
    expected_returns,
    covariance,
    risk_free_rate=0.0,
    long_only=True
):
    """
    Build maximum-Sharpe portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    if expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be 1-dimensional"
        )

    if len(expected_returns) != (
        covariance_array.shape[0]
    ):
        raise ValueError(
            "expected_returns and covariance "
            "dimensions do not match"
        )

    asset_names = get_asset_names(
        covariance,
        len(expected_returns)
    )

    return maximum_sharpe_series(
        expected_returns=expected_returns,
        covariance=covariance_array,
        asset_names=asset_names,
        risk_free_rate=risk_free_rate,
        long_only=long_only
    )


def build_risk_parity(
    covariance,
    target_risk_budgets=None
):
    """
    Build risk-parity portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    asset_names = get_asset_names(
        covariance,
        n_assets
    )

    return risk_parity_series(
        covariance=covariance_array,
        asset_names=asset_names,
        target_risk_budgets=target_risk_budgets
    )


def build_hrp(
    covariance
):
    """
    Build Hierarchical Risk Parity portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    asset_names = get_asset_names(
        covariance,
        n_assets
    )

    return hrp_series(
        covariance=covariance_array,
        asset_names=asset_names
    )


def build_black_litterman(
    covariance,
    market_weights,
    views,
    view_matrix,
    risk_aversion=2.5,
    tau=0.05,
    risk_free_rate=0.0
):
    """
    Build long-only Black-Litterman portfolio.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    asset_names = get_asset_names(
        covariance,
        n_assets
    )

    market_weights = np.asarray(
        market_weights,
        dtype=float
    )

    if len(market_weights) != n_assets:
        raise ValueError(
            "market_weights must match "
            "number of assets"
        )

    weights = black_litterman_portfolio_weights(
        covariance=covariance_array,
        market_weights=market_weights,
        views=views,
        view_matrix=view_matrix,
        risk_aversion=risk_aversion,
        tau=tau,
        risk_free_rate=risk_free_rate
    )

    weights = validate_weights(
        weights
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def build_portfolio(
    method,
    covariance,
    expected_returns=None,
    risk_free_rate=0.0,
    market_weights=None,
    views=None,
    view_matrix=None,
    target_risk_budgets=None,
    long_only=True,
    risk_aversion=2.5,
    tau=0.05
):
    """
    Main portfolio construction interface.

    Parameters
    ----------
    method : str
        Portfolio method.

    covariance : pd.DataFrame or array-like
        Covariance matrix.

    expected_returns : array-like, optional
        Required for maximum Sharpe.

    risk_free_rate : float
        Risk-free rate.

    market_weights : array-like, optional
        Required for Black-Litterman.

    views : array-like, optional
        Required for Black-Litterman.

    view_matrix : array-like, optional
        Required for Black-Litterman.

    target_risk_budgets : array-like, optional
        Optional risk budgets for risk parity.

    long_only : bool
        Restrict portfolio to non-negative weights.

    risk_aversion : float
        Black-Litterman risk-aversion coefficient.

    tau : float
        Black-Litterman uncertainty parameter.

    Returns
    -------
    pd.Series
        Portfolio weights.
    """

    method = validate_method(
        method
    )

    if method == "equal_weight":

        weights = build_equal_weight(
            covariance
        )

    elif method == "minimum_variance":

        weights = build_minimum_variance(
            covariance,
            long_only=long_only
        )

    elif method == "maximum_sharpe":

        if expected_returns is None:
            raise ValueError(
                "expected_returns is required "
                "for maximum_sharpe"
            )

        weights = build_maximum_sharpe(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_free_rate=risk_free_rate,
            long_only=long_only
        )

    elif method == "risk_parity":

        weights = build_risk_parity(
            covariance=covariance,
            target_risk_budgets=target_risk_budgets
        )

    elif method == "hrp":

        weights = build_hrp(
            covariance=covariance
        )

    elif method == "black_litterman":

        if market_weights is None:
            raise ValueError(
                "market_weights is required "
                "for black_litterman"
            )

        if views is None:
            raise ValueError(
                "views is required "
                "for black_litterman"
            )

        if view_matrix is None:
            raise ValueError(
                "view_matrix is required "
                "for black_litterman"
            )

        weights = build_black_litterman(
            covariance=covariance,
            market_weights=market_weights,
            views=views,
            view_matrix=view_matrix,
            risk_aversion=risk_aversion,
            tau=tau,
            risk_free_rate=risk_free_rate
        )

    else:
        raise ValueError(
            f"Unsupported portfolio method: {method}"
        )

    validate_weights(
        weights.values
    )

    return weights


def build_all_portfolios(
    covariance,
    expected_returns,
    risk_free_rate=0.0,
    market_weights=None,
    views=None,
    view_matrix=None
):
    """
    Build all standard portfolio methods.

    Black-Litterman is included only when the required
    inputs are supplied.

    Returns
    -------
    pd.DataFrame
        Assets × portfolio methods.
    """

    results = {}

    methods = [
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
        "risk_parity",
        "hrp",
    ]

    for method in methods:

        results[method] = build_portfolio(
            method=method,
            covariance=covariance,
            expected_returns=expected_returns,
            risk_free_rate=risk_free_rate
        )

    if (
        market_weights is not None
        and views is not None
        and view_matrix is not None
    ):

        results["black_litterman"] = (
            build_portfolio(
                method="black_litterman",
                covariance=covariance,
                expected_returns=expected_returns,
                risk_free_rate=risk_free_rate,
                market_weights=market_weights,
                views=views,
                view_matrix=view_matrix
            )
        )

    return pd.DataFrame(results)


def portfolio_summary(
    weights,
    covariance,
    expected_returns=None,
    risk_free_rate=0.0
):
    """
    Create a simple summary for portfolio weights.
    """

    weights = pd.Series(
        weights,
        dtype=float
    )

    covariance_array = validate_covariance(
        covariance
    )

    if len(weights) != covariance_array.shape[0]:
        raise ValueError(
            "weights and covariance dimensions "
            "do not match"
        )

    weights_array = validate_weights(
        weights.values
    )

    variance = (
        weights_array
        @ covariance_array
        @ weights_array
    )

    volatility = np.sqrt(
        max(variance, 0.0)
    )

    result = {
        "portfolio_volatility": float(
            volatility
        ),
        "gross_exposure": float(
            np.sum(
                np.abs(weights_array)
            )
        ),
        "net_exposure": float(
            np.sum(weights_array)
        ),
        "number_of_assets": int(
            np.count_nonzero(
                np.abs(weights_array)
                > 1e-10
            )
        ),
    }

    if expected_returns is not None:

        expected_returns = np.asarray(
            expected_returns,
            dtype=float
        )

        portfolio_return = np.dot(
            weights_array,
            expected_returns
        )

        result["expected_return"] = float(
            portfolio_return
        )

        if not np.isclose(
            volatility,
            0.0
        ):

            result["sharpe_ratio"] = float(
                (
                    portfolio_return
                    - risk_free_rate
                )
                / volatility
            )

        else:

            result["sharpe_ratio"] = 0.0

    return result