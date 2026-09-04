import numpy as np
import pandas as pd


def validate_weights(weights):
    """
    Validate portfolio weights.

    Parameters
    ----------
    weights : array-like or pd.Series
        Portfolio weights.

    Returns
    -------
    np.ndarray
        Validated weights.
    """

    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be a 1-dimensional array")

    if len(weights) == 0:
        raise ValueError("weights cannot be empty")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    return weights


def validate_covariance(covariance):
    """
    Validate covariance matrix.

    Parameters
    ----------
    covariance : pd.DataFrame or array-like

    Returns
    -------
    np.ndarray
        Validated covariance matrix.
    """

    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim != 2:
        raise ValueError("covariance must be a 2-dimensional matrix")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square")

    if covariance.shape[0] == 0:
        raise ValueError("covariance matrix cannot be empty")

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "covariance matrix must contain only finite values"
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


def portfolio_return(weights, expected_returns):
    """
    Calculate expected portfolio return.

    Formula:
        Rp = w'μ
    """

    weights = validate_weights(weights)

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    if expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be 1-dimensional"
        )

    if len(weights) != len(expected_returns):
        raise ValueError(
            "weights and expected_returns must have "
            "the same length"
        )

    if not np.all(np.isfinite(expected_returns)):
        raise ValueError(
            "expected_returns must contain only finite values"
        )

    return float(
        np.dot(weights, expected_returns)
    )


def portfolio_variance(weights, covariance):
    """
    Calculate portfolio variance.

    Formula:
        σ²p = w'Σw
    """

    weights = validate_weights(weights)
    covariance = validate_covariance(covariance)

    if len(weights) != covariance.shape[0]:
        raise ValueError(
            "weights and covariance dimensions do not match"
        )

    return float(
        weights @ covariance @ weights
    )


def portfolio_volatility(weights, covariance):
    """
    Calculate portfolio volatility.

    Formula:
        σp = sqrt(w'Σw)
    """

    variance = portfolio_variance(
        weights,
        covariance
    )

    if variance < 0 and not np.isclose(
        variance,
        0.0,
        atol=1e-12
    ):
        raise ValueError(
            "portfolio variance cannot be negative"
        )

    return float(
        np.sqrt(max(variance, 0.0))
    )


def sharpe_ratio(
    weights,
    expected_returns,
    covariance,
    risk_free_rate=0.0
):
    """
    Calculate portfolio Sharpe ratio.

    Formula:
        Sharpe = (Rp - Rf) / σp
    """

    portfolio_expected_return = portfolio_return(
        weights,
        expected_returns
    )

    volatility = portfolio_volatility(
        weights,
        covariance
    )

    excess_return = (
        portfolio_expected_return
        - risk_free_rate
    )

    if np.isclose(volatility, 0.0):
        if np.isclose(excess_return, 0.0):
            return 0.0

        return np.inf if excess_return > 0 else -np.inf

    return float(
        excess_return / volatility
    )


def downside_deviation(
    returns,
    target=0.0,
    annualization=252
):
    """
    Calculate annualized downside deviation.

    Parameters
    ----------
    returns : array-like
        Periodic portfolio returns.
    target : float
        Minimum acceptable return per period.
    annualization : int
        Number of periods per year.
    """

    returns = np.asarray(
        returns,
        dtype=float
    )

    if returns.ndim != 1:
        raise ValueError(
            "returns must be 1-dimensional"
        )

    if len(returns) == 0:
        raise ValueError(
            "returns cannot be empty"
        )

    if not np.all(np.isfinite(returns)):
        raise ValueError(
            "returns must contain only finite values"
        )

    downside = np.minimum(
        returns - target,
        0.0
    )

    return float(
        np.sqrt(
            np.mean(downside ** 2)
        ) * np.sqrt(annualization)
    )


def sortino_ratio(
    returns,
    target=0.0,
    annualization=252
):
    """
    Calculate annualized Sortino ratio.
    """

    returns = np.asarray(
        returns,
        dtype=float
    )

    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError(
            "returns must be a non-empty 1-dimensional array"
        )

    annualized_return = (
        np.mean(returns) * annualization
    )

    downside = downside_deviation(
        returns,
        target,
        annualization
    )

    if np.isclose(downside, 0.0):
        if np.isclose(annualized_return - target, 0.0):
            return 0.0

        return (
            np.inf
            if annualized_return > target
            else -np.inf
        )

    return float(
        (annualized_return - target)
        / downside
    )


def tracking_error(
    portfolio_returns,
    benchmark_returns,
    annualization=252
):
    """
    Calculate annualized tracking error.
    """

    portfolio_returns = np.asarray(
        portfolio_returns,
        dtype=float
    )

    benchmark_returns = np.asarray(
        benchmark_returns,
        dtype=float
    )

    if portfolio_returns.ndim != 1:
        raise ValueError(
            "portfolio_returns must be 1-dimensional"
        )

    if benchmark_returns.ndim != 1:
        raise ValueError(
            "benchmark_returns must be 1-dimensional"
        )

    if len(portfolio_returns) != len(
        benchmark_returns
    ):
        raise ValueError(
            "portfolio_returns and benchmark_returns "
            "must have the same length"
        )

    active_returns = (
        portfolio_returns
        - benchmark_returns
    )

    return float(
        np.std(
            active_returns,
            ddof=1
        ) * np.sqrt(annualization)
    )


def information_ratio(
    portfolio_returns,
    benchmark_returns,
    annualization=252
):
    """
    Calculate information ratio.

    Formula:
        IR = annualized active return / tracking error
    """

    portfolio_returns = np.asarray(
        portfolio_returns,
        dtype=float
    )

    benchmark_returns = np.asarray(
        benchmark_returns,
        dtype=float
    )

    if len(portfolio_returns) != len(
        benchmark_returns
    ):
        raise ValueError(
            "return series must have the same length"
        )

    active_returns = (
        portfolio_returns
        - benchmark_returns
    )

    active_return = (
        np.mean(active_returns)
        * annualization
    )

    error = tracking_error(
        portfolio_returns,
        benchmark_returns,
        annualization
    )

    if np.isclose(error, 0.0):
        if np.isclose(active_return, 0.0):
            return 0.0

        return (
            np.inf
            if active_return > 0
            else -np.inf
        )

    return float(
        active_return / error
    )


def portfolio_metrics(
    weights,
    expected_returns,
    covariance,
    risk_free_rate=0.0
):
    """
    Return the core portfolio metrics together.
    """

    weights = validate_weights(weights)

    expected_returns = np.asarray(
        expected_returns,
        dtype=float
    )

    covariance = validate_covariance(
        covariance
    )

    if len(weights) != len(expected_returns):
        raise ValueError(
            "weights and expected_returns must "
            "have the same length"
        )

    if len(weights) != covariance.shape[0]:
        raise ValueError(
            "weights and covariance dimensions "
            "do not match"
        )

    expected_return = portfolio_return(
        weights,
        expected_returns
    )

    variance = portfolio_variance(
        weights,
        covariance
    )

    volatility = portfolio_volatility(
        weights,
        covariance
    )

    sharpe = sharpe_ratio(
        weights,
        expected_returns,
        covariance,
        risk_free_rate
    )

    return {
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
    }


def portfolio_metrics_table(
    weights,
    expected_returns,
    covariance,
    asset_names=None,
    risk_free_rate=0.0
):
    """
    Return portfolio weights and core statistics
    as a DataFrame.
    """

    weights = validate_weights(weights)

    if asset_names is None:
        asset_names = [
            f"Asset_{i}"
            for i in range(len(weights))
        ]

    if len(asset_names) != len(weights):
        raise ValueError(
            "asset_names and weights must have "
            "the same length"
        )

    metrics = portfolio_metrics(
        weights,
        expected_returns,
        covariance,
        risk_free_rate
    )

    table = pd.DataFrame(
        {
            "Asset": asset_names,
            "Weight": weights,
        }
    )

    table.attrs["portfolio_metrics"] = metrics

    return table