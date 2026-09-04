import numpy as np


def validate_weight_bounds(
    lower_bounds,
    upper_bounds,
    n_assets
):
    """
    Validate lower and upper portfolio weight bounds.
    """

    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    lower_bounds = np.asarray(
        lower_bounds,
        dtype=float
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=float
    )

    if len(lower_bounds) != n_assets:
        raise ValueError(
            "lower_bounds must match number of assets"
        )

    if len(upper_bounds) != n_assets:
        raise ValueError(
            "upper_bounds must match number of assets"
        )

    if not np.all(np.isfinite(lower_bounds)):
        raise ValueError(
            "lower_bounds must contain finite values"
        )

    if not np.all(np.isfinite(upper_bounds)):
        raise ValueError(
            "upper_bounds must contain finite values"
        )

    if np.any(lower_bounds > upper_bounds):
        raise ValueError(
            "lower_bounds cannot exceed upper_bounds"
        )

    return lower_bounds, upper_bounds


def long_only_constraint(weights):
    """
    Check whether all portfolio weights are non-negative.
    """

    weights = np.asarray(weights, dtype=float)

    return bool(np.all(weights >= 0))


def fully_invested_constraint(
    weights,
    tolerance=1e-8
):
    """
    Check whether portfolio weights sum to 1.
    """

    weights = np.asarray(weights, dtype=float)

    return bool(
        np.isclose(
            np.sum(weights),
            1.0,
            atol=tolerance
        )
    )


def weight_bounds_constraint(
    weights,
    lower_bounds,
    upper_bounds,
    tolerance=1e-8
):
    """
    Check whether every asset weight satisfies
    its lower and upper bounds.
    """

    weights = np.asarray(weights, dtype=float)

    lower_bounds, upper_bounds = (
        validate_weight_bounds(
            lower_bounds,
            upper_bounds,
            len(weights)
        )
    )

    return bool(
        np.all(weights >= lower_bounds - tolerance)
        and
        np.all(weights <= upper_bounds + tolerance)
    )


def maximum_position_constraint(
    weights,
    maximum_weight
):
    """
    Check whether no individual position exceeds
    the maximum allowed weight.
    """

    weights = np.asarray(weights, dtype=float)

    if maximum_weight < 0:
        raise ValueError(
            "maximum_weight must be non-negative"
        )

    return bool(
        np.max(np.abs(weights))
        <= maximum_weight
    )


def gross_exposure(weights):
    """
    Calculate gross portfolio exposure.

    Formula:
        Gross Exposure = Σ |wi|
    """

    weights = np.asarray(weights, dtype=float)

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    return float(
        np.sum(np.abs(weights))
    )


def net_exposure(weights):
    """
    Calculate net portfolio exposure.

    Formula:
        Net Exposure = Σ wi
    """

    weights = np.asarray(weights, dtype=float)

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    return float(
        np.sum(weights)
    )


def leverage_ratio(weights):
    """
    Calculate portfolio leverage.

    For a fully invested long-only portfolio:
        leverage = 1

    Formula:
        Leverage = Σ |wi|
    """

    return gross_exposure(weights)


def leverage_constraint(
    weights,
    maximum_leverage,
    tolerance=1e-8
):
    """
    Check whether portfolio leverage is within
    the permitted limit.
    """

    if maximum_leverage < 0:
        raise ValueError(
            "maximum_leverage must be non-negative"
        )

    return bool(
        leverage_ratio(weights)
        <= maximum_leverage + tolerance
    )


def turnover(
    old_weights,
    new_weights
):
    """
    Calculate portfolio turnover.

    Formula:
        Turnover = Σ |new_weight - old_weight|
    """

    old_weights = np.asarray(
        old_weights,
        dtype=float
    )

    new_weights = np.asarray(
        new_weights,
        dtype=float
    )

    if old_weights.shape != new_weights.shape:
        raise ValueError(
            "old_weights and new_weights "
            "must have the same shape"
        )

    if not (
        np.all(np.isfinite(old_weights))
        and
        np.all(np.isfinite(new_weights))
    ):
        raise ValueError(
            "weights must contain finite values"
        )

    return float(
        np.sum(
            np.abs(
                new_weights - old_weights
            )
        )
    )


def turnover_constraint(
    old_weights,
    new_weights,
    maximum_turnover,
    tolerance=1e-8
):
    """
    Check whether portfolio turnover is within
    the permitted limit.
    """

    if maximum_turnover < 0:
        raise ValueError(
            "maximum_turnover must be non-negative"
        )

    return bool(
        turnover(
            old_weights,
            new_weights
        )
        <= maximum_turnover + tolerance
    )


def sector_exposure(
    weights,
    sectors
):
    """
    Calculate portfolio exposure by sector.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    sectors : array-like
        Sector label for each asset.

    Returns
    -------
    dict
        Sector exposure.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    sectors = np.asarray(sectors)

    if len(weights) != len(sectors):
        raise ValueError(
            "weights and sectors must have "
            "the same length"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    exposure = {}

    for weight, sector in zip(
        weights,
        sectors
    ):
        exposure[sector] = (
            exposure.get(sector, 0.0)
            + weight
        )

    return exposure


def maximum_sector_exposure_constraint(
    weights,
    sectors,
    maximum_exposure,
    tolerance=1e-8
):
    """
    Check whether no sector exceeds the
    maximum permitted exposure.
    """

    if maximum_exposure < 0:
        raise ValueError(
            "maximum_exposure must be non-negative"
        )

    exposures = sector_exposure(
        weights,
        sectors
    )

    return bool(
        all(
            abs(exposure)
            <= maximum_exposure + tolerance
            for exposure in exposures.values()
        )
    )


def diversification_constraint(
    weights,
    minimum_assets
):
    """
    Check whether the portfolio contains at least
    the required number of non-zero positions.
    """

    if minimum_assets < 1:
        raise ValueError(
            "minimum_assets must be at least 1"
        )

    weights = np.asarray(
        weights,
        dtype=float
    )

    active_assets = np.count_nonzero(
        np.abs(weights) > 1e-10
    )

    return bool(
        active_assets >= minimum_assets
    )


def validate_portfolio_constraints(
    weights,
    fully_invested=True,
    long_only=False,
    maximum_weight=None,
    maximum_leverage=None,
    tolerance=1e-8
):
    """
    Validate a portfolio against common constraints.

    Returns
    -------
    bool
        True when all requested constraints pass.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim != 1:
        raise ValueError(
            "weights must be 1-dimensional"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    if fully_invested:
        if not fully_invested_constraint(
            weights,
            tolerance
        ):
            return False

    if long_only:
        if not long_only_constraint(weights):
            return False

    if maximum_weight is not None:
        if not maximum_position_constraint(
            weights,
            maximum_weight
        ):
            return False

    if maximum_leverage is not None:
        if not leverage_constraint(
            weights,
            maximum_leverage,
            tolerance
        ):
            return False

    return True