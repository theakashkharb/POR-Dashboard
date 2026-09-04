import numpy as np
import pandas as pd


def equal_weight(n_assets):
    """
    Create an equal-weight portfolio.

    Each asset receives:
        1 / N

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Portfolio weights.
    """

    if not isinstance(n_assets, (int, np.integer)):
        raise TypeError("n_assets must be an integer")

    if n_assets <= 0:
        raise ValueError("n_assets must be greater than zero")

    return np.full(
        n_assets,
        1.0 / n_assets
    )


def equal_weight_series(asset_names):
    """
    Create equal weights with asset names.

    Parameters
    ----------
    asset_names : sequence
        Asset identifiers.

    Returns
    -------
    pd.Series
        Equal portfolio weights indexed by asset.
    """

    asset_names = list(asset_names)

    if len(asset_names) == 0:
        raise ValueError(
            "asset_names cannot be empty"
        )

    if len(set(asset_names)) != len(asset_names):
        raise ValueError(
            "asset_names must be unique"
        )

    weights = equal_weight(
        len(asset_names)
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def equal_weight_dataframe(asset_names):
    """
    Return equal-weight portfolio as a DataFrame.
    """

    weights = equal_weight_series(
        asset_names
    )

    return weights.to_frame()


def validate_equal_weight(weights):
    """
    Validate that weights represent an equal-weight
    fully invested portfolio.
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

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    expected_weight = 1.0 / len(weights)

    return bool(
        np.allclose(
            weights,
            expected_weight,
            atol=1e-10
        )
        and
        np.isclose(
            weights.sum(),
            1.0,
            atol=1e-10
        )
    )