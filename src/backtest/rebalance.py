from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.validation import (
    validate_returns,
    validate_selected_assets,
)

from src.backtest.turnover import (
    apply_turnover_constraint,
)


# ============================================================
# EXPECTED RETURNS
# ============================================================

def historical_expected_returns(
    returns,
    annualization=252,
):
    """
    Calculate annualized historical arithmetic
    expected returns.
    """
    returns = validate_returns(returns)

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    return returns.mean() * annualization


# ============================================================
# COVARIANCE
# ============================================================

def covariance_matrix(
    returns,
    annualization=252,
):
    """
    Calculate annualized covariance matrix
    directly from a return matrix.
    """
    returns = validate_returns(returns)

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    return returns.cov() * annualization


# ============================================================
# OPTIMIZER INTERFACE
# ============================================================

def optimize_portfolio(
    expected_returns,
    covariance,
    optimizer,
    old_weights=None,
    max_turnover=None,
    **optimizer_kwargs,
):
    """
    Run a portfolio optimizer using the optimizer's
    actual interface.
    """

    if not callable(optimizer):
        raise TypeError(
            "optimizer must be callable"
        )

    expected_returns = pd.Series(
        expected_returns,
        dtype=float,
    )

    covariance = pd.DataFrame(
        covariance,
        dtype=float,
    )

    if expected_returns.empty:
        raise ValueError(
            "expected_returns cannot be empty"
        )

    if covariance.empty:
        raise ValueError(
            "covariance cannot be empty"
        )

    if not np.isfinite(
        expected_returns.values
    ).all():
        raise ValueError(
            "expected_returns contain non-finite values"
        )

    if not np.isfinite(
        covariance.values
    ).all():
        raise ValueError(
            "covariance contains non-finite values"
        )

    assets = expected_returns.index

    missing_assets = [
        asset
        for asset in assets
        if asset not in covariance.index
        or asset not in covariance.columns
    ]

    if missing_assets:
        raise ValueError(
            "covariance missing assets: "
            f"{missing_assets}"
        )

    covariance = covariance.loc[
        assets,
        assets,
    ]

    # --------------------------------------------------------
    # Identify optimizer
    # --------------------------------------------------------

    optimizer_name = getattr(
        optimizer,
        "__name__",
        "",
    )

    # --------------------------------------------------------
    # Equal Weight
    # Signature:
    #     equal_weight(n_assets)
    # --------------------------------------------------------

    if optimizer_name == "equal_weight":

        target_weights = optimizer(
            len(assets)
        )

    # --------------------------------------------------------
    # Minimum Variance
    # Signature:
    #     minimum_variance_weights(
    #         covariance,
    #         long_only=True,
    #         ...
    #     )
    # --------------------------------------------------------

    elif optimizer_name == "minimum_variance_weights":

        target_weights = optimizer(
            covariance,
            **optimizer_kwargs,
        )

    # --------------------------------------------------------
    # Maximum Sharpe
    # Signature:
    #     maximum_sharpe(
    #         expected_returns,
    #         covariance,
    #         ...
    #     )
    # --------------------------------------------------------

    elif optimizer_name == "maximum_sharpe":

        target_weights = optimizer(
            expected_returns,
            covariance,
            **optimizer_kwargs,
        )

    # --------------------------------------------------------
    # Risk Parity
    # Signature:
    #     risk_parity_weights(
    #         covariance,
    #         target_risk_budgets=None,
    #         ...
    #     )
    # --------------------------------------------------------

    elif optimizer_name == "risk_parity_weights":

        target_weights = optimizer(
            covariance,
            **optimizer_kwargs,
        )

    # --------------------------------------------------------
    # HRP
    # Signature:
    #     hrp_weights(covariance)
    # --------------------------------------------------------

    elif optimizer_name == "hrp_weights":

        target_weights = optimizer(
            covariance
        )

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unsupported optimizer interface: "
            f"{optimizer_name}"
        )

    # --------------------------------------------------------
    # Convert result to Series
    # --------------------------------------------------------

    if isinstance(
        target_weights,
        pd.DataFrame,
    ):

        if target_weights.shape[1] != 1:
            raise ValueError(
                "Optimizer returned a multi-column DataFrame"
            )

        target_weights = (
            target_weights.iloc[:, 0]
        )

    target_weights = pd.Series(
        target_weights,
        dtype=float,
    )

    # --------------------------------------------------------
    # Align optimizer output
    # --------------------------------------------------------

    if isinstance(
        target_weights.index,
        pd.RangeIndex,
    ):

        if len(target_weights) != len(assets):
            raise ValueError(
                "Optimizer returned incorrect "
                "number of weights"
            )

        target_weights.index = assets

    else:

        missing_weights = [
            asset
            for asset in assets
            if asset not in target_weights.index
        ]

        if missing_weights:
            raise ValueError(
                "Optimizer output missing assets: "
                f"{missing_weights}"
            )

        target_weights = (
            target_weights.loc[assets]
        )

    # --------------------------------------------------------
    # Validate weights
    # --------------------------------------------------------

    if not np.isfinite(
        target_weights.values
    ).all():
        raise ValueError(
            "Optimizer returned non-finite weights"
        )

    target_weights = target_weights.clip(
        lower=0.0
    )

    if target_weights.sum() <= 0:
        raise ValueError(
            "Optimizer returned zero weights"
        )

    target_weights = (
        target_weights
        / target_weights.sum()
    )

    # --------------------------------------------------------
    # Turnover constraint
    # --------------------------------------------------------

    if (
        old_weights is not None
        and max_turnover is not None
    ):

        target_weights = (
            apply_turnover_constraint(
                old_weights=old_weights,
                target_weights=target_weights,
                max_turnover=max_turnover,
            )
        )

    return target_weights


# ============================================================
# REBALANCE PORTFOLIO
# ============================================================

def rebalance_portfolio(
    training_returns,
    selected_assets,
    optimizer,
    old_weights=None,
    max_turnover=None,
    expected_return_method="historical",
    annualization=252,
    **optimizer_kwargs,
):
    """
    Construct an optimized portfolio using
    historical training data.

    Returns
    -------
    weights : pd.Series
        Optimized portfolio weights.

    expected_returns : pd.Series
        Annualized expected returns.

    covariance : pd.DataFrame
        Annualized covariance matrix.
    """

    # --------------------------------------------------------
    # Validate returns
    # --------------------------------------------------------

    training_returns = validate_returns(
        training_returns
    )

    # --------------------------------------------------------
    # Validate selected assets
    # --------------------------------------------------------

    selected_assets = validate_selected_assets(
        training_returns,
        selected_assets,
    )

    # --------------------------------------------------------
    # Select assets
    # --------------------------------------------------------

    training_subset = (
        training_returns[
            selected_assets
        ]
        .copy()
    )

    training_subset = (
        training_subset
        .dropna(
            axis=1,
            how="all",
        )
    )

    if training_subset.empty:
        raise ValueError(
            "No assets available for optimization"
        )

    # --------------------------------------------------------
    # Remove rows where every asset is missing
    # --------------------------------------------------------

    training_subset = (
        training_subset
        .dropna(
            how="all"
        )
    )

    if training_subset.empty:
        raise ValueError(
            "No valid training observations available"
        )

    # --------------------------------------------------------
    # Expected returns
    # --------------------------------------------------------

    if expected_return_method == "historical":

        expected_returns = (
            historical_expected_returns(
                training_subset,
                annualization=annualization,
            )
        )

    else:

        raise ValueError(
            "Unknown expected return method: "
            f"{expected_return_method}"
        )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    covariance = covariance_matrix(
        training_subset,
        annualization=annualization,
    )

    # --------------------------------------------------------
    # Keep common valid assets
    # --------------------------------------------------------

    valid_assets = (
        expected_returns.index
        .intersection(
            covariance.index
        )
    )

    if len(valid_assets) == 0:
        raise ValueError(
            "No valid assets available for optimization"
        )

    expected_returns = (
        expected_returns
        .loc[valid_assets]
    )

    covariance = (
        covariance
        .loc[
            valid_assets,
            valid_assets,
        ]
    )

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        optimizer=optimizer,
        old_weights=old_weights,
        max_turnover=max_turnover,
        **optimizer_kwargs,
    )

    return (
        weights,
        expected_returns,
        covariance,
    )