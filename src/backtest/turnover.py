from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# TURNOVER
# ============================================================

def calculate_turnover(
    old_weights,
    new_weights,
):
    """
    Calculate one-way portfolio turnover.

    Turnover:
        0.5 * sum(abs(new_weight - old_weight))
    """
    old_weights = pd.Series(
        old_weights,
        dtype=float,
    )

    new_weights = pd.Series(
        new_weights,
        dtype=float,
    )

    if old_weights.empty:
        raise ValueError(
            "old_weights cannot be empty"
        )

    if new_weights.empty:
        raise ValueError(
            "new_weights cannot be empty"
        )

    if not np.isfinite(
        old_weights.values
    ).all():
        raise ValueError(
            "old_weights contain non-finite values"
        )

    if not np.isfinite(
        new_weights.values
    ).all():
        raise ValueError(
            "new_weights contain non-finite values"
        )

    assets = (
        old_weights.index
        .union(new_weights.index)
    )

    old_weights = old_weights.reindex(
        assets,
        fill_value=0.0,
    )

    new_weights = new_weights.reindex(
        assets,
        fill_value=0.0,
    )

    turnover = (
        0.5
        * np.abs(
            new_weights - old_weights
        ).sum()
    )

    return float(turnover)


# ============================================================
# TURNOVER CONSTRAINT
# ============================================================

def apply_turnover_constraint(
    old_weights,
    target_weights,
    max_turnover,
):
    """
    Move the current portfolio toward the target
    without exceeding max_turnover.

    Both portfolios are normalized to be fully invested.
    """
    old_weights = pd.Series(
        old_weights,
        dtype=float,
    )

    target_weights = pd.Series(
        target_weights,
        dtype=float,
    )

    if old_weights.empty:
        raise ValueError(
            "old_weights cannot be empty"
        )

    if target_weights.empty:
        raise ValueError(
            "target_weights cannot be empty"
        )

    if not np.isfinite(
        old_weights.values
    ).all():
        raise ValueError(
            "old_weights contain non-finite values"
        )

    if not np.isfinite(
        target_weights.values
    ).all():
        raise ValueError(
            "target_weights contain non-finite values"
        )

    if max_turnover is None:
        raise ValueError(
            "max_turnover cannot be None"
        )

    if not np.isfinite(
        max_turnover
    ):
        raise ValueError(
            "max_turnover must be finite"
        )

    if max_turnover < 0:
        raise ValueError(
            "max_turnover cannot be negative"
        )

    if old_weights.sum() == 0:
        raise ValueError(
            "Old portfolio weights sum to zero"
        )

    if target_weights.sum() == 0:
        raise ValueError(
            "Target portfolio weights sum to zero"
        )

    # --------------------------------------------------------
    # Include every asset appearing in either portfolio
    # --------------------------------------------------------

    assets = (
        old_weights.index
        .union(target_weights.index)
    )

    old_weights = old_weights.reindex(
        assets,
        fill_value=0.0,
    )

    target_weights = target_weights.reindex(
        assets,
        fill_value=0.0,
    )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    target_weights = target_weights.clip(
        lower=0.0
    )

    if target_weights.sum() == 0:
        raise ValueError(
            "Target portfolio contains no positive weights"
        )

    target_weights = (
        target_weights
        / target_weights.sum()
    )

    # --------------------------------------------------------
    # Normalize old portfolio
    # --------------------------------------------------------

    old_weights = old_weights.clip(
        lower=0.0
    )

    if old_weights.sum() == 0:
        raise ValueError(
            "Old portfolio contains no positive weights"
        )

    old_weights = (
        old_weights
        / old_weights.sum()
    )

    # --------------------------------------------------------
    # Calculate required turnover
    # --------------------------------------------------------

    turnover = calculate_turnover(
        old_weights,
        target_weights,
    )

    if turnover <= max_turnover + 1e-12:
        return target_weights

    # --------------------------------------------------------
    # Interpolate between old and target portfolios
    # --------------------------------------------------------

    difference = (
        target_weights
        - old_weights
    )

    difference_abs_sum = (
        np.abs(difference).sum()
    )

    if difference_abs_sum == 0:
        return old_weights

    scale = (
        2.0
        * max_turnover
        / difference_abs_sum
    )

    scale = min(
        1.0,
        scale,
    )

    constrained_weights = (
        old_weights
        + difference * scale
    )

    constrained_weights = (
        constrained_weights.clip(
            lower=0.0
        )
    )

    constrained_weights = (
        constrained_weights
        / constrained_weights.sum()
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    final_turnover = calculate_turnover(
        old_weights,
        constrained_weights,
    )

    if final_turnover > max_turnover + 1e-10:
        raise ValueError(
            "Turnover constraint could not be satisfied"
        )

    return constrained_weights


# ============================================================
# TURNOVER HISTORY
# ============================================================

def calculate_turnover_history(
    weight_history,
):
    """
    Calculate turnover between consecutive portfolios.

    Parameters
    ----------
    weight_history : DataFrame
        Rows represent rebalance dates.
        Columns represent assets.
    """
    if not isinstance(
        weight_history,
        pd.DataFrame,
    ):
        raise TypeError(
            "weight_history must be a pandas DataFrame"
        )

    if weight_history.empty:
        return pd.Series(
            dtype=float,
            name="turnover",
        )

    turnovers = []

    for i in range(1, len(weight_history)):

        old_weights = (
            weight_history.iloc[i - 1]
        )

        new_weights = (
            weight_history.iloc[i]
        )

        turnovers.append(
            calculate_turnover(
                old_weights,
                new_weights,
            )
        )

    index = weight_history.index[1:]

    return pd.Series(
        turnovers,
        index=index,
        name="turnover",
    )