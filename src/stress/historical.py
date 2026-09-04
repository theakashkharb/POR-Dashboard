from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    """
    Validate asset return data.
    """
    returns = pd.DataFrame(
        returns,
        dtype=float,
    )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    if not np.isfinite(
        returns.values
    ).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    return returns


def _validate_weights(weights, assets):
    """
    Validate and align portfolio weights.
    """
    weights = pd.Series(
        weights,
        dtype=float,
    )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    if not np.isfinite(
        weights.values
    ).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    if isinstance(
        weights.index,
        pd.RangeIndex,
    ):
        if len(weights) != len(assets):
            raise ValueError(
                "weights length must match "
                "number of assets."
            )

        weights.index = assets

    else:
        missing = [
            asset
            for asset in assets
            if asset not in weights.index
        ]

        if missing:
            raise ValueError(
                f"weights missing assets: {missing}"
            )

        weights = weights.loc[assets]

    if (
        weights.values < 0
    ).any():
        raise ValueError(
            "negative weights are not supported."
        )

    total = float(
        weights.sum()
    )

    if total <= 0:
        raise ValueError(
            "weights must have a positive sum."
        )

    return weights / total


# ============================================================
# HISTORICAL WINDOW
# ============================================================

def select_historical_window(
    returns,
    start_date=None,
    end_date=None,
):
    """
    Extract a historical period from return data.

    Parameters
    ----------
    returns : DataFrame
        Asset return matrix with a DatetimeIndex.

    start_date : optional
        Beginning of stress period.

    end_date : optional
        End of stress period.

    Returns
    -------
    DataFrame
        Selected historical returns.
    """
    returns = _validate_returns(
        returns
    )

    if not isinstance(
        returns.index,
        pd.DatetimeIndex,
    ):
        try:
            returns.index = pd.to_datetime(
                returns.index
            )
        except Exception as exc:
            raise ValueError(
                "returns index must contain valid dates."
            ) from exc

    if start_date is not None:
        start_date = pd.Timestamp(
            start_date
        )

    if end_date is not None:
        end_date = pd.Timestamp(
            end_date
        )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise ValueError(
            "start_date cannot be after end_date."
        )

    selected = returns

    if start_date is not None:
        selected = selected.loc[
            selected.index >= start_date
        ]

    if end_date is not None:
        selected = selected.loc[
            selected.index <= end_date
        ]

    if selected.empty:
        raise ValueError(
            "historical window contains no observations."
        )

    return selected


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def calculate_historical_portfolio_returns(
    returns,
    weights,
):
    """
    Calculate historical portfolio returns
    from asset returns and portfolio weights.
    """
    returns = _validate_returns(
        returns
    )

    weights = _validate_weights(
        weights,
        returns.columns,
    )

    portfolio_returns = (
        returns
        @ weights
    )

    portfolio_returns.name = (
        "portfolio_return"
    )

    return portfolio_returns


# ============================================================
# HISTORICAL STRESS STATISTICS
# ============================================================

def historical_stress_statistics(
    portfolio_returns,
):
    """
    Calculate stress statistics for
    historical portfolio returns.
    """
    portfolio_returns = pd.Series(
        portfolio_returns,
        dtype=float,
    ).dropna()

    if portfolio_returns.empty:
        raise ValueError(
            "portfolio_returns cannot be empty."
        )

    minimum = float(
        portfolio_returns.min()
    )

    maximum = float(
        portfolio_returns.max()
    )

    cumulative_return = float(
        (1 + portfolio_returns).prod()
        - 1
    )

    volatility = float(
        portfolio_returns.std()
        * np.sqrt(252)
    )

    return {
        "observations": int(
            len(portfolio_returns)
        ),
        "worst_daily_return": minimum,
        "best_daily_return": maximum,
        "cumulative_return":
            cumulative_return,
        "annualized_volatility":
            volatility,
    }


# ============================================================
# WORST PERIOD
# ============================================================

def worst_historical_day(
    returns,
    weights,
):
    """
    Find the worst historical portfolio day.
    """
    portfolio_returns = (
        calculate_historical_portfolio_returns(
            returns,
            weights,
        )
    )

    worst_date = (
        portfolio_returns.idxmin()
    )

    return {
        "date": worst_date,
        "return": float(
            portfolio_returns.loc[
                worst_date
            ]
        ),
    }


def worst_historical_period(
    returns,
    weights,
    window=20,
):
    """
    Find the worst cumulative return over
    a rolling historical window.

    Parameters
    ----------
    window : int
        Number of observations in the
        rolling stress period.
    """
    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    portfolio_returns = (
        calculate_historical_portfolio_returns(
            returns,
            weights,
        )
    )

    if len(portfolio_returns) < window:
        raise ValueError(
            "window cannot exceed "
            "number of observations."
        )

    rolling_returns = (
        (1 + portfolio_returns)
        .rolling(window)
        .apply(
            np.prod,
            raw=True,
        )
        - 1
    )

    worst_end = (
        rolling_returns.idxmin()
    )

    worst_return = float(
        rolling_returns.loc[
            worst_end
        ]
    )

    end_position = (
        portfolio_returns.index
        .get_loc(worst_end)
    )

    start_position = (
        end_position - window + 1
    )

    start_date = (
        portfolio_returns.index[
            start_position
        ]
    )

    return {
        "start_date": start_date,
        "end_date": worst_end,
        "window": window,
        "return": worst_return,
    }


# ============================================================
# HISTORICAL SCENARIO
# ============================================================

def run_historical_scenario(
    returns,
    weights,
    start_date,
    end_date,
    scenario_name=None,
):
    """
    Run one historical stress scenario.
    """
    selected = select_historical_window(
        returns,
        start_date=start_date,
        end_date=end_date,
    )

    portfolio_returns = (
        calculate_historical_portfolio_returns(
            selected,
            weights,
        )
    )

    statistics = (
        historical_stress_statistics(
            portfolio_returns
        )
    )

    result = {
        "scenario":
            scenario_name
            or "Historical Scenario",
        "start_date":
            selected.index.min(),
        "end_date":
            selected.index.max(),
        **statistics,
    }

    return result


# ============================================================
# SCENARIO COMPARISON
# ============================================================

def compare_historical_scenarios(
    returns,
    weights,
    scenarios,
):
    """
    Compare multiple historical stress periods.

    Parameters
    ----------
    scenarios : dict

        Example:

        {
            "Scenario A": (
                "2020-02-01",
                "2020-04-30",
            ),
            "Scenario B": (
                "2022-01-01",
                "2022-06-30",
            ),
        }

    Returns
    -------
    DataFrame
    """
    if not scenarios:
        raise ValueError(
            "scenarios cannot be empty."
        )

    results = []

    for name, dates in scenarios.items():

        if len(dates) != 2:
            raise ValueError(
                f"Scenario '{name}' "
                "must contain start and end dates."
            )

        result = run_historical_scenario(
            returns,
            weights,
            dates[0],
            dates[1],
            scenario_name=name,
        )

        results.append(result)

    return pd.DataFrame(
        results
    )


# ============================================================
# HISTORICAL STRESS ENGINE
# ============================================================

def historical_stress_analysis(
    returns,
    weights,
    scenarios,
):
    """
    Complete historical stress analysis.

    Returns
    -------
    dict
        Scenario results plus worst-case scenario.
    """
    comparison = (
        compare_historical_scenarios(
            returns,
            weights,
            scenarios,
        )
    )

    worst_index = (
        comparison[
            "cumulative_return"
        ].idxmin()
    )

    worst_case = (
        comparison.loc[
            worst_index
        ].to_dict()
    )

    return {
        "scenarios": comparison,
        "worst_case": worst_case,
    }


# ============================================================
# EMPIRICAL DISTRIBUTION
# ============================================================

def historical_loss_distribution(
    returns,
    weights,
):
    """
    Generate the historical distribution
    of portfolio losses.

    Positive values represent losses.
    """
    portfolio_returns = (
        calculate_historical_portfolio_returns(
            returns,
            weights,
        )
    )

    losses = (
        -portfolio_returns
    )

    losses.name = "loss"

    return losses


def historical_percentile_loss(
    returns,
    weights,
    percentile=95,
):
    """
    Calculate a historical percentile loss.

    Example:
        percentile=95 gives the 95th
        percentile historical loss.
    """
    if not 0 < percentile < 100:
        raise ValueError(
            "percentile must be between 0 and 100."
        )

    losses = historical_loss_distribution(
        returns,
        weights,
    )

    return float(
        np.percentile(
            losses.values,
            percentile,
        )
    )