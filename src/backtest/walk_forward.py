from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.validation import (
    validate_prices,
    validate_selected_assets,
    validate_train_window,
    validate_rebalance_frequency,
    validate_max_turnover,
    validate_initial_capital,
    validate_benchmark_prices,
)

from src.backtest.portfolio_returns import (
    calculate_returns,
    calculate_portfolio_returns,
    calculate_nav,
)

from src.backtest.turnover import (
    calculate_turnover,
)

from src.backtest.rebalance import (
    rebalance_portfolio,
)


def walk_forward_backtest(
    prices,
    selected_assets,
    optimizer,
    benchmark_prices=None,
    train_window=252,
    rebalance_frequency="M",
    max_turnover=0.25,
    initial_capital=1.0,
    **optimizer_kwargs,
):
    """
    Run a walk-forward portfolio backtest.

    At each rebalance date:

    1. Use only historical data before the rebalance date.
    2. Estimate portfolio parameters.
    3. Optimize portfolio weights.
    4. Apply turnover constraint.
    5. Trade only in the following out-of-sample period.

    Returns
    -------
    dict
        Portfolio returns, NAV, weights, turnover,
        and benchmark NAV.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    prices = validate_prices(prices)

    selected_assets = validate_selected_assets(
        prices,
        selected_assets,
    )

    train_window = validate_train_window(
        train_window,
        minimum_observations=len(prices),
    )

    rebalance_frequency = validate_rebalance_frequency(
        rebalance_frequency
    )

    max_turnover = validate_max_turnover(
        max_turnover
    )

    initial_capital = validate_initial_capital(
        initial_capital
    )

    benchmark_prices = validate_benchmark_prices(
        benchmark_prices
    )

    if not callable(optimizer):
        raise TypeError(
            "optimizer must be callable"
        )

    # ========================================================
    # PRICE / RETURN DATA
    # ========================================================

    prices = prices[
        selected_assets
    ].copy()

    prices = prices.sort_index()

    asset_returns = calculate_returns(
        prices
    )

    if asset_returns.empty:
        raise ValueError(
            "No valid return observations available"
        )

    # ========================================================
    # BENCHMARK
    # ========================================================

    benchmark_returns = None

    if benchmark_prices is not None:
        benchmark_returns = (
            benchmark_prices
            .pct_change()
            .dropna()
        )

    # ========================================================
    # REBALANCE DATES
    # ========================================================

    available_dates = asset_returns.index[
        train_window:
    ]

    if len(available_dates) == 0:
        raise ValueError(
            "Not enough observations for walk-forward backtest"
        )

    rebalance_dates = (
        pd.Series(
            available_dates,
            index=available_dates,
        )
        .groupby(
            available_dates.to_period(
                rebalance_frequency
            )
        )
        .first()
        .tolist()
    )

    if not rebalance_dates:
        raise ValueError(
            "No rebalance dates generated"
        )

    # ========================================================
    # STORAGE
    # ========================================================

    portfolio_returns_list = []
    weight_history = []
    turnover_history = []

    previous_weights = None

    # ========================================================
    # WALK FORWARD LOOP
    # ========================================================

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):

        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        training_returns = asset_returns.loc[
            asset_returns.index < rebalance_date
        ]

        if len(training_returns) < train_window:
            continue

        training_returns = (
            training_returns
            .iloc[-train_window:]
        )

        # ----------------------------------------------------
        # Rebalance
        # ----------------------------------------------------

        weights, expected_returns, covariance = (
            rebalance_portfolio(
                training_returns=training_returns,
                selected_assets=selected_assets,
                optimizer=optimizer,
                old_weights=previous_weights,
                max_turnover=max_turnover
                if previous_weights is not None
                else None,
                **optimizer_kwargs,
            )
        )

        # ----------------------------------------------------
        # Turnover
        # ----------------------------------------------------

        if previous_weights is None:
            turnover = 1.0
        else:
            turnover = calculate_turnover(
                previous_weights,
                weights,
            )

            if turnover > max_turnover + 1e-10:
                raise ValueError(
                    "Turnover constraint violated"
                )

        # ----------------------------------------------------
        # Store weights
        # ----------------------------------------------------

        weights.name = rebalance_date

        weight_history.append(
            weights
        )

        turnover_history.append(
            {
                "date": rebalance_date,
                "turnover": turnover,
            }
        )

        # ----------------------------------------------------
        # Determine out-of-sample period
        # ----------------------------------------------------

        if i + 1 < len(rebalance_dates):
            next_rebalance_date = (
                rebalance_dates[i + 1]
            )

            oos_returns = asset_returns.loc[
                (asset_returns.index >= rebalance_date)
                & (
                    asset_returns.index
                    < next_rebalance_date
                )
            ]
        else:
            oos_returns = asset_returns.loc[
                asset_returns.index >= rebalance_date
            ]

        # ----------------------------------------------------
        # Avoid using rebalance-date return
        # ----------------------------------------------------

        if not oos_returns.empty:
            oos_returns = oos_returns.iloc[1:]

        if oos_returns.empty:
            previous_weights = weights
            continue

        # ----------------------------------------------------
        # Portfolio returns
        # ----------------------------------------------------

        period_returns = (
            calculate_portfolio_returns(
                oos_returns,
                weights,
            )
        )

        portfolio_returns_list.append(
            period_returns
        )

        previous_weights = weights

    # ========================================================
    # VALIDATE RESULTS
    # ========================================================

    if not portfolio_returns_list:
        raise ValueError(
            "Backtest produced no portfolio returns"
        )

    # ========================================================
    # COMBINE PORTFOLIO RETURNS
    # ========================================================

    portfolio_returns = pd.concat(
        portfolio_returns_list
    )

    portfolio_returns = (
        portfolio_returns[
            ~portfolio_returns.index.duplicated(
                keep="first"
            )
        ]
        .sort_index()
    )

    portfolio_returns.name = (
        "portfolio_return"
    )

    # ========================================================
    # NAV
    # ========================================================

    nav = calculate_nav(
        portfolio_returns,
        initial_value=initial_capital,
    )

    nav.name = "portfolio_nav"

    # ========================================================
    # WEIGHT HISTORY
    # ========================================================

    weights_df = pd.DataFrame(
        weight_history
    )

    if not weights_df.empty:
        weights_df = (
            weights_df
            .fillna(0.0)
            .sort_index()
        )

    # ========================================================
    # TURNOVER HISTORY
    # ========================================================

    turnover_df = pd.DataFrame(
        turnover_history
    )

    if not turnover_df.empty:
        turnover_df = (
            turnover_df
            .set_index("date")
            .sort_index()
        )

    # ========================================================
    # BENCHMARK NAV
    # ========================================================

    benchmark_nav = None

    if benchmark_returns is not None:

        benchmark_returns = (
            benchmark_returns
            .reindex(portfolio_returns.index)
            .dropna()
        )

        benchmark_nav = calculate_nav(
            benchmark_returns,
            initial_value=initial_capital,
        )

        benchmark_nav.name = (
            "benchmark_nav"
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_nav": nav,
        "weights": weights_df,
        "turnover": turnover_df,
        "benchmark_nav": benchmark_nav,
        "rebalance_dates": rebalance_dates,
    }