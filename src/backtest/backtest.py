# src/backtest/backtest.py

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_prices(prices):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices cannot be empty")

    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices


def _validate_returns(returns):
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    returns = returns.sort_index()
    returns = returns.loc[:, ~returns.columns.duplicated()]

    return returns


def _validate_selected_assets(prices, selected_assets):
    if not selected_assets:
        raise ValueError("selected_assets cannot be empty")

    missing_assets = [
        asset
        for asset in selected_assets
        if asset not in prices.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Selected assets not found in prices: {missing_assets}"
        )

    return list(selected_assets)


# ============================================================
# RETURNS
# ============================================================

def calculate_returns(prices):
    """
    Calculate simple asset returns.
    """

    prices = _validate_prices(prices)

    returns = prices.pct_change()

    return returns.dropna(how="all")


def calculate_portfolio_returns(asset_returns, weights):
    """
    Calculate portfolio returns using fixed portfolio weights.

    Portfolio return:
        Rp = sum(weight_i * return_i)
    """

    asset_returns = _validate_returns(asset_returns)

    weights = pd.Series(weights, dtype=float)

    common_assets = (
        asset_returns.columns
        .intersection(weights.index)
    )

    if len(common_assets) == 0:
        raise ValueError(
            "No common assets between returns and weights"
        )

    aligned_returns = asset_returns[common_assets]

    aligned_weights = weights[common_assets]

    if aligned_weights.sum() == 0:
        raise ValueError(
            "Portfolio weights sum to zero"
        )

    # Normalize once so weights sum to 1.
    aligned_weights = (
        aligned_weights /
        aligned_weights.sum()
    )

    portfolio_returns = (
        aligned_returns
        .mul(aligned_weights, axis=1)
        .sum(axis=1)
    )

    return portfolio_returns


# ============================================================
# PORTFOLIO RETURN / NAV
# ============================================================

def calculate_portfolio_return(returns):
    """
    Calculate cumulative portfolio return.
    """

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return np.nan

    return (
        (1 + returns).prod() - 1
    )


def calculate_nav(
    returns,
    initial_value=1.0
):
    """
    Construct portfolio NAV.
    """

    returns = pd.Series(returns).dropna()

    return (
        initial_value *
        (1 + returns).cumprod()
    )


# ============================================================
# TURNOVER
# ============================================================

def calculate_turnover(
    old_weights,
    new_weights
):
    """
    Calculate one-way portfolio turnover.

    Turnover =
        0.5 * sum(abs(new_weight - old_weight))
    """

    old_weights = pd.Series(
        old_weights,
        dtype=float
    )

    new_weights = pd.Series(
        new_weights,
        dtype=float
    )

    assets = (
        old_weights.index
        .union(new_weights.index)
    )

    old_weights = old_weights.reindex(
        assets,
        fill_value=0.0
    )

    new_weights = new_weights.reindex(
        assets,
        fill_value=0.0
    )

    turnover = (
        0.5 *
        np.abs(
            new_weights -
            old_weights
        ).sum()
    )

    return float(turnover)


def apply_turnover_constraint(
    old_weights,
    target_weights,
    max_turnover
):
    """
    Move current portfolio toward target portfolio
    without exceeding max_turnover.

    Both old and target portfolios must sum to 1.
    """

    old_weights = pd.Series(
        old_weights,
        dtype=float
    )

    target_weights = pd.Series(
        target_weights,
        dtype=float
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
        fill_value=0.0
    )

    target_weights = target_weights.reindex(
        assets,
        fill_value=0.0
    )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    target_weights = target_weights.clip(
        lower=0.0
    )

    target_weights = (
        target_weights /
        target_weights.sum()
    )

    # --------------------------------------------------------
    # Calculate required turnover
    # --------------------------------------------------------

    turnover = calculate_turnover(
        old_weights,
        target_weights
    )

    if turnover <= max_turnover + 1e-12:
        return target_weights

    # --------------------------------------------------------
    # Interpolate between old and target portfolios
    # --------------------------------------------------------

    difference = (
        target_weights -
        old_weights
    )

    difference_abs_sum = (
        np.abs(difference).sum()
    )

    if difference_abs_sum == 0:
        return old_weights

    scale = (
        2.0 *
        max_turnover /
        difference_abs_sum
    )

    scale = min(
        1.0,
        scale
    )

    constrained_weights = (
        old_weights +
        difference * scale
    )

    constrained_weights = (
        constrained_weights.clip(
            lower=0.0
        )
    )

    # Interpolation between two fully invested
    # portfolios already sums approximately to 1.
    constrained_weights = (
        constrained_weights /
        constrained_weights.sum()
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    final_turnover = calculate_turnover(
        old_weights,
        constrained_weights
    )

    if final_turnover > max_turnover + 1e-10:
        raise ValueError(
            "Turnover constraint could not be satisfied"
        )

    return constrained_weights


# ============================================================
# EXPECTED RETURNS
# ============================================================

def historical_expected_returns(
    returns,
    annualization=252
):
    """
    Historical arithmetic expected return.
    """

    returns = _validate_returns(
        returns
    )

    return (
        returns.mean() *
        annualization
    )


# ============================================================
# COVARIANCE
# ============================================================

def covariance_matrix(
    returns,
    annualization=252
):
    """
    Annualized covariance matrix.
    """

    returns = _validate_returns(
        returns
    )

    return (
        returns.cov() *
        annualization
    )


# ============================================================
# OPTIMIZER INTERFACE
# ============================================================

def optimize_portfolio(
    expected_returns,
    covariance,
    optimizer,
    old_weights=None,
    max_turnover=None,
    **optimizer_kwargs
):
    """
    Run the selected portfolio optimizer.

    Turnover constraint is applied only when
    an existing portfolio is supplied.
    """

    target_weights = optimizer(
        expected_returns=expected_returns,
        covariance=covariance,
        **optimizer_kwargs
    )

    target_weights = pd.Series(
        target_weights,
        index=expected_returns.index,
        dtype=float
    )

    target_weights = target_weights.clip(
        lower=0.0
    )

    if target_weights.sum() == 0:
        raise ValueError(
            "Optimizer returned zero weights"
        )

    target_weights = (
        target_weights /
        target_weights.sum()
    )

    # --------------------------------------------------------
    # Apply turnover constraint on rebalancing
    # --------------------------------------------------------

    if (
        old_weights is not None
        and max_turnover is not None
    ):

        target_weights = (
            apply_turnover_constraint(
                old_weights=old_weights,
                target_weights=target_weights,
                max_turnover=max_turnover
            )
        )

    return target_weights


# ============================================================
# REBALANCE
# ============================================================

def rebalance_portfolio(
    training_returns,
    selected_assets,
    optimizer,
    old_weights=None,
    max_turnover=None,
    expected_return_method="historical",
    **optimizer_kwargs
):
    """
    Construct a portfolio using only historical
    training data.
    """

    training_returns = _validate_returns(
        training_returns
    )

    selected_assets = _validate_selected_assets(
        training_returns,
        selected_assets
    )

    training_subset = (
        training_returns[
            selected_assets
        ]
        .dropna(
            axis=1,
            how="all"
        )
    )

    if training_subset.shape[1] == 0:
        raise ValueError(
            "No assets available for optimization"
        )

    # --------------------------------------------------------
    # Expected returns
    # --------------------------------------------------------

    if expected_return_method == "historical":

        expected_returns = (
            historical_expected_returns(
                training_subset
            )
        )

    else:

        raise ValueError(
            f"Unknown expected return method: "
            f"{expected_return_method}"
        )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    covariance = covariance_matrix(
        training_subset
    )

    # --------------------------------------------------------
    # Valid assets
    # --------------------------------------------------------

    valid_assets = (
        expected_returns.index
        .intersection(
            covariance.index
        )
    )

    expected_returns = (
        expected_returns.loc[
            valid_assets
        ]
    )

    covariance = (
        covariance.loc[
            valid_assets,
            valid_assets
        ]
    )

    # --------------------------------------------------------
    # Keep complete previous portfolio
    # --------------------------------------------------------

    if old_weights is not None:

        old_weights = pd.Series(
            old_weights,
            dtype=float
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
        **optimizer_kwargs
    )

    return (
        weights,
        expected_returns,
        covariance
    )


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_backtest(
    prices,
    selected_assets,
    optimizer,
    benchmark_prices=None,
    train_window=252,
    rebalance_frequency="M",
    max_turnover=0.25,
    initial_capital=1.0,
    **optimizer_kwargs
):
    """
    Walk-forward portfolio backtest.

    IMPORTANT:
    The stock universe remains FIXED.

    The backtest does NOT:
        - remove underperformers
        - select replacements
        - change the selected stock universe

    It only:
        1. Uses historical training data
        2. Calculates expected returns
        3. Calculates covariance
        4. Optimizes the portfolio
        5. Applies turnover constraint
        6. Holds portfolio out-of-sample
        7. Rebalances periodically
        8. Records portfolio performance

    This allows different portfolio construction methods
    to be compared fairly on the same selected stocks.
    """

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    prices = _validate_prices(
        prices
    )

    selected_assets = _validate_selected_assets(
        prices,
        selected_assets
    )

    if train_window <= 0:
        raise ValueError(
            "train_window must be greater than zero"
        )

    if len(prices) <= train_window:
        raise ValueError(
            "Not enough data for train_window"
        )

    # ========================================================
    # RETURNS
    # ========================================================

    returns = calculate_returns(
        prices[
            selected_assets
        ]
    )

    if len(returns) <= train_window:
        raise ValueError(
            "Not enough return observations "
            "for train_window"
        )

    # ========================================================
    # BENCHMARK
    # ========================================================

    if benchmark_prices is not None:

        benchmark_returns = (
            calculate_returns(
                benchmark_prices
            )
        )

        if isinstance(
            benchmark_returns,
            pd.DataFrame
        ):

            if benchmark_returns.shape[1] != 1:
                raise ValueError(
                    "benchmark_prices must contain "
                    "exactly one column"
                )

            benchmark_returns = (
                benchmark_returns.iloc[:, 0]
            )

        benchmark_returns = (
            benchmark_returns.dropna()
        )

    else:

        benchmark_returns = None

    # ========================================================
    # REBALANCE DATES
    # ========================================================

    available_dates = returns.index[
        train_window:
    ]

    rebalance_dates = (
        pd.Series(
            available_dates,
            index=available_dates
        )
        .groupby(
            available_dates.to_period(
                rebalance_frequency
            )
        )
        .first()
        .tolist()
    )

    if len(rebalance_dates) == 0:
        raise ValueError(
            "No rebalance dates generated"
        )

    # ========================================================
    # STORAGE
    # ========================================================

    portfolio_returns = []

    weight_history = []

    turnover_history = []

    previous_weights = None

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):

        # ----------------------------------------------------
        # TRAINING WINDOW
        # ----------------------------------------------------

        training_end_position = (
            returns.index.get_loc(
                rebalance_date
            )
        )

        training_start_position = max(
            0,
            training_end_position -
            train_window
        )

        training_returns = (
            returns.iloc[
                training_start_position:
                training_end_position
            ]
        )

        # ----------------------------------------------------
        # OUT-OF-SAMPLE PERIOD
        # ----------------------------------------------------

        if i + 1 < len(
            rebalance_dates
        ):

            next_rebalance_date = (
                rebalance_dates[i + 1]
            )

            oos_returns = returns.loc[
                rebalance_date:
                next_rebalance_date
            ]

        else:

            oos_returns = returns.loc[
                rebalance_date:
            ]

        # Remove the rebalance date itself.
        # Portfolio weights are determined using
        # information available BEFORE this date.
        oos_returns = oos_returns.iloc[1:]

        if oos_returns.empty:
            continue

        # ----------------------------------------------------
        # OLD PORTFOLIO
        # ----------------------------------------------------

        old_weights = None

        if previous_weights is not None:

            old_weights = (
                previous_weights.copy()
            )

        # ----------------------------------------------------
        # REBALANCE
        # ----------------------------------------------------

        (
            weights,
            expected_returns,
            covariance
        ) = rebalance_portfolio(
            training_returns=training_returns,
            selected_assets=selected_assets,
            optimizer=optimizer,
            old_weights=old_weights,
            max_turnover=max_turnover,
            **optimizer_kwargs
        )

        # ----------------------------------------------------
        # TURNOVER
        # ----------------------------------------------------

        if previous_weights is None:

            # Initial portfolio construction.
            turnover = 1.0

        else:

            turnover = calculate_turnover(
                previous_weights,
                weights
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if (
            previous_weights is not None
            and max_turnover is not None
            and turnover > max_turnover + 1e-10
        ):

            raise ValueError(
                f"Turnover constraint violated on "
                f"{rebalance_date}: "
                f"{turnover:.8f} > "
                f"{max_turnover:.8f}"
            )

        turnover_history.append({
            "date": rebalance_date,
            "turnover": turnover
        })

        # ----------------------------------------------------
        # OUT-OF-SAMPLE PORTFOLIO RETURN
        # ----------------------------------------------------

        oos_assets = [
            asset
            for asset in weights.index
            if asset in oos_returns.columns
        ]

        if len(oos_assets) == 0:
            continue

        oos_weights = (
            weights.reindex(
                oos_assets,
                fill_value=0.0
            )
        )

        if oos_weights.sum() == 0:
            continue

        oos_weights = (
            oos_weights /
            oos_weights.sum()
        )

        period_returns = (
            calculate_portfolio_returns(
                oos_returns[
                    oos_assets
                ],
                oos_weights
            )
        )

        portfolio_returns.append(
            period_returns
        )

        # ----------------------------------------------------
        # WEIGHT HISTORY
        # ----------------------------------------------------

        weight_history.append(
            pd.Series(
                weights,
                name=rebalance_date
            )
        )

        # ----------------------------------------------------
        # UPDATE CURRENT PORTFOLIO
        # ----------------------------------------------------

        previous_weights = (
            weights.copy()
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(portfolio_returns) == 0:
        raise ValueError(
            "Backtest produced no portfolio returns"
        )

    # ========================================================
    # PORTFOLIO RETURNS
    # ========================================================

    portfolio_returns = pd.concat(
        portfolio_returns
    ).sort_index()

    portfolio_returns = (
        portfolio_returns[
            ~portfolio_returns.index.duplicated()
        ]
    )

    # ========================================================
    # NAV
    # ========================================================

    nav = calculate_nav(
        portfolio_returns,
        initial_capital
    )

    # ========================================================
    # WEIGHT HISTORY
    # ========================================================

    weights_df = (
        pd.DataFrame(
            weight_history
        )
        .fillna(0.0)
    )

    # ========================================================
    # TURNOVER HISTORY
    # ========================================================

    turnover_df = pd.DataFrame(
        turnover_history
    )

    # ========================================================
    # BENCHMARK NAV
    # ========================================================

    benchmark_nav = None

    if benchmark_returns is not None:

        benchmark_aligned = (
            benchmark_returns
            .reindex(
                portfolio_returns.index
            )
            .dropna()
        )

        if len(benchmark_aligned) > 0:

            benchmark_nav = calculate_nav(
                benchmark_aligned,
                initial_capital
            )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "portfolio_returns": portfolio_returns,
        "nav": nav,
        "benchmark_nav": benchmark_nav,
        "weights": weights_df,
        "turnover": turnover_df
    }


# ============================================================
# BACKTEST SUMMARY
# ============================================================

def backtest_summary(
    portfolio_returns,
    benchmark_returns=None,
    annualization=252
):
    """
    Generate portfolio performance statistics.
    """

    portfolio_returns = pd.Series(
        portfolio_returns
    ).dropna()

    if len(portfolio_returns) == 0:
        raise ValueError(
            "portfolio_returns cannot be empty"
        )

    # ========================================================
    # TOTAL RETURN
    # ========================================================

    total_return = (
        (1 + portfolio_returns).prod() - 1
    )

    # ========================================================
    # CAGR
    # ========================================================

    years = (
        len(portfolio_returns) /
        annualization
    )

    if years > 0:

        cagr = (
            (1 + total_return) **
            (1 / years)
        ) - 1

    else:

        cagr = np.nan

    # ========================================================
    # ANNUALIZED VOLATILITY
    # ========================================================

    volatility = (
        portfolio_returns.std() *
        np.sqrt(annualization)
    )

    # ========================================================
    # SHARPE
    # ========================================================

    if portfolio_returns.std() != 0:

        sharpe = (
            portfolio_returns.mean() /
            portfolio_returns.std()
        ) * np.sqrt(
            annualization
        )

    else:

        sharpe = np.nan

    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative /
        running_max
    ) - 1

    max_drawdown = drawdown.min()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown
    }

    # ========================================================
    # BENCHMARK
    # ========================================================

    if benchmark_returns is not None:

        if isinstance(
            benchmark_returns,
            pd.DataFrame
        ):

            if benchmark_returns.shape[1] != 1:
                raise ValueError(
                    "benchmark_returns must contain "
                    "exactly one column"
                )

            benchmark_returns = (
                benchmark_returns.iloc[:, 0]
            )

        benchmark_returns = pd.Series(
            benchmark_returns
        ).dropna()

        common_index = (
            portfolio_returns.index
            .intersection(
                benchmark_returns.index
            )
        )

        if len(common_index) > 0:

            portfolio_common = (
                portfolio_returns.loc[
                    common_index
                ]
            )

            benchmark_common = (
                benchmark_returns.loc[
                    common_index
                ]
            )

            portfolio_total = (
                (1 + portfolio_common).prod()
                - 1
            )

            benchmark_total = (
                (1 + benchmark_common).prod()
                - 1
            )

            summary[
                "Benchmark Return"
            ] = benchmark_total

            summary[
                "Excess Return"
            ] = (
                portfolio_total -
                benchmark_total
            )

    return pd.Series(
        summary
    )


# ============================================================
# COMPLETE BACKTEST
# ============================================================

def run_backtest(
    prices,
    selected_assets,
    optimizer,
    benchmark_prices=None,
    train_window=252,
    rebalance_frequency="M",
    max_turnover=0.25,
    initial_capital=1.0,
    **optimizer_kwargs
):
    """
    Complete portfolio backtest.

    The selected stock universe remains fixed.
    """

    results = walk_forward_backtest(
        prices=prices,
        selected_assets=selected_assets,
        optimizer=optimizer,
        benchmark_prices=benchmark_prices,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_turnover=max_turnover,
        initial_capital=initial_capital,
        **optimizer_kwargs
    )

    benchmark_returns = None

    if benchmark_prices is not None:

        benchmark_returns = (
            calculate_returns(
                benchmark_prices
            )
        )

        if isinstance(
            benchmark_returns,
            pd.DataFrame
        ):

            benchmark_returns = (
                benchmark_returns.iloc[:, 0]
            )

    results["summary"] = (
        backtest_summary(
            portfolio_returns=results[
                "portfolio_returns"
            ],
            benchmark_returns=benchmark_returns
        )
    )

    return results