from __future__ import annotations

import pandas as pd

from src.backtest.walk_forward import (
    walk_forward_backtest,
)

from src.backtest.results import (
    summarize_backtest,
    compare_backtests,
)


SUPPORTED_BACKTEST_ENGINES = [
    "walk_forward",
]


def list_backtest_engines():
    """
    Return available backtest engines.
    """
    return SUPPORTED_BACKTEST_ENGINES.copy()


def validate_backtest_engine(engine):
    """
    Validate the requested backtest engine.
    """
    if engine not in SUPPORTED_BACKTEST_ENGINES:
        raise ValueError(
            f"Unsupported backtest engine: {engine}. "
            f"Supported engines: {SUPPORTED_BACKTEST_ENGINES}"
        )

    return engine


def run_walk_forward_backtest(
    prices,
    selected_assets,
    optimizer,
    benchmark_prices=None,
    train_window=252,
    rebalance_frequency="M",
    max_turnover=0.25,
    initial_capital=1.0,
    risk_free_rate=0.0,
    annualization=252,
    **optimizer_kwargs,
):
    """
    Run walk-forward backtest and generate
    performance summary.
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
        **optimizer_kwargs,
    )

    benchmark_returns = None

    if benchmark_prices is not None:

        if isinstance(
            benchmark_prices,
            pd.DataFrame,
        ):
            benchmark_prices = (
                benchmark_prices.iloc[:, 0]
            )

        benchmark_returns = (
            benchmark_prices
            .pct_change()
            .dropna()
            .reindex(
                results["portfolio_returns"].index
            )
            .dropna()
        )

    performance = summarize_backtest(
        returns=results["portfolio_returns"],
        nav=results["portfolio_nav"],
        benchmark_returns=benchmark_returns,
        turnover=results["turnover"]["turnover"]
        if not results["turnover"].empty
        else None,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
    )

    results["performance"] = performance

    return results


def run_backtest(
    prices,
    selected_assets,
    optimizer,
    engine="walk_forward",
    benchmark_prices=None,
    train_window=252,
    rebalance_frequency="M",
    max_turnover=0.25,
    initial_capital=1.0,
    risk_free_rate=0.0,
    annualization=252,
    **optimizer_kwargs,
):
    """
    Main backtest entry point.
    """

    validate_backtest_engine(
        engine
    )

    if engine == "walk_forward":

        return run_walk_forward_backtest(
            prices=prices,
            selected_assets=selected_assets,
            optimizer=optimizer,
            benchmark_prices=benchmark_prices,
            train_window=train_window,
            rebalance_frequency=rebalance_frequency,
            max_turnover=max_turnover,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            annualization=annualization,
            **optimizer_kwargs,
        )

    raise ValueError(
        f"Backtest engine not implemented: {engine}"
    )


def compare_strategies(
    strategy_results,
    risk_free_rate=0.0,
    annualization=252,
):
    """
    Compare multiple backtest strategies.

    Parameters
    ----------
    strategy_results : dict
        Mapping:
            strategy_name -> backtest result dict
    """

    if not strategy_results:
        raise ValueError(
            "strategy_results cannot be empty"
        )

    returns = {}

    for name, result in strategy_results.items():

        if (
            not isinstance(result, dict)
            or "portfolio_returns" not in result
        ):
            raise ValueError(
                f"Invalid backtest result for strategy: {name}"
            )

        returns[name] = result[
            "portfolio_returns"
        ]

    return compare_backtests(
        returns,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
    )


def backtest_summary(
    result,
):
    """
    Extract the main summary from a backtest result.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "result must be a dictionary"
        )

    required = [
        "portfolio_returns",
        "portfolio_nav",
        "weights",
        "turnover",
        "performance",
    ]

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise ValueError(
            f"Backtest result missing keys: {missing}"
        )

    return {
        "performance": result["performance"],
        "portfolio_returns": result[
            "portfolio_returns"
        ],
        "portfolio_nav": result[
            "portfolio_nav"
        ],
        "weights": result["weights"],
        "turnover": result["turnover"],
    }