from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    if returns.empty:
        raise ValueError(
            "returns cannot be empty"
        )

    if not np.isfinite(
        returns.values
    ).all():
        raise ValueError(
            "returns contain non-finite values"
        )

    return returns


def _validate_nav(nav):
    nav = pd.Series(
        nav,
        dtype=float,
    ).dropna()

    if nav.empty:
        raise ValueError(
            "nav cannot be empty"
        )

    if not np.isfinite(
        nav.values
    ).all():
        raise ValueError(
            "nav contains non-finite values"
        )

    return nav


# ============================================================
# RETURN METRICS
# ============================================================

def total_return(returns):
    """
    Calculate cumulative portfolio return.
    """
    returns = _validate_returns(
        returns
    )

    return float(
        (1.0 + returns).prod() - 1.0
    )


def annualized_return(
    returns,
    annualization=252,
):
    """
    Calculate annualized geometric return.
    """
    returns = _validate_returns(
        returns
    )

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    periods = len(returns)

    if periods == 0:
        return np.nan

    cumulative = (
        1.0 + total_return(returns)
    )

    return float(
        cumulative
        ** (annualization / periods)
        - 1.0
    )


def annualized_volatility(
    returns,
    annualization=252,
):
    """
    Calculate annualized volatility.
    """
    returns = _validate_returns(
        returns
    )

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    return float(
        returns.std()
        * np.sqrt(annualization)
    )


# ============================================================
# RISK-ADJUSTED METRICS
# ============================================================

def sharpe_ratio(
    returns,
    risk_free_rate=0.0,
    annualization=252,
):
    """
    Calculate annualized Sharpe ratio.
    """
    returns = _validate_returns(
        returns
    )

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    excess_returns = (
        returns
        - risk_free_rate / annualization
    )

    volatility = excess_returns.std()

    if volatility == 0:
        return np.nan

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(annualization)
    )


def sortino_ratio(
    returns,
    risk_free_rate=0.0,
    annualization=252,
):
    """
    Calculate annualized Sortino ratio.
    """
    returns = _validate_returns(
        returns
    )

    if annualization <= 0:
        raise ValueError(
            "annualization must be greater than zero"
        )

    excess_returns = (
        returns
        - risk_free_rate / annualization
    )

    downside = np.minimum(
        excess_returns,
        0.0,
    )

    downside_deviation = np.sqrt(
        np.mean(
            downside ** 2
        )
    )

    if downside_deviation == 0:
        return np.nan

    return float(
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(annualization)
    )


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_drawdown(nav):
    """
    Calculate drawdown series from NAV.
    """
    nav = _validate_nav(
        nav
    )

    high_water_mark = (
        nav.cummax()
    )

    drawdown = (
        nav / high_water_mark
        - 1.0
    )

    drawdown.name = "drawdown"

    return drawdown


def maximum_drawdown(nav):
    """
    Calculate maximum drawdown.
    """
    drawdown = calculate_drawdown(
        nav
    )

    return float(
        drawdown.min()
    )


# ============================================================
# CALMAR
# ============================================================

def calmar_ratio(
    returns,
    annualization=252,
):
    """
    Calculate Calmar ratio.
    """
    returns = _validate_returns(
        returns
    )

    nav = (
        1.0 + returns
    ).cumprod()

    mdd = abs(
        maximum_drawdown(nav)
    )

    if mdd == 0:
        return np.nan

    return float(
        annualized_return(
            returns,
            annualization,
        )
        / mdd
    )


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

def benchmark_comparison(
    portfolio_returns,
    benchmark_returns,
    annualization=252,
):
    """
    Compare portfolio performance against benchmark.
    """
    portfolio_returns = _validate_returns(
        portfolio_returns
    )

    benchmark_returns = _validate_returns(
        benchmark_returns
    )

    aligned = pd.concat(
        [
            portfolio_returns.rename(
                "portfolio"
            ),
            benchmark_returns.rename(
                "benchmark"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping observations "
            "between portfolio and benchmark"
        )

    active_returns = (
        aligned["portfolio"]
        - aligned["benchmark"]
    )

    tracking_error = (
        active_returns.std()
        * np.sqrt(annualization)
    )

    if tracking_error == 0:
        information_ratio = np.nan
    else:
        information_ratio = (
            active_returns.mean()
            / active_returns.std()
            * np.sqrt(annualization)
        )

    return {
        "portfolio_return": total_return(
            aligned["portfolio"]
        ),
        "benchmark_return": total_return(
            aligned["benchmark"]
        ),
        "active_return": total_return(
            aligned["portfolio"]
        )
        - total_return(
            aligned["benchmark"]
        ),
        "tracking_error": float(
            tracking_error
        ),
        "information_ratio": float(
            information_ratio
        )
        if np.isfinite(information_ratio)
        else np.nan,
    }


# ============================================================
# RESULT SUMMARY
# ============================================================

def summarize_backtest(
    returns,
    nav=None,
    benchmark_returns=None,
    turnover=None,
    risk_free_rate=0.0,
    annualization=252,
):
    """
    Create a complete backtest performance summary.
    """
    returns = _validate_returns(
        returns
    )

    if nav is None:
        nav = (
            1.0 + returns
        ).cumprod()
    else:
        nav = _validate_nav(
            nav
        )

    summary = {
        "observations": int(
            len(returns)
        ),
        "total_return": total_return(
            returns
        ),
        "annualized_return": annualized_return(
            returns,
            annualization,
        ),
        "annualized_volatility": annualized_volatility(
            returns,
            annualization,
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate,
            annualization,
        ),
        "sortino_ratio": sortino_ratio(
            returns,
            risk_free_rate,
            annualization,
        ),
        "maximum_drawdown": maximum_drawdown(
            nav
        ),
        "calmar_ratio": calmar_ratio(
            returns,
            annualization,
        ),
    }

    if turnover is not None:
        turnover = pd.Series(
            turnover,
            dtype=float,
        ).dropna()

        if not turnover.empty:
            summary[
                "average_turnover"
            ] = float(
                turnover.mean()
            )

            summary[
                "maximum_turnover"
            ] = float(
                turnover.max()
            )

    if benchmark_returns is not None:
        comparison = benchmark_comparison(
            returns,
            benchmark_returns,
            annualization,
        )

        summary.update(
            comparison
        )

    return pd.Series(
        summary,
        dtype=float,
    )


def compare_backtests(
    results,
    risk_free_rate=0.0,
    annualization=252,
):
    """
    Compare multiple backtest return series.

    Parameters
    ----------
    results : dict
        Mapping of strategy name -> returns Series.
    """
    if not results:
        raise ValueError(
            "results cannot be empty"
        )

    rows = []

    for name, returns in results.items():

        returns = _validate_returns(
            returns
        )

        nav = (
            1.0 + returns
        ).cumprod()

        rows.append(
            {
                "strategy": name,
                "total_return": total_return(
                    returns
                ),
                "annualized_return": annualized_return(
                    returns,
                    annualization,
                ),
                "annualized_volatility": annualized_volatility(
                    returns,
                    annualization,
                ),
                "sharpe_ratio": sharpe_ratio(
                    returns,
                    risk_free_rate,
                    annualization,
                ),
                "maximum_drawdown": maximum_drawdown(
                    nav
                ),
                "calmar_ratio": calmar_ratio(
                    returns,
                    annualization,
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).set_index("strategy")