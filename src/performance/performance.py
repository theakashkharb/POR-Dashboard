import numpy as np
import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_returns(returns):
    """
    Validate a return series.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be a pandas Series or DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if returns.isna().any().any():
        raise ValueError("returns contains NaN values.")

    return returns.astype(float)


# ============================================================
# TOTAL RETURN
# ============================================================
def calculate_total_return(returns):

    returns = pd.Series(
        returns,
        dtype=float
    )

    if returns.empty:

        raise ValueError(
            "returns cannot be empty."
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "returns contain non-finite values."
        )

    if (
        values <= -1
    ).any():

        raise ValueError(
            "returns cannot be less than or equal to -100%."
        )

    return float(
        (1 + returns).prod() - 1
    )

# ============================================================
# CAGR
# ============================================================

def calculate_cagr(returns, annualization=252):
    """
    Calculate Compound Annual Growth Rate.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns.
    annualization : int
        Number of periods per year.

    Returns
    -------
    float
    """
    returns = _validate_returns(returns)

    periods = len(returns)

    if periods == 0:
        return np.nan

    total_return = (1 + returns).prod()

    years = periods / annualization

    if years <= 0:
        return np.nan

    if total_return <= 0:
        return np.nan

    return total_return ** (1 / years) - 1


# ============================================================
# ANNUALIZED VOLATILITY
# ============================================================

def calculate_annualized_volatility(returns, annualization=252):
    """
    Calculate annualized volatility.
    """
    returns = _validate_returns(returns)

    return returns.std(ddof=1) * np.sqrt(annualization)


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.0,
    annualization=252
):
    """
    Calculate annualized Sharpe Ratio.

    risk_free_rate should be annualized.
    """
    returns = _validate_returns(returns)

    periodic_rf = (1 + risk_free_rate) ** (1 / annualization) - 1

    excess_returns = returns - periodic_rf

    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        return np.nan

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(annualization)
    )


# ============================================================
# DOWNSIDE DEVIATION
# ============================================================

def calculate_downside_deviation(
    returns,
    target_return=0.0,
    annualization=252
):
    """
    Calculate annualized downside deviation.

    Only returns below the target are considered.
    """
    returns = _validate_returns(returns)

    downside = np.minimum(
        returns - target_return,
        0
    )

    downside_squared = downside ** 2

    return np.sqrt(
        downside_squared.mean() * annualization
    )


# ============================================================
# SORTINO RATIO
# ============================================================

def calculate_sortino_ratio(
    returns,
    target_return=0.0,
    annualization=252
):
    """
    Calculate annualized Sortino Ratio.
    """
    returns = _validate_returns(returns)

    excess_return = (
        returns.mean() * annualization
        - target_return
    )

    downside_deviation = calculate_downside_deviation(
        returns,
        target_return=target_return / annualization,
        annualization=annualization
    )

    if downside_deviation == 0:
        return np.nan

    return excess_return / downside_deviation


# ============================================================
# NAV
# ============================================================

def calculate_nav(
    returns,
    initial_value=1.0
):
    """
    Convert periodic returns into a NAV/equity curve.
    """
    returns = _validate_returns(returns)

    return initial_value * (1 + returns).cumprod()


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_drawdown(nav):
    """
    Calculate drawdown series from NAV.
    """
    nav = _validate_returns(nav)

    running_peak = nav.cummax()

    drawdown = nav / running_peak - 1

    return drawdown


def calculate_maximum_drawdown(nav):
    """
    Calculate maximum drawdown.
    """
    drawdown = calculate_drawdown(nav)

    return drawdown.min()


# ============================================================
# DRAWDOWN DETAILS
# ============================================================

def calculate_drawdown_statistics(nav):
    """
    Calculate maximum drawdown and drawdown duration.

    Returns
    -------
    dict
    """
    nav = _validate_returns(nav)

    drawdown = calculate_drawdown(nav)

    max_drawdown = drawdown.min()

    max_drawdown_date = drawdown.idxmin()

    # Find the peak before the maximum drawdown
    peak_before_drawdown = nav.loc[:max_drawdown_date].idxmax()

    # Recovery date
    recovery_date = None

    peak_value = nav.loc[peak_before_drawdown]

    after_drawdown = nav.loc[max_drawdown_date:]

    recovered = after_drawdown[
        after_drawdown >= peak_value
    ]

    if not recovered.empty:
        recovery_date = recovered.index[0]

    # Maximum drawdown duration
    negative_drawdown = drawdown < 0

    max_duration = 0
    current_duration = 0

    for value in negative_drawdown:
        if value:
            current_duration += 1
            max_duration = max(
                max_duration,
                current_duration
            )
        else:
            current_duration = 0

    return {
        "maximum_drawdown": max_drawdown,
        "drawdown_start": peak_before_drawdown,
        "maximum_drawdown_date": max_drawdown_date,
        "recovery_date": recovery_date,
        "maximum_drawdown_duration": max_duration
    }


# ============================================================
# CALMAR RATIO
# ============================================================

def calculate_calmar_ratio(
    returns,
    annualization=252
):
    """
    Calculate Calmar Ratio.

    Calmar = CAGR / absolute Maximum Drawdown
    """
    returns = _validate_returns(returns)

    nav = calculate_nav(returns)

    cagr = calculate_cagr(
        returns,
        annualization=annualization
    )

    max_drawdown = calculate_maximum_drawdown(nav)

    if max_drawdown == 0:
        return np.nan

    return cagr / abs(max_drawdown)


# ============================================================
# BENCHMARK ANALYTICS
# ============================================================

def calculate_benchmark_metrics(
    portfolio_returns,
    benchmark_returns,
    annualization=252
):
    """
    Compare portfolio performance against benchmark.
    """
    portfolio_returns = _validate_returns(portfolio_returns)
    benchmark_returns = _validate_returns(benchmark_returns)

    combined = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark")
        ],
        axis=1
    ).dropna()

    if combined.empty:
        raise ValueError(
            "Portfolio and benchmark have no overlapping observations."
        )

    portfolio = combined["portfolio"]
    benchmark = combined["benchmark"]

    portfolio_total_return = calculate_total_return(portfolio)
    benchmark_total_return = calculate_total_return(benchmark)

    portfolio_cagr = calculate_cagr(
        portfolio,
        annualization=annualization
    )

    benchmark_cagr = calculate_cagr(
        benchmark,
        annualization=annualization
    )

    return {
        "portfolio_return": portfolio_total_return,
        "benchmark_return": benchmark_total_return,
        "excess_return": (
            portfolio_total_return
            - benchmark_total_return
        ),
        "portfolio_cagr": portfolio_cagr,
        "benchmark_cagr": benchmark_cagr,
        "excess_cagr": (
            portfolio_cagr
            - benchmark_cagr
        )
    }


# ============================================================
# ROLLING METRICS
# ============================================================

def calculate_rolling_volatility(
    returns,
    window=63,
    annualization=252
):
    """
    Calculate rolling annualized volatility.
    """
    returns = _validate_returns(returns)

    return (
        returns
        .rolling(window)
        .std()
        * np.sqrt(annualization)
    )


def calculate_rolling_sharpe(
    returns,
    window=63,
    risk_free_rate=0.0,
    annualization=252
):
    """
    Calculate rolling annualized Sharpe Ratio.
    """
    returns = _validate_returns(returns)

    periodic_rf = (
        (1 + risk_free_rate)
        ** (1 / annualization)
        - 1
    )

    excess_returns = returns - periodic_rf

    rolling_mean = excess_returns.rolling(window).mean()

    rolling_std = excess_returns.rolling(window).std()

    rolling_sharpe = (
        rolling_mean
        / rolling_std
        * np.sqrt(annualization)
    )

    return rolling_sharpe


# ============================================================
# RECOVERY PERIOD
# ============================================================

def calculate_recovery_period(nav):
    """
    Calculate the number of observations required
    to recover from the maximum drawdown.

    Returns
    -------
    int or None
    """
    stats = calculate_drawdown_statistics(nav)

    start = stats["maximum_drawdown_date"]
    recovery = stats["recovery_date"]

    if recovery is None:
        return None

    try:
        return len(
            nav.loc[start:recovery]
        ) - 1
    except Exception:
        return None


# ============================================================
# COMPLETE PERFORMANCE SUMMARY
# ============================================================

def performance_summary(
    returns,
    benchmark_returns=None,
    risk_free_rate=0.0,
    annualization=252
):
    """
    Generate complete performance analytics.

    Returns
    -------
    pd.Series
    """
    returns = _validate_returns(returns)

    nav = calculate_nav(returns)

    total_return = calculate_total_return(returns)

    cagr = calculate_cagr(
        returns,
        annualization=annualization
    )

    volatility = calculate_annualized_volatility(
        returns,
        annualization=annualization
    )

    sharpe = calculate_sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        annualization=annualization
    )

    sortino = calculate_sortino_ratio(
        returns,
        target_return=risk_free_rate,
        annualization=annualization
    )

    max_drawdown = calculate_maximum_drawdown(nav)

    calmar = calculate_calmar_ratio(
        returns,
        annualization=annualization
    )

    drawdown_stats = calculate_drawdown_statistics(nav)

    recovery_period = calculate_recovery_period(nav)

    result = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Maximum Drawdown Duration": (
            drawdown_stats["maximum_drawdown_duration"]
        ),
        "Recovery Period": recovery_period
    }

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    if benchmark_returns is not None:

        benchmark_metrics = calculate_benchmark_metrics(
            returns,
            benchmark_returns,
            annualization=annualization
        )

        result.update({
            "Benchmark Return": (
                benchmark_metrics["benchmark_return"]
            ),
            "Excess Return": (
                benchmark_metrics["excess_return"]
            ),
            "Benchmark CAGR": (
                benchmark_metrics["benchmark_cagr"]
            ),
            "Excess CAGR": (
                benchmark_metrics["excess_cagr"]
            )
        })

    return pd.Series(result)


# ============================================================
# MULTI-PORTFOLIO COMPARISON
# ============================================================

def compare_portfolios(
    portfolio_returns,
    benchmark_returns=None,
    risk_free_rate=0.0,
    annualization=252
):
    """
    Compare multiple portfolios.

    Parameters
    ----------
    portfolio_returns : dict
        Example:
        {
            "Equal Weight": series,
            "Minimum Variance": series,
            "Maximum Sharpe": series
        }

    benchmark_returns : pd.Series, optional
        Benchmark return series.

    Returns
    -------
    pd.DataFrame
    """
    results = {}

    for portfolio_name, returns in portfolio_returns.items():

        results[portfolio_name] = performance_summary(
            returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            annualization=annualization
        )

    return pd.DataFrame(results).T


# ============================================================
# PERFORMANCE RANKING
# ============================================================

def rank_portfolios(
    performance_df,
    weights=None
):
    """
    Rank portfolios across major performance metrics.

    Default ranking:
        CAGR          -> higher is better
        Sharpe        -> higher is better
        Sortino       -> higher is better
        Max Drawdown  -> closer to zero is better
        Volatility    -> lower is better
        Calmar        -> higher is better

    Parameters
    ----------
    performance_df : pd.DataFrame
        Output of compare_portfolios()

    weights : dict, optional
        Metric weights.

    Returns
    -------
    pd.DataFrame
    """

    required_columns = [
        "CAGR",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Annualized Volatility",
        "Calmar Ratio"
    ]

    missing = [
        col for col in required_columns
        if col not in performance_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing performance columns: {missing}"
        )

    if weights is None:
        weights = {
            "CAGR": 1,
            "Sharpe Ratio": 1,
            "Sortino Ratio": 1,
            "Maximum Drawdown": 1,
            "Annualized Volatility": 1,
            "Calmar Ratio": 1
        }

    ranking = pd.DataFrame(index=performance_df.index)

    # Higher is better
    ranking["CAGR Rank"] = (
        performance_df["CAGR"]
        .rank(ascending=False, method="min")
    )

    ranking["Sharpe Rank"] = (
        performance_df["Sharpe Ratio"]
        .rank(ascending=False, method="min")
    )

    ranking["Sortino Rank"] = (
        performance_df["Sortino Ratio"]
        .rank(ascending=False, method="min")
    )

    ranking["Calmar Rank"] = (
        performance_df["Calmar Ratio"]
        .rank(ascending=False, method="min")
    )

    # Drawdown:
    # -10% is better than -20%
    ranking["Drawdown Rank"] = (
        performance_df["Maximum Drawdown"]
        .rank(ascending=False, method="min")
    )

    # Lower volatility is better
    ranking["Volatility Rank"] = (
        performance_df["Annualized Volatility"]
        .rank(ascending=True, method="min")
    )

    ranking["Overall Rank"] = (
        ranking["CAGR Rank"] * weights["CAGR"]
        + ranking["Sharpe Rank"] * weights["Sharpe Ratio"]
        + ranking["Sortino Rank"] * weights["Sortino Ratio"]
        + ranking["Drawdown Rank"] * weights["Maximum Drawdown"]
        + ranking["Volatility Rank"] * weights["Annualized Volatility"]
        + ranking["Calmar Rank"] * weights["Calmar Ratio"]
    )

    ranking = ranking.sort_values(
        "Overall Rank"
    )

    return ranking


# ============================================================
# COMPLETE PERFORMANCE ENGINE
# ============================================================

def run_performance_analysis(
    portfolio_returns,
    benchmark_returns=None,
    risk_free_rate=0.0,
    annualization=252,
    ranking_weights=None
):
    """
    Run the complete performance analytics pipeline.

    Returns
    -------
    dict
        {
            "performance": performance_table,
            "ranking": ranking_table,
            "nav": nav_data,
            "drawdown": drawdown_data
        }
    """

    performance = compare_portfolios(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        annualization=annualization
    )

    ranking = rank_portfolios(
        performance,
        weights=ranking_weights
    )

    nav_data = pd.DataFrame()

    for name, returns in portfolio_returns.items():

        nav_data[name] = calculate_nav(
            returns
        )

    drawdown_data = pd.DataFrame()

    for name in portfolio_returns:

        drawdown_data[name] = calculate_drawdown(
            nav_data[name]
        )

    return {
        "performance": performance,
        "ranking": ranking,
        "nav": nav_data,
        "drawdown": drawdown_data
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PERFORMANCE ANALYTICS ENGINE TEST")
    print("=" * 70)

    np.random.seed(42)

    dates = pd.bdate_range(
        start="2022-01-01",
        periods=756
    )

    # Synthetic portfolio returns
    portfolio_returns = {}

    for name, mean, volatility in [
        ("Equal Weight", 0.0004, 0.010),
        ("Minimum Variance", 0.00042, 0.009),
        ("Maximum Sharpe", 0.00055, 0.011),
        ("Risk Parity", 0.00041, 0.0095),
        ("HRP", 0.00040, 0.0098),
        ("Black-Litterman", 0.00043, 0.010)
    ]:

        portfolio_returns[name] = pd.Series(
            np.random.normal(
                mean,
                volatility,
                len(dates)
            ),
            index=dates
        )

    benchmark_returns = pd.Series(
        np.random.normal(
            0.00045,
            0.011,
            len(dates)
        ),
        index=dates
    )

    # --------------------------------------------------------
    # Individual performance test
    # --------------------------------------------------------

    summary = performance_summary(
        portfolio_returns["Equal Weight"],
        benchmark_returns=benchmark_returns
    )

    print("\nINDIVIDUAL PERFORMANCE:")
    print(summary)

    # --------------------------------------------------------
    # Portfolio comparison
    # --------------------------------------------------------

    comparison = compare_portfolios(
        portfolio_returns,
        benchmark_returns=benchmark_returns
    )

    print("\nPORTFOLIO COMPARISON:")
    print(comparison.round(4))

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranking = rank_portfolios(
        comparison
    )

    print("\nPORTFOLIO RANKING:")
    print(ranking)

    # --------------------------------------------------------
    # Complete engine
    # --------------------------------------------------------

    results = run_performance_analysis(
        portfolio_returns,
        benchmark_returns=benchmark_returns
    )

    print("\nNAV SHAPE:")
    print(results["nav"].shape)

    print("\nDRAWDOWN SHAPE:")
    print(results["drawdown"].shape)

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    # Total return
    test_returns = pd.Series(
        [0.10, -0.05, 0.10]
    )

    expected_total_return = (
        (1.10 * 0.95 * 1.10) - 1
    )

    calculated_total_return = calculate_total_return(
        test_returns
    )

    assert np.isclose(
        calculated_total_return,
        expected_total_return
    )

    # NAV
    nav = calculate_nav(test_returns)

    assert np.isclose(
        nav.iloc[-1],
        1 + expected_total_return
    )

    # Drawdown
    drawdown = calculate_drawdown(nav)

    assert drawdown.iloc[0] == 0

    # Ranking
    assert not ranking.empty

    # Weight-independent portfolio count
    assert len(results["performance"]) == 6

    print("\n" + "=" * 70)
    print("🟢 ALL PERFORMANCE ENGINE TESTS PASSED")
    print("=" * 70)