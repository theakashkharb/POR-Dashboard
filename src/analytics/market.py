from __future__ import annotations

import pandas as pd

from src.analytics.sector import (
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    sharpe_ratio,
)


# ============================================================
# SECTOR RETURNS
# ============================================================

def calculate_sector_returns(
    market_data: pd.DataFrame,
    universe_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build equal-weighted daily return series for every sector.

    Uses one global price matrix and one pct_change() call,
    preserving the exact Pandas semantics of the previous
    implementation while avoiding repeated per-sector pivots.

    Returns:
        DataFrame
        index   = Date
        columns = sectors
    """

    if market_data.empty:
        return pd.DataFrame()

    if universe_data.empty:
        return pd.DataFrame()

    required_market = {
        "Ticker",
        "Date",
        "Close",
    }

    required_universe = {
        "yf_ticker",
        "sector",
    }

    if not required_market.issubset(
        market_data.columns
    ):
        raise ValueError(
            f"Market data must contain: {required_market}"
        )

    if not required_universe.issubset(
        universe_data.columns
    ):
        raise ValueError(
            f"Universe data must contain: {required_universe}"
        )

    # --------------------------------------------------------
    # Merge market data with sector metadata
    # --------------------------------------------------------

    merged = market_data.merge(
        universe_data[
            [
                "yf_ticker",
                "sector",
            ]
        ],
        left_on="Ticker",
        right_on="yf_ticker",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # One global price matrix
    # --------------------------------------------------------

    prices = (
        merged[
            [
                "Date",
                "Ticker",
                "Close",
            ]
        ]
        .pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )
        .sort_index()
    )

    if prices.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Pandas 3.0 reference behavior is:
    # pct_change(fill_method=None)
    #
    # This deliberately does NOT forward-fill missing prices.
    # --------------------------------------------------------

    returns = prices.pct_change(
        fill_method=None
    )

    if returns.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Build ticker → sector mapping
    #
    # The universe is expected to contain one sector per
    # Yahoo ticker.
    # --------------------------------------------------------

    ticker_sector = (
        universe_data[
            [
                "yf_ticker",
                "sector",
            ]
        ]
        .dropna(
            subset=[
                "yf_ticker",
                "sector",
            ]
        )
        .drop_duplicates(
            subset="yf_ticker",
        )
        .set_index("yf_ticker")["sector"]
    )

    # --------------------------------------------------------
    # Align mapping with price/return columns
    # --------------------------------------------------------

    sector_by_ticker = ticker_sector.reindex(
        returns.columns
    )

    valid_columns = (
        sector_by_ticker.notna()
    )

    if not valid_columns.any():
        return pd.DataFrame()

    returns = returns.loc[
        :,
        valid_columns.values,
    ]

    sector_by_ticker = (
        sector_by_ticker.loc[
            returns.columns
        ]
    )

    # --------------------------------------------------------
    # Equal-weight sector aggregation
    #
    # Transpose so sectors become the grouping key.
    # This reproduces:
    #
    # for each sector:
    #     sector_prices
    #     sector_prices.pct_change(fill_method=None)
    #     mean(axis=1)
    #
    # without repeating the operation 20 times.
    # --------------------------------------------------------

    sector_returns = (
        returns
        .T
        .groupby(
            sector_by_ticker,
            sort=True,
        )
        .mean()
        .T
    )

    sector_returns.columns.name = None

    return sector_returns.dropna(
        how="all"
    )


# ============================================================
# MARKET INDEX
# ============================================================

def calculate_market_index(
    sector_returns: pd.DataFrame,
) -> pd.Series:
    """
    Create an equal-weighted market index from sector returns.
    """

    if sector_returns.empty:
        return pd.Series(
            dtype=float,
            name="Market",
        )

    market_returns = sector_returns.mean(
        axis=1
    )

    market_returns.name = "Market"

    return market_returns.dropna()


# ============================================================
# CUMULATIVE RETURNS
# ============================================================

def cumulative_returns(
    returns: pd.DataFrame | pd.Series,
):
    """
    Convert periodic returns into cumulative growth.
    """

    return (
        (1 + returns)
        .cumprod()
        - 1
    )


# ============================================================
# SECTOR PERFORMANCE
# ============================================================

def sector_performance_table(
    sector_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate performance statistics for every sector.
    """

    if sector_returns.empty:
        return pd.DataFrame()

    rows = []

    for sector in sector_returns.columns:

        returns = (
            sector_returns[sector]
            .dropna()
        )

        if returns.empty:
            continue

        rows.append(
            {
                "sector": sector,
                "total_return": (
                    (1 + returns).prod()
                    - 1
                ),
                "annualized_return": (
                    annualized_return(
                        returns
                    )
                ),
                "annualized_volatility": (
                    annualized_volatility(
                        returns
                    )
                ),
                "sharpe_ratio": (
                    sharpe_ratio(
                        returns,
                        risk_free_rate=(
                            risk_free_rate
                        ),
                    )
                ),
                "maximum_drawdown": (
                    maximum_drawdown(
                        returns
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "annualized_return",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# SECTOR RANKING
# ============================================================

def sector_ranking(
    sector_returns: pd.DataFrame,
    metric: str = "annualized_return",
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Rank sectors using a selected performance metric.
    """

    metrics = sector_performance_table(
        sector_returns,
        risk_free_rate=risk_free_rate,
    )

    if metrics.empty:
        return metrics

    if metric not in metrics.columns:
        raise ValueError(
            f"Unknown ranking metric: {metric}"
        )

    return (
        metrics
        .sort_values(
            metric,
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# CORRELATION
# ============================================================

def sector_correlation(
    sector_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlation between sector returns.
    """

    if sector_returns.empty:
        return pd.DataFrame()

    return sector_returns.corr()


# ============================================================
# RISK / RETURN
# ============================================================

def sector_risk_return(
    sector_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Prepare risk-return data for visualization.
    """

    metrics = sector_performance_table(
        sector_returns,
        risk_free_rate=risk_free_rate,
    )

    if metrics.empty:
        return metrics

    return metrics[
        [
            "sector",
            "annualized_volatility",
            "annualized_return",
            "sharpe_ratio",
        ]
    ].copy()


# ============================================================
# COMPLETE MARKET ANALYSIS
# ============================================================

def get_market_analysis(
    market_data: pd.DataFrame,
    universe_data: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Complete market-level analytics package.

    Returns:
        {
            "sector_returns": ...,
            "market_returns": ...,
            "performance": ...,
            "ranking": ...,
            "correlation": ...,
            "risk_return": ...
        }
    """

    sector_returns = calculate_sector_returns(
        market_data,
        universe_data,
    )

    if sector_returns.empty:

        return {
            "sector_returns": pd.DataFrame(),
            "market_returns": pd.Series(
                dtype=float
            ),
            "performance": pd.DataFrame(),
            "ranking": pd.DataFrame(),
            "correlation": pd.DataFrame(),
            "risk_return": pd.DataFrame(),
        }

    market_returns = calculate_market_index(
        sector_returns
    )

    performance = sector_performance_table(
        sector_returns,
        risk_free_rate=risk_free_rate,
    )

    ranking = sector_ranking(
        sector_returns,
        metric="annualized_return",
        risk_free_rate=risk_free_rate,
    )

    correlation = sector_correlation(
        sector_returns
    )

    risk_return = sector_risk_return(
        sector_returns,
        risk_free_rate=risk_free_rate,
    )

    return {
        "sector_returns": sector_returns,
        "market_returns": market_returns,
        "performance": performance,
        "ranking": ranking,
        "correlation": correlation,
        "risk_return": risk_return,
    }