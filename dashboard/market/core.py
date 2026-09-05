from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS = 252

PERIOD_OPTIONS = {
    "1 Week": ("days", 7),
    "2 Weeks": ("days", 14),
    "3 Weeks": ("days", 21),
    "10 Weeks": ("days", 70),

    "1 Month": ("months", 1),
    "2 Months": ("months", 2),
    "3 Months": ("months", 3),
    "6 Months": ("months", 6),
    "9 Months": ("months", 9),

    "1 Year": ("months", 12),
    "2 Years": ("months", 24),
    "3 Years": ("months", 36),
    "5 Years": ("months", 60),
    "10 Years": ("months", 120),

    "All Time": None,
}


# ============================================================
# SHARED HELPERS
# ============================================================

def _safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def _format_percent(value):
    value = _safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.2%}"


def _format_sharpe(value):
    value = _safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.2f}"


# ============================================================
# DATE HANDLING
# ============================================================

def _normalize_dates(
    data: pd.DataFrame,
) -> pd.DataFrame:

    if data.empty:
        return data.copy()

    data = data.copy()

    data["Date"] = (
        pd.to_datetime(data["Date"])
        .dt.tz_localize(None)
    )

    return data


def _analysis_window(
    data: pd.DataFrame,
    start_date,
    end_date,
    period,
) -> pd.DataFrame:

    if data.empty:
        return data.copy()

    data = _normalize_dates(data)

    end_ts = pd.Timestamp(end_date)

    # --------------------------------------------------------
    # ALL TIME
    # --------------------------------------------------------

    if period is None:

        start_ts = pd.Timestamp(start_date)

    # --------------------------------------------------------
    # NEW PERIOD FORMAT
    # ("days", 7)
    # ("months", 1)
    # --------------------------------------------------------

    elif isinstance(period, tuple):

        unit, value = period

        if unit == "days":

            start_ts = (
                end_ts
                - pd.Timedelta(days=value)
            )

        elif unit == "months":

            start_ts = (
                end_ts
                - pd.DateOffset(months=value)
            )

        else:

            raise ValueError(
                f"Unsupported period unit: {unit}"
            )

    # --------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # --------------------------------------------------------

    else:

        start_ts = (
            end_ts
            - pd.DateOffset(months=int(period))
        )

    result = data[
        (data["Date"] >= start_ts)
        & (data["Date"] <= end_ts)
    ].copy()

    return result


# ============================================================
# PRICE / RETURN FUNCTIONS
# ============================================================

def _get_price_matrix(
    data: pd.DataFrame,
) -> pd.DataFrame:

    if data.empty:
        return pd.DataFrame()

    required = {
        "Date",
        "Ticker",
        "Close",
    }

    if not required.issubset(
        data.columns
    ):
        return pd.DataFrame()

    prices = (
        data[
            [
                "Date",
                "Ticker",
                "Close",
            ]
        ]
        .dropna(
            subset=[
                "Date",
                "Ticker",
                "Close",
            ]
        )
        .drop_duplicates(
            [
                "Date",
                "Ticker",
            ]
        )
        .pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )
        .sort_index()
    )

    return prices


def _get_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:

    prices = _get_price_matrix(data)

    if prices.empty:
        return pd.DataFrame()

    return prices.pct_change(
        fill_method=None
    )


def _annualized_return(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    total_growth = (
        1.0 + returns
    ).prod()

    if total_growth <= 0:
        return np.nan

    years = (
        len(returns)
        / TRADING_DAYS
    )

    if years <= 0:
        return np.nan

    return (
        total_growth
        ** (1.0 / years)
        - 1.0
    )


def _total_return(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    return (
        1.0 + returns
    ).prod() - 1.0


def _annualized_volatility(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )


def _sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    daily_rf = (
        (1.0 + risk_free_rate)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    excess = (
        returns - daily_rf
    )

    std = excess.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return (
        excess.mean()
        / std
        * np.sqrt(TRADING_DAYS)
    )


def _maximum_drawdown(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    wealth = (
        1.0 + returns
    ).cumprod()

    peak = wealth.cummax()

    drawdown = (
        wealth / peak
        - 1.0
    )

    return drawdown.min()


# ============================================================
# SECTOR RETURNS
# ============================================================

def _calculate_sector_returns(
    data: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:

    if (
        data.empty
        or universe.empty
    ):
        return pd.DataFrame()

    prices = _get_price_matrix(data)

    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change(
        fill_method=None
    )

    mapping = (
        universe[
            [
                "yf_ticker",
                "sector",
            ]
        ]
        .dropna()
        .drop_duplicates(
            "yf_ticker"
        )
        .set_index(
            "yf_ticker"
        )["sector"]
    )

    mapping = mapping.reindex(
        returns.columns
    )

    valid = mapping.notna()

    returns = returns.loc[
        :,
        valid.values,
    ]

    mapping = mapping.loc[
        returns.columns
    ]

    if returns.empty:
        return pd.DataFrame()

    sector_returns = (
        returns.T
        .groupby(
            mapping,
            sort=True,
        )
        .mean()
        .T
    )

    return sector_returns


# ============================================================
# MARKET METRICS
# ============================================================

def _market_metrics(
    sector_returns: pd.DataFrame,
) -> dict:

    if sector_returns.empty:

        return {
            "market_returns": pd.Series(
                dtype=float
            ),
            "total_return": np.nan,
            "avg_sector_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "regime": "N/A",
        }

    market_returns = (
        sector_returns
        .mean(
            axis=1,
            skipna=True,
        )
        .dropna()
    )

    if market_returns.empty:

        return {
            "market_returns": pd.Series(
                dtype=float
            ),
            "total_return": np.nan,
            "avg_sector_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "regime": "N/A",
        }

    total_return = _total_return(
        market_returns
    )

    annualized_sector_returns = [
        _annualized_return(
            sector_returns[col]
        )
        for col in sector_returns.columns
    ]

    annualized_sector_returns = (
        pd.Series(
            annualized_sector_returns
        )
        .dropna()
    )

    avg_sector_return = (
        annualized_sector_returns.mean()
        if not annualized_sector_returns.empty
        else np.nan
    )

    volatility = _annualized_volatility(
        market_returns
    )

    sharpe = _sharpe_ratio(
        market_returns
    )

    max_drawdown = _maximum_drawdown(
        market_returns
    )

    recent_window = market_returns.tail(
        min(63, len(market_returns))
    )

    recent_return = _total_return(
        recent_window
    )

    if np.isnan(recent_return):

        regime = "Mixed"

    elif recent_return > 0.02:

        regime = "Risk-On"

    elif recent_return < -0.02:

        regime = "Risk-Off"

    else:

        regime = "Mixed"

    return {
        # IMPORTANT:
        # Snapshot needs the actual market return series.
        "market_returns": market_returns,

        "total_return": total_return,
        "avg_sector_return": avg_sector_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "regime": regime,
    }


# ============================================================
# TABLE STYLING
# ============================================================

def _pastel_sector_style(
    dataframe: pd.DataFrame,
):

    styled = (
        dataframe.style
        .format(
            {
                "Total Return": "{:.2%}",
                "Annualized Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
            }
        )
    )

    styled = (
        styled
        .background_gradient(
            cmap="YlGn",
            subset=[
                "Total Return",
                "Annualized Return",
                "Sharpe",
            ],
            vmin=-1,
            vmax=1,
        )
        .background_gradient(
            cmap="OrRd",
            subset=[
                "Volatility"
            ],
        )
        .background_gradient(
            cmap="OrRd",
            subset=[
                "Max Drawdown"
            ],
        )
    )

    return styled


def _pastel_stock_style(
    dataframe: pd.DataFrame,
):

    styled = (
        dataframe.style
        .format(
            {
                "Total Return": "{:.2%}",
                "Annualized Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
            }
        )
        .background_gradient(
            cmap="YlGn",
            subset=[
                "Total Return",
                "Annualized Return",
                "Sharpe",
            ],
            vmin=-1,
            vmax=1,
        )
        .background_gradient(
            cmap="OrRd",
            subset=[
                "Volatility"
            ],
        )
    )

    return styled


# ============================================================
# CORRELATION RELATIONSHIPS
# ============================================================

def _correlation_relationships(
    correlation: pd.DataFrame,
) -> dict:

    if (
        correlation.empty
        or len(correlation.columns) < 2
    ):

        return {
            "highest_positive": None,
            "strongest_negative": None,
            "weakest": None,
        }

    matrix = correlation.copy()

    pairs = []

    columns = list(
        matrix.columns
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns),
        ):

            value = _safe_float(
                matrix.iloc[i, j]
            )

            if np.isnan(value):
                continue

            pairs.append(
                (
                    columns[i],
                    columns[j],
                    value,
                )
            )

    if not pairs:

        return {
            "highest_positive": None,
            "strongest_negative": None,
            "weakest": None,
        }

    highest_positive = max(
        pairs,
        key=lambda x: x[2],
    )

    strongest_negative = min(
        pairs,
        key=lambda x: x[2],
    )

    weakest = min(
        pairs,
        key=lambda x: abs(x[2]),
    )

    return {
        "highest_positive": highest_positive,
        "strongest_negative": strongest_negative,
        "weakest": weakest,
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "_safe_float",
    "_format_percent",
    "_format_sharpe",
    "_normalize_dates",
    "_analysis_window",
    "_get_price_matrix",
    "_get_returns",
    "_annualized_return",
    "_total_return",
    "_annualized_volatility",
    "_sharpe_ratio",
    "_maximum_drawdown",
    "_calculate_sector_returns",
    "_market_metrics",
    "_pastel_sector_style",
    "_pastel_stock_style",
    "_correlation_relationships",
    "TRADING_DAYS",
    "PERIOD_OPTIONS",
]
