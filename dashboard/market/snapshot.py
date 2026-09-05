from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


TRADING_DAYS = 252

# Minimum observations required before presenting
# correlation relationships as meaningful.
MIN_CORRELATION_OBSERVATIONS = 40

# Minimum observations required before assigning
# Bullish / Bearish / Mixed market regimes.
MIN_REGIME_OBSERVATIONS = 40


# ============================================================
# FORMATTING
# ============================================================

def _format_pct(value) -> str:

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.2%}"


def _format_number(value) -> str:

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.2f}"


# ============================================================
# PERIOD RETURN
# ============================================================

def _period_return(
    returns,
) -> float:

    if returns is None:
        return np.nan

    series = (
        pd.Series(returns)
        .dropna()
    )

    if series.empty:
        return np.nan

    return float(
        (1.0 + series).prod() - 1.0
    )


# ============================================================
# MARKET REGIME
# ============================================================

def _market_regime(
    market_return,
    market_returns,
    sector_returns,
) -> str:
    """
    Market regime uses:
    - selected-period market return
    - recent trend
    - sector breadth

    Very short samples are classified as Short-Term
    rather than being given a confident Bullish/Bearish label.
    """

    market_series = (
        pd.Series(
            market_returns
        )
        .dropna()
    )

    # --------------------------------------------------------
    # SHORT SAMPLE
    # --------------------------------------------------------

    if (
        len(market_series)
        < MIN_REGIME_OBSERVATIONS
    ):

        return "Short-Term"

    market_return = (
        float(market_return)
        if market_return is not None
        and not pd.isna(market_return)
        else np.nan
    )

    # --------------------------------------------------------
    # RECENT TREND
    # --------------------------------------------------------

    short_window = min(
        10,
        len(market_series),
    )

    recent_returns = (
        market_series.tail(
            short_window
        )
    )

    trend_return = _period_return(
        recent_returns
    )

    # --------------------------------------------------------
    # SECTOR BREADTH
    # --------------------------------------------------------

    positive_sectors = 0
    negative_sectors = 0
    total_sectors = 0

    if (
        sector_returns is not None
        and not sector_returns.empty
    ):

        for sector in sector_returns.columns:

            sector_return = _period_return(
                sector_returns[sector]
            )

            if pd.isna(sector_return):
                continue

            total_sectors += 1

            if sector_return > 0:
                positive_sectors += 1

            elif sector_return < 0:
                negative_sectors += 1

    if total_sectors > 0:

        positive_breadth = (
            positive_sectors
            / total_sectors
        )

        negative_breadth = (
            negative_sectors
            / total_sectors
        )

    else:

        positive_breadth = np.nan
        negative_breadth = np.nan

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    bullish_score = 0
    bearish_score = 0

    if not pd.isna(market_return):

        if market_return > 0:
            bullish_score += 1

        elif market_return < 0:
            bearish_score += 1

    if not pd.isna(trend_return):

        if trend_return > 0:
            bullish_score += 1

        elif trend_return < 0:
            bearish_score += 1

    if not pd.isna(positive_breadth):

        if positive_breadth > 0.50:
            bullish_score += 1

        elif positive_breadth < 0.50:
            bearish_score += 1

    # --------------------------------------------------------
    # FINAL REGIME
    # --------------------------------------------------------

    if bullish_score >= 3:

        return "Bullish"

    if bearish_score >= 3:

        return "Bearish"

    return "Mixed"


# ============================================================
# MARKET LEADERS
# ============================================================

def _get_market_leaders(
    metrics,
    sector_returns,
    market_returns,
    correlation,
):

    result = {
        "leading_sector": None,
        "leading_sector_return": np.nan,
        "weakest_sector": None,
        "weakest_sector_return": np.nan,
    }

    sector_period_returns = {}

    if (
        sector_returns is not None
        and not sector_returns.empty
    ):

        for sector in sector_returns.columns:

            returns = (
                sector_returns[sector]
                .dropna()
            )

            if returns.empty:
                continue

            sector_period_returns[
                sector
            ] = _period_return(
                returns
            )

    if sector_period_returns:

        sector_series = (
            pd.Series(
                sector_period_returns,
                dtype=float,
            )
            .dropna()
        )

        if not sector_series.empty:

            leading_sector = (
                sector_series.idxmax()
            )

            weakest_sector = (
                sector_series.idxmin()
            )

            result[
                "leading_sector"
            ] = leading_sector

            result[
                "leading_sector_return"
            ] = float(
                sector_series.loc[
                    leading_sector
                ]
            )

            result[
                "weakest_sector"
            ] = weakest_sector

            result[
                "weakest_sector_return"
            ] = float(
                sector_series.loc[
                    weakest_sector
                ]
            )

    return result


# ============================================================
# TOP STOCK
# ============================================================

def _get_top_stock(
    stock_performance,
):

    if (
        stock_performance is None
        or stock_performance.empty
    ):

        return None, np.nan

    symbol_column = None

    if "Symbol" in stock_performance.columns:

        symbol_column = "Symbol"

    elif "Ticker" in stock_performance.columns:

        symbol_column = "Ticker"

    if (
        symbol_column is None
        or "Total Return"
        not in stock_performance.columns
    ):

        return None, np.nan

    data = stock_performance[
        [
            symbol_column,
            "Total Return",
        ]
    ].copy()

    data = data.dropna(
        subset=[
            symbol_column,
            "Total Return",
        ]
    )

    if data.empty:
        return None, np.nan

    positive = data[
        data["Total Return"] > 0
    ]

    if positive.empty:
        return None, np.nan

    row = positive.loc[
        positive["Total Return"].idxmax()
    ]

    return (
        str(row[symbol_column]),
        float(row["Total Return"]),
    )


# ============================================================
# CORRELATION LABEL
# ============================================================

def _correlation_label(
    value,
) -> str:

    if value is None or pd.isna(value):

        return "No relationship"

    value = float(value)

    if value >= 0.70:
        return "High positive"

    if value >= 0.30:
        return "Moderate positive"

    if value > -0.10:
        return "Near zero"

    if value > -0.70:
        return "Moderate negative"

    return "High negative"


# ============================================================
# CORRELATION RELATIONSHIPS
# ============================================================

def _correlation_relationships(
    correlation,
    observation_count,
):

    result = {
        "highest_positive": None,
        "strongest_negative": None,
        "weakest": None,
        "sufficient": (
            observation_count
            >= MIN_CORRELATION_OBSERVATIONS
        ),
    }

    # --------------------------------------------------------
    # SHORT SAMPLE
    # --------------------------------------------------------

    if (
        observation_count
        < MIN_CORRELATION_OBSERVATIONS
    ):

        return result

    if (
        correlation is None
        or correlation.empty
        or correlation.shape[1] < 2
    ):

        return result

    sectors = list(
        correlation.columns
    )

    relationships = []

    for i in range(
        len(sectors)
    ):

        for j in range(
            i + 1,
            len(sectors),
        ):

            first = sectors[i]
            second = sectors[j]

            value = correlation.loc[
                first,
                second,
            ]

            if pd.isna(value):
                continue

            value = float(value)

            relationships.append(
                {
                    "first": first,
                    "second": second,
                    "value": value,
                    "label": _correlation_label(
                        value
                    ),
                }
            )

    if not relationships:
        return result

    positive = [
        x
        for x in relationships
        if x["value"] > 0
    ]

    negative = [
        x
        for x in relationships
        if x["value"] < 0
    ]

    if positive:

        result[
            "highest_positive"
        ] = max(
            positive,
            key=lambda x: x["value"],
        )

    if negative:

        result[
            "strongest_negative"
        ] = min(
            negative,
            key=lambda x: x["value"],
        )

    result[
        "weakest"
    ] = min(
        relationships,
        key=lambda x: abs(
            x["value"]
        ),
    )

    return result


# ============================================================
# MARKET SNAPSHOT
# ============================================================

def _render_market_snapshot(
    metrics,
    sector_returns,
    market_returns,
    correlation,
    stock_performance=None,
):

    # ========================================================
    # MARKET RETURN
    # ========================================================

    market_return = _period_return(
        market_returns
    )

    # ========================================================
    # AVERAGE SECTOR RETURN
    # ========================================================

    sector_period_returns = []

    if (
        sector_returns is not None
        and not sector_returns.empty
    ):

        for sector in sector_returns.columns:

            returns = (
                sector_returns[sector]
                .dropna()
            )

            if returns.empty:
                continue

            sector_period_returns.append(
                _period_return(
                    returns
                )
            )

    if sector_period_returns:

        avg_sector_return = float(
            np.mean(
                sector_period_returns
            )
        )

    else:

        avg_sector_return = np.nan

    # ========================================================
    # MARKET SERIES
    # ========================================================

    market_series = (
        pd.Series(
            market_returns
        )
        .dropna()
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    if len(market_series) >= 2:

        volatility = float(
            market_series.std()
            * np.sqrt(TRADING_DAYS)
        )

    else:

        volatility = np.nan

    # ========================================================
    # SHARPE
    # ========================================================

    if (
        len(market_series) >= 2
        and market_series.std() > 0
    ):

        sharpe = float(
            market_series.mean()
            / market_series.std()
            * np.sqrt(TRADING_DAYS)
        )

    else:

        sharpe = np.nan

    # ========================================================
    # MAX DRAWDOWN
    # ========================================================

    if not market_series.empty:

        wealth = (
            1.0 + market_series
        ).cumprod()

        drawdown = (
            wealth
            / wealth.cummax()
            - 1.0
        )

        max_drawdown = float(
            drawdown.min()
        )

    else:

        max_drawdown = np.nan

    # ========================================================
    # REGIME
    # ========================================================

    regime = _market_regime(
        market_return=market_return,
        market_returns=market_series,
        sector_returns=sector_returns,
    )

    # ========================================================
    # LEADERS
    # ========================================================

    leaders = _get_market_leaders(
        metrics=metrics,
        sector_returns=sector_returns,
        market_returns=market_returns,
        correlation=correlation,
    )

    # ========================================================
    # TOP STOCK
    # ========================================================

    top_stock, top_stock_return = (
        _get_top_stock(
            stock_performance
        )
    )

    # ========================================================
    # CORRELATION RELATIONSHIPS
    # ========================================================

    relationships = (
        _correlation_relationships(
            correlation=correlation,
            observation_count=len(
                market_series
            ),
        )
    )

    # ========================================================
    # CORE METRICS
    # ========================================================

    cols = st.columns(6)

    cols[0].caption("Regime")
    cols[0].markdown(
        f"**{regime}**"
    )

    cols[1].caption("Return")
    cols[1].markdown(
        f"**{_format_pct(market_return)}**"
    )

    cols[2].caption(
        "Avg Sector Return"
    )
    cols[2].markdown(
        f"**{_format_pct(avg_sector_return)}**"
    )

    cols[3].caption("Volatility")
    cols[3].markdown(
        f"**{_format_pct(volatility)}**"
    )

    cols[4].caption("Sharpe")
    cols[4].markdown(
        f"**{_format_number(sharpe)}**"
    )

    cols[5].caption("Max Drawdown")
    cols[5].markdown(
        f"**{_format_pct(max_drawdown)}**"
    )

    # ========================================================
    # LEADERS & RELATIONSHIPS
    # ========================================================

    leader_cols = st.columns(5)

    # ========================================================
    # LEADING SECTOR
    # ========================================================

    leader_cols[0].caption(
        "Leading Sector"
    )

    if leaders["leading_sector"] is not None:

        leader_cols[0].markdown(
            f"**{leaders['leading_sector']}**"
        )

        leader_cols[0].caption(
            _format_pct(
                leaders[
                    "leading_sector_return"
                ]
            )
        )

    else:

        leader_cols[0].markdown(
            "**—**"
        )

    # ========================================================
    # WEAKEST SECTOR
    # ========================================================

    leader_cols[1].caption(
        "Weakest Sector"
    )

    if leaders["weakest_sector"] is not None:

        leader_cols[1].markdown(
            f"**{leaders['weakest_sector']}**"
        )

        leader_cols[1].caption(
            _format_pct(
                leaders[
                    "weakest_sector_return"
                ]
            )
        )

    else:

        leader_cols[1].markdown(
            "**—**"
        )

    # ========================================================
    # TOP STOCK
    # ========================================================

    leader_cols[2].caption(
        "Top Stock"
    )

    if top_stock is not None:

        leader_cols[2].markdown(
            f"**{top_stock}**"
        )

        leader_cols[2].caption(
            _format_pct(
                top_stock_return
            )
        )

    else:

        leader_cols[2].markdown(
            "**—**"
        )

    # ========================================================
    # HIGHEST POSITIVE CORRELATION
    # ========================================================

    leader_cols[3].caption(
        "Highest Positive Correlation"
    )

    if not relationships["sufficient"]:

        leader_cols[3].markdown(
            "**Insufficient observations**"
        )

        leader_cols[3].caption(
            f"Need at least "
            f"{MIN_CORRELATION_OBSERVATIONS} "
            f"observations"
        )

    else:

        highest_positive = (
            relationships[
                "highest_positive"
            ]
        )

        if highest_positive:

            leader_cols[3].markdown(
                f"{highest_positive['first']} ↔ "
                f"{highest_positive['second']} "
                f"({highest_positive['value']:+.2f})"
            )

            leader_cols[3].caption(
                highest_positive["label"]
            )

        else:

            leader_cols[3].markdown(
                "**—**"
            )

    # ========================================================
    # WEAKEST CORRELATION
    # ========================================================

    leader_cols[4].caption(
        "Weakest Correlation"
    )

    if not relationships["sufficient"]:

        leader_cols[4].markdown(
            "**Insufficient observations**"
        )

        leader_cols[4].caption(
            f"Need at least "
            f"{MIN_CORRELATION_OBSERVATIONS} "
            f"observations"
        )

    else:

        weakest = relationships[
            "weakest"
        ]

        if weakest:

            leader_cols[4].markdown(
                f"{weakest['first']} ↔ "
                f"{weakest['second']} "
                f"({weakest['value']:+.2f})"
            )

            leader_cols[4].caption(
                weakest["label"]
            )

        else:

            leader_cols[4].markdown(
                "**—**"
            )

    # ========================================================
    # STRONGEST NEGATIVE
    # ========================================================

    if relationships["sufficient"]:

        strongest_negative = (
            relationships[
                "strongest_negative"
            ]
        )

        if strongest_negative:

            st.caption(
                "Strongest negative correlation: "
                f"{strongest_negative['first']} ↔ "
                f"{strongest_negative['second']} "
                f"({strongest_negative['value']:+.2f}) "
                f"• {strongest_negative['label']}"
            )

    else:

        st.caption(
            "Correlation relationships are hidden for "
            f"short windows with fewer than "
            f"{MIN_CORRELATION_OBSERVATIONS} observations."
        )


# ============================================================
# PUBLIC COMPATIBILITY
# ============================================================

render_market_snapshot = (
    _render_market_snapshot
)
