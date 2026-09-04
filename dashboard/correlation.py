import numpy as np
import pandas as pd
import streamlit as st

from src.features.returns import create_return_matrix


# ============================================================
# HELPERS
# ============================================================

def _symbol_name(ticker: str) -> str:
    return ticker.replace(".NS", "")


def _prepare_returns(data: pd.DataFrame) -> pd.DataFrame:

    returns = create_return_matrix(data)

    if returns.empty:
        raise ValueError(
            "Unable to calculate return matrix."
        )

    return returns


# ============================================================
# CORRELATION MATRIX
# ============================================================

def render_correlation_matrix(
    data: pd.DataFrame,
):

    st.subheader("Correlation Matrix")

    st.caption(
        "Pearson correlation of daily returns "
        "across the selected assets."
    )

    try:

        returns = _prepare_returns(data)

        correlation = returns.corr()

        display_matrix = correlation.copy()

        display_matrix.index = [
            _symbol_name(x)
            for x in display_matrix.index
        ]

        display_matrix.columns = [
            _symbol_name(x)
            for x in display_matrix.columns
        ]

        st.dataframe(
            display_matrix.style.format("{:.2f}"),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Unable to calculate correlation matrix: {e}"
        )


# ============================================================
# ROLLING CORRELATION
# ============================================================

def render_rolling_correlation(
    data: pd.DataFrame,
):

    st.subheader("Rolling Correlation")

    tickers = sorted(
        data["Ticker"].unique().tolist()
    )

    if len(tickers) < 2:

        st.info(
            "Select at least two stocks "
            "for correlation analysis."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        ticker_a = st.selectbox(
            "Asset A",
            tickers,
            format_func=_symbol_name,
            key="rolling_corr_asset_a",
        )

    remaining = [
        ticker
        for ticker in tickers
        if ticker != ticker_a
    ]

    with col2:

        ticker_b = st.selectbox(
            "Asset B",
            remaining,
            format_func=_symbol_name,
            key="rolling_corr_asset_b",
        )

    with col3:

        window = st.select_slider(
            "Window",
            options=[20, 60, 90, 180, 252],
            value=90,
            key="rolling_corr_window",
        )

    returns = _prepare_returns(data)

    pair = returns[
        [ticker_a, ticker_b]
    ].dropna()

    if len(pair) < window:

        st.warning(
            f"Not enough observations for a "
            f"{window}-day rolling correlation."
        )

        return

    rolling = (
        pair[ticker_a]
        .rolling(window)
        .corr(pair[ticker_b])
        .dropna()
    )

    st.line_chart(
        rolling.to_frame("Correlation"),
        use_container_width=True,
        y_label="Correlation",
    )

    current = rolling.iloc[-1]
    average = rolling.mean()
    minimum = rolling.min()
    maximum = rolling.max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Current", f"{current:.2f}")
    c2.metric("Average", f"{average:.2f}")
    c3.metric("Minimum", f"{minimum:.2f}")
    c4.metric("Maximum", f"{maximum:.2f}")


# ============================================================
# DIVERSIFICATION RANKING
# ============================================================

def calculate_diversification_ranking(
    data: pd.DataFrame,
) -> pd.DataFrame:

    returns = _prepare_returns(data)

    correlation = returns.corr()

    rows = []

    for ticker in correlation.index:

        other_correlations = (
            correlation
            .loc[ticker]
            .drop(index=ticker)
        )

        if other_correlations.empty:

            score = np.nan

        else:

            score = (
                other_correlations
                .abs()
                .mean()
            )

        rows.append(
            {
                "Ticker": ticker,
                "Average Absolute Correlation":
                    score,
            }
        )

    ranking = pd.DataFrame(rows)

    ranking = ranking.sort_values(
        "Average Absolute Correlation",
        ascending=True,
    ).reset_index(drop=True)

    ranking["Diversification Rank"] = (
        np.arange(1, len(ranking) + 1)
    )

    return ranking


def render_diversification_ranking(
    data: pd.DataFrame,
):

    st.subheader("Diversification Ranking")

    st.caption(
        "Lower average absolute correlation "
        "indicates more independent return behavior."
    )

    try:

        ranking = calculate_diversification_ranking(
            data
        )

        display = ranking.copy()

        display["Asset"] = (
            display["Ticker"].map(_symbol_name)
        )

        display = display[
            [
                "Diversification Rank",
                "Asset",
                "Average Absolute Correlation",
            ]
        ]

        display = display.rename(
            columns={
                "Average Absolute Correlation":
                    "Avg. |Correlation|",
            }
        )

        st.dataframe(
            display.style.format(
                {
                    "Avg. |Correlation|":
                        "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:

        st.error(
            f"Unable to calculate diversification ranking: {e}"
        )


# ============================================================
# RELATIONSHIP SECTION
# ============================================================

def render_correlation_section(
    data: pd.DataFrame,
):

    st.header(
        "Relationship & Diversification Analysis"
    )

    analysis = st.selectbox(
        "Analysis",
        [
            "Correlation Matrix",
            "Rolling Correlation",
            "Diversification Ranking",
        ],
        key="relationship_analysis_view",
    )

    st.divider()

    if analysis == "Correlation Matrix":

        render_correlation_matrix(data)

    elif analysis == "Rolling Correlation":

        render_rolling_correlation(data)

    elif analysis == "Diversification Ranking":

        render_diversification_ranking(data)