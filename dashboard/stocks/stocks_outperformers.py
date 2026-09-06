from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics.stocks.risk import (
    calculate_stock_volatility,
    calculate_stock_sharpe,
    calculate_stock_drawdown,
)
from src.analytics.stocks.performance import calculate_win_rate


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytics"
    / "stock_daily_returns.parquet"
)


@st.cache_data
def load_stock_daily_returns() -> pd.DataFrame:
    if not STOCK_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Precalculated stock returns file not found: "
            f"{STOCK_RETURNS_FILE}"
        )

    data = pd.read_parquet(STOCK_RETURNS_FILE)

    data["Date"] = pd.to_datetime(data["Date"])
    data["Return"] = pd.to_numeric(
        data["Return"],
        errors="coerce",
    )

    return data


def _calculate_period_returns(
    returns: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    data = returns[
        (returns["Date"] >= start)
        & (returns["Date"] <= end)
    ].copy()

    data = data.dropna(subset=["Return"])

    if data.empty:
        return pd.DataFrame(
            columns=["Ticker", "Return"]
        )

    result = (
        data.groupby("Ticker")["Return"]
        .agg(lambda x: (1 + x).prod() - 1)
        .reset_index()
    )

    return result


def _calculate_stock_stats(
    returns: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:

    rows = []

    for ticker in tickers:

        stock_returns = (
            returns[
                returns["Ticker"] == ticker
            ]["Return"]
            .dropna()
        )

        if stock_returns.empty:
            continue

        volatility = calculate_stock_volatility(
            stock_returns
        )

        sharpe = calculate_stock_sharpe(
            stock_returns
        )

        drawdown = calculate_stock_drawdown(
            stock_returns
        )

        if isinstance(drawdown, pd.Series):
            max_drawdown = drawdown.min()
        elif isinstance(drawdown, tuple):
            max_drawdown = drawdown[0]
        else:
            max_drawdown = drawdown

        rows.append(
            {
                "Ticker": ticker,
                "Volatility": volatility,
                "Sharpe": sharpe,
                "Max Drawdown": max_drawdown,
            }
        )

    return pd.DataFrame(rows)


def calculate_outperformers(
    start_date,
    end_date,
) -> pd.DataFrame:

    returns = load_stock_daily_returns()

    period_returns = _calculate_period_returns(
        returns,
        start_date,
        end_date,
    )

    if period_returns.empty:
        return pd.DataFrame()

    benchmark = period_returns[
        period_returns["Ticker"] == "^NSEI"
    ]

    if benchmark.empty:
        return pd.DataFrame()

    nifty50_return = benchmark["Return"].iloc[0]

    stocks = period_returns[
        period_returns["Ticker"] != "^NSEI"
    ].copy()

    stocks["NIFTY 50 Return"] = nifty50_return

    stocks["Excess Return"] = (
        stocks["Return"]
        - nifty50_return
    )

    stocks = stocks[
        stocks["Excess Return"] > 0
    ].copy()

    if stocks.empty:
        return pd.DataFrame()

    stock_tickers = stocks["Ticker"].tolist()

    stats = _calculate_stock_stats(
        returns[
            (returns["Date"] >= pd.Timestamp(start_date))
            & (returns["Date"] <= pd.Timestamp(end_date))
        ],
        stock_tickers,
    )

    stocks = stocks.merge(
        stats,
        on="Ticker",
        how="left",
    )

    stocks = stocks.rename(
        columns={
            "Return": "Stock Return",
        }
    )

    stocks = stocks.sort_values(
        "Excess Return",
        ascending=False,
    ).reset_index(drop=True)

    return stocks[
        [
            "Ticker",
            "Stock Return",
            "NIFTY 50 Return",
            "Excess Return",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
        ]
    ]


def render_stock_outperformers(
    market_data,
    start_date,
    end_date,
) -> None:

    st.subheader("NIFTY 50 Outperformers")

    returns = load_stock_daily_returns()

    period_returns = _calculate_period_returns(
        returns,
        start_date,
        end_date,
    )

    if period_returns.empty:
        st.warning(
            "No return data available for the selected period."
        )
        return

    benchmark = period_returns[
        period_returns["Ticker"] == "^NSEI"
    ]

    if benchmark.empty:
        st.warning(
            "NIFTY 50 benchmark data is not available."
        )
        return

    nifty50_return = benchmark["Return"].iloc[0]

    outperformers = calculate_outperformers(
        start_date,
        end_date,
    )

    stock_count = period_returns[
        period_returns["Ticker"] != "^NSEI"
    ]["Ticker"].nunique()

    outperformer_count = len(outperformers)

    row = st.columns(2)

    row[0].metric(
        "Outperforming Stocks",
        f"{outperformer_count} / {stock_count}",
    )

    row[1].metric(
        "NIFTY 50 Return",
        f"{nifty50_return * 100:.2f}%",
    )

    if outperformers.empty:
        st.info(
            "No stocks outperformed the NIFTY 50 "
            "during the selected period."
        )
        return

    display = outperformers.copy()

    display["Stock Return"] = (
        display["Stock Return"] * 100
    ).round(2)

    display["NIFTY 50 Return"] = (
        display["NIFTY 50 Return"] * 100
    ).round(2)

    display["Excess Return"] = (
        display["Excess Return"] * 100
    ).round(2)

    display["Volatility"] = (
        display["Volatility"] * 100
    ).round(2)

    display["Max Drawdown"] = (
        display["Max Drawdown"] * 100
    ).round(2)

    display = display.rename(
        columns={
            "Stock Return": "Stock Return (%)",
            "NIFTY 50 Return": "NIFTY 50 Return (%)",
            "Excess Return": "Excess Return (%)",
            "Volatility": "Volatility (%)",
            "Sharpe": "Sharpe Ratio",
            "Max Drawdown": "Max Drawdown (%)",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )