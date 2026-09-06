from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.stocks.data import _prepare_stock_data

VOLUME_ZSCORE_DAYS = 20


def calculate_stock_liquidity_metrics(
    market_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    data = _prepare_stock_data(
        market_data,
        ticker,
        start_date,
        end_date,
    )

    if "Volume" not in data.columns:
        return {
            "Average Volume": float("nan"),
            "Average Traded Value": float("nan"),
            "Amihud Illiquidity": float("nan"),
            "Volume Z-Score": float("nan"),
        }

    data["TradedValue"] = (
        data["Close"] * data["Volume"]
    )

    data["Return"] = (
        data["Close"].pct_change()
    )

    valid_traded_value = (
        data["TradedValue"] > 0
    )

    average_volume = float(
        data["Volume"].mean()
    )

    average_traded_value = float(
        data.loc[
            valid_traded_value,
            "TradedValue",
        ].mean()
    )

    amihud_values = (
        data.loc[
            valid_traded_value,
            "Return",
        ].abs()
        / data.loc[
            valid_traded_value,
            "TradedValue",
        ]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    amihud = float(
        amihud_values.mean()
    )

    volume_mean = (
        data["Volume"]
        .shift(1)
        .rolling(VOLUME_ZSCORE_DAYS)
        .mean()
        .iloc[-1]
    )

    volume_std = (
        data["Volume"]
        .shift(1)
        .rolling(VOLUME_ZSCORE_DAYS)
        .std(ddof=1)
        .iloc[-1]
    )

    latest_volume = data["Volume"].iloc[-1]

    if (
        pd.isna(volume_mean)
        or pd.isna(volume_std)
        or volume_std == 0
    ):
        volume_zscore = float("nan")
    else:
        volume_zscore = float(
            (latest_volume - volume_mean)
            / volume_std
        )

    return {
        "Average Volume": average_volume,
        "Average Traded Value":
            average_traded_value,
        "Amihud Illiquidity":
            amihud,
        "Volume Z-Score":
            volume_zscore,
    }